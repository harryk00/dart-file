"""1단계: 전환사채권발행결정 전수 수집. (v3)

v3 변경점
---------
[치명 버그 수정] merge 시 filings의 corp_cls를 유지하도록 수정.
  v2는 collapse 과정에서 corp_cls를 흘려 run()의 마지막 줄에서 KeyError.
[폴백 견고화] 이름 기반 티커 매핑을 best-effort로 강등. 실패해도 파이프라인이
  죽지 않고, corpCode.xml로 매핑된 대다수 기업으로 완주한다. 매핑 실패건은
  03_unmapped.parquet 으로 따로 저장해 사후 진단 가능.
[생존편향 수정 유지] list 스캔에서 corp_cls 필터 없음(KONEX만 제외).
"""
from __future__ import annotations

import re

import pandas as pd

from . import config as C
from .dart_client import get_json, load_corp_map

try:
    from pykrx import stock as krx
except ImportError:
    krx = None


def _month_ranges(bgn: str, end: str):
    dates = pd.date_range(pd.Timestamp(bgn), pd.Timestamp(end), freq="MS")
    for s in dates:
        e = min(s + pd.offsets.MonthEnd(0), pd.Timestamp(C.COLLECT_END))
        yield s.strftime("%Y%m%d"), e.strftime("%Y%m%d")


def scan_filings() -> pd.DataFrame:
    rows = []
    for bgn, end in _month_ranges(C.COLLECT_BGN, C.COLLECT_END):
        page = 1
        while True:
            data = get_json("list", bgn_de=bgn, end_de=end,
                            pblntf_ty="B", page_no=page, page_count=100)
            items = data.get("list", []) or []
            for it in items:
                if "전환사채권발행결정" not in it.get("report_nm", ""):
                    continue
                cls = it.get("corp_cls", "")
                if cls == "N":                       # KONEX 제외
                    continue
                rows.append({
                    "corp_code": it["corp_code"],
                    "corp_name": it["corp_name"],
                    "corp_cls": cls,                 # '현재' 분류 (E=상폐 추정)
                    "rcept_no": it["rcept_no"],
                    "rcept_dt": it["rcept_dt"],
                    "report_nm": it["report_nm"],
                    "is_correction": it["report_nm"].strip().startswith("["),
                })
            total_page = int(data.get("total_page", 1) or 1)
            if page >= total_page:
                break
            page += 1
    df = pd.DataFrame(rows).drop_duplicates("rcept_no")
    df.to_parquet(C.OUT_DIR / "01_filings.parquet", index=False)
    return df


def fetch_terms(filings: pd.DataFrame) -> pd.DataFrame:
    corp_codes = filings["corp_code"].unique()
    rows = []
    for cc in corp_codes:
        data = get_json("cvbdIsDecsn", corp_code=cc,
                        bgn_de=C.COLLECT_BGN, end_de=C.COLLECT_END)
        for it in data.get("list", []) or []:
            it["corp_code"] = cc
            rows.append(it)
    terms = pd.DataFrame(rows)
    if terms.empty:
        raise RuntimeError("cvbdIsDecsn 결과 없음: API 키/기간 확인")
    terms.to_parquet(C.OUT_DIR / "02_terms_raw.parquet", index=False)
    return terms


# ------------------------------------------------------------ 티커 폴백 매핑
def _norm_name(s: str) -> str:
    return re.sub(r"\(주\)|주식회사|\s", "", str(s)).strip()


def _year_name_map(year: int) -> dict:
    """해당 연도 초 상장 전체(상폐 포함)의 (정규화 종목명 → 티커) 맵.

    KRX 회원제 전환 이후 delist 조회가 불안정할 수 있으므로 best-effort.
    실패 시 빈 dict를 캐시해 재시도로 시간 낭비하지 않는다.
    """
    cp = C.CACHE_DIR / f"namemap_{year}.parquet"
    if cp.exists():
        d = pd.read_parquet(cp)
        return dict(zip(d["name"], d["ticker"]))
    out: dict = {}
    if krx is not None:
        for mkt in ("KOSPI", "KOSDAQ"):
            try:
                df = krx.get_market_price_change_by_ticker(
                    f"{year}0102", f"{year}0131", market=mkt, delist=True)
                if df is None or df.empty or "종목명" not in df.columns:
                    continue
                for tkr, nm in df["종목명"].items():
                    nm = _norm_name(nm)
                    if nm:
                        out[nm] = str(tkr).zfill(6)
            except Exception:
                continue
    pd.DataFrame({"name": list(out.keys()), "ticker": list(out.values())}
                 ).to_parquet(cp, index=False)
    return out


def merge_and_dedupe(filings: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
    """접수목록·조건 조인 + 회차 단위 정리. corp_cls·corp_name 유지."""
    # filings의 메타(회차 판별 전 접수단위)를 terms에 결합
    meta = filings[["rcept_no", "corp_name", "corp_cls", "rcept_dt", "is_correction"]]
    df = terms.merge(meta, on="rcept_no", how="inner")
    df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], format="%Y%m%d")

    key = ["corp_code", "bd_tm"] if "bd_tm" in df.columns else ["corp_code", "rcept_no"]
    collapsed = []
    for _, g in df.groupby(key, dropna=False):
        g = g.sort_values("rcept_dt")
        first = g.iloc[0].copy()                 # 조건·시점·메타 모두 최초 공시 기준
        first["event_dt"] = g.iloc[0]["rcept_dt"]
        first["n_corrections"] = int(g["is_correction"].sum())
        collapsed.append(first)
    out = pd.DataFrame(collapsed).reset_index(drop=True)

    # 1차: corpCode.xml 매핑
    corp_map = load_corp_map()
    out = out.merge(corp_map[["corp_code", "stock_code"]], on="corp_code", how="left")

    # 2차: 이름 기반 폴백 (best-effort — 실패해도 진행)
    miss = out["stock_code"].isna()
    n_fallback = 0
    if miss.any() and "corp_name" in out.columns:
        for idx in out.index[miss]:
            nm = _norm_name(out.at[idx, "corp_name"])
            year = int(out.at[idx, "event_dt"].year)
            for y in (year, year + 1, year - 1):
                m = _year_name_map(y)
                if nm in m:
                    out.at[idx, "stock_code"] = m[nm]
                    n_fallback += 1
                    break

    # 매핑 실패건은 따로 저장(사후 진단용)하고 본류에서 제외
    unmapped = out[out["stock_code"].isna()]
    if len(unmapped):
        cols = [c for c in ["corp_code", "corp_name", "corp_cls", "event_dt"]
                if c in unmapped.columns]
        unmapped[cols].to_parquet(C.OUT_DIR / "03_unmapped.parquet", index=False)
    print(f"[collect] 폴백매핑 성공 {n_fallback}건 / "
          f"매핑실패 {len(unmapped)}건(→03_unmapped.parquet, 사후진단)")

    out = out.dropna(subset=["stock_code"]).reset_index(drop=True)
    out.to_parquet(C.OUT_DIR / "03_events.parquet", index=False)
    return out


def run() -> pd.DataFrame:
    filings = scan_filings()
    terms = fetch_terms(filings)
    events = merge_and_dedupe(filings, terms)
    n_e = int((events.get("corp_cls", pd.Series(dtype=str)) == "E").sum())
    print(f"[collect] filings={len(filings)}  terms={len(terms)}  "
          f"events={len(events)}  (현재 상폐추정 corp_cls=E: {n_e}건)")
    return events
