"""3단계: 재무 연동 — '결의일 이전에 이미 공시된' 최근 정기보고서만 사용.

핵심: 보고서의 대상기간이 아니라 접수일(rcept_dt) 기준으로 매핑한다.
예) 3월 초 발행 건에 3월 말 공시되는 사업보고서를 붙이면 look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .dart_client import get_json

_PERIODIC_KEYWORDS = {"사업보고서": "11011", "반기보고서": "11012",
                      "분기보고서": None}   # 분기는 1Q/3Q 판별 필요


def _periodic_filings(corp_code: str) -> pd.DataFrame:
    """해당 회사의 정기보고서 접수 이력 (kind='A')."""
    rows = []
    page = 1
    while True:
        data = get_json("list", corp_code=corp_code,
                        bgn_de="20140101", end_de=C.COLLECT_END,
                        pblntf_ty="A", page_no=page, page_count=100)
        items = data.get("list", []) or []
        rows.extend(items)
        if page >= int(data.get("total_page", 1) or 1):
            break
        page += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], format="%Y%m%d")
    return df


def _detect_reprt(report_nm: str) -> tuple[str | None, int | None]:
    """보고서명 → (reprt_code, bsns_year). 예: '분기보고서 (2023.03)' → 11013, 2023"""
    import re
    m = re.search(r"\((\d{4})\.(\d{2})\)", report_nm)
    if not m:
        return None, None
    year, month = int(m.group(1)), int(m.group(2))
    if "사업보고서" in report_nm:
        return "11011", year
    if "반기" in report_nm:
        return "11012", year
    if "분기" in report_nm:
        return ("11013", year) if month <= 4 else ("11014", year)
    return None, None


def _pull_accounts(corp_code: str, bsns_year: int, reprt_code: str) -> dict:
    """fnlttSinglAcntAll에서 연결 우선(없으면 별도)으로 주요 계정 추출."""
    out = {k: np.nan for k in C.FIN_ACCOUNTS}
    for fs_div in ("CFS", "OFS"):
        data = get_json("fnlttSinglAcntAll", corp_code=corp_code,
                        bsns_year=str(bsns_year), reprt_code=reprt_code,
                        fs_div=fs_div)
        items = data.get("list", []) or []
        if not items:
            continue
        acct = pd.DataFrame(items)
        for key, names in C.FIN_ACCOUNTS.items():
            for nm in names:
                hit = acct[acct["account_nm"].str.replace(" ", "")
                           .str.contains(nm, na=False)]
                if len(hit):
                    val = str(hit.iloc[0].get("thstrm_amount", "")).replace(",", "")
                    try:
                        out[key] = float(val)
                    except ValueError:
                        pass
                    break
        out["fs_div"] = fs_div
        return out
    return out


def attach(features: pd.DataFrame) -> pd.DataFrame:
    """이벤트별로 결의일 직전 공시 재무를 붙이고 압박 지표를 계산."""
    rows = []
    for corp_code, g in features.groupby("corp_code"):
        plist = _periodic_filings(corp_code)
        for idx, ev in g.iterrows():
            rec = {"_idx": idx}
            if not plist.empty:
                prior = plist[plist["rcept_dt"] < ev["event_dt"]]
                if len(prior):
                    latest = prior.sort_values("rcept_dt").iloc[-1]
                    reprt, year = _detect_reprt(latest["report_nm"])
                    if reprt and year:
                        rec.update(_pull_accounts(corp_code, year, reprt))
                        rec["fin_rcept_dt"] = latest["rcept_dt"]
                        rec["fin_lag_days"] = (ev["event_dt"]
                                               - latest["rcept_dt"]).days
            rows.append(rec)
    fin = pd.DataFrame(rows).set_index("_idx")
    df = features.join(fin)

    # --- 재무 압박 피처 (결의 시점 기준)
    df["cash_total"] = df[["cash", "st_fin"]].sum(axis=1, min_count=1)
    df["cb_to_cash"] = df["amt_total"] / df["cash_total"]
    df["impairment"] = 1 - (df["equity"] / df["capital"])       # 자본잠식률
    df["is_impaired"] = (df["impairment"] > 0).astype("Int64")
    df["cfo_negative"] = (df["cfo"] < 0).astype("Int64")

    df.to_parquet(C.OUT_DIR / "05_features_fin.parquet", index=False)
    return df
