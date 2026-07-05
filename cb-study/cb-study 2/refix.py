"""2라운드: '전환가액의조정' 공시를 트리거로 하는 이벤트 데이터셋.

질문
----
(a) 전환가액이 어느 수준(depth)까지 조정되어야 이후 반등이 오는가?
    → refix_depth, at_floor(최저조정가 도달), px_over_newcv 층화
(b) 회사가 재무적으로 '주가를 올려야만 하는' 지점은 어디인가?
    → 조정 시점 재무(현금 대비 CB 부담) × 풋 도래 임박도 상호작용

트리거 시점 정보만 피처화 (look-ahead 차단 원칙 유지).
1라운드 산출물(07_dataset.parquet)과 (corp_code, 회차)로 연결해
최초 전환가·최저조정가·콜/풋 조건을 승계한다.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C
from .dart_client import get_json, get_document_xml
from .labels import _prices, _index          # 원주가·지수 캐시 공유
from .financials import _periodic_filings, _detect_reprt, _pull_accounts

# ---------------------------------------------------------------- 설정
RF_BGN, RF_END = "20160101", "20251231"      # 트리거 수집 범위
RF_TRAIN_END = pd.Timestamp("2022-12-31")    # 시간 분할 (트리거일 기준)
RF_VALID_END = pd.Timestamp("2023-12-31")
R_HORIZONS = (60, 180)                        # 라벨 관측창(일)
RCV_MULT, RCV_MIN_DAYS = 1.30, 3              # 전환유도 성공 정의


def _num(x) -> float:
    if x is None:
        return np.nan
    s = re.sub(r"[,\s원]", "", str(x))
    try:
        v = float(s)
        return v if 50 <= v <= 10_000_000 else np.nan   # 주가 범위 밖 배제
    except ValueError:
        return np.nan


# ---------------------------------------------------------------- 1. 수집
def scan_refix() -> pd.DataFrame:
    """거래소공시(I)에서 '전환가액의조정' 접수 목록 스캔."""
    rows = []
    for s in pd.date_range(RF_BGN, RF_END, freq="MS"):
        e = min(s + pd.offsets.MonthEnd(0), pd.Timestamp(RF_END))
        page = 1
        while True:
            data = get_json("list", bgn_de=s.strftime("%Y%m%d"),
                            end_de=e.strftime("%Y%m%d"),
                            pblntf_ty="I", page_no=page, page_count=100)
            items = data.get("list", []) or []
            for it in items:
                nm = it.get("report_nm", "")
                if "전환가액의조정" not in nm:
                    continue
                if nm.strip().startswith("["):        # 기재정정 제외(원본 유지)
                    continue
                if it.get("corp_cls") == "N":
                    continue
                rows.append({
                    "corp_code": it["corp_code"],
                    "corp_name": it["corp_name"],
                    "corp_cls": it.get("corp_cls", ""),
                    "rcept_no": it["rcept_no"],
                    "rcept_dt": it["rcept_dt"],
                    "report_nm": nm,
                })
            if page >= int(data.get("total_page", 1) or 1):
                break
            page += 1
    df = pd.DataFrame(rows).drop_duplicates("rcept_no")
    df.to_parquet(C.OUT_DIR / "r2_01_filings.parquet", index=False)
    return df


# ---------------------------------------------------------------- 2. 원문 파싱 (v2)
# v1 결함: 표 레이아웃(헤더행→값행)에서 '조정후' 뒤 첫 숫자가 조정'전' 값이라
# 전/후 스왑 → 하향이 상향으로 오분류(73%가 상향으로 잡히는 참사).
# v2: 태그 제거·공백 정규화 후, '조정전'/'조정후' 토큰과 숫자들의 '위치'로 페어링.
TM_PAT = re.compile(r"제\s*(\d+)\s*회")
TAG_RE = re.compile(r"<[^>]+>")
SIGA_PAT = re.compile(r"시가\s*하락")
NUM_RE = re.compile(r"(\d[\d,]{2,})")
_UNIT_BAD = ("년", "월", "일", "회", "차", "%", "주", "호", "부")   # 날짜·회차 오인 차단


def _extract_nums(seg: str) -> list:
    """구간 내 (위치, 값). 날짜/회차/비율 등으로 보이는 숫자는 배제."""
    out = []
    for m in NUM_RE.finditer(seg):
        tail = seg[m.end():m.end() + 2].strip()
        if tail[:1] in _UNIT_BAD:
            continue
        v = _num(m.group(1))
        if not np.isnan(v) and 100 <= v <= 5_000_000:
            out.append((m.start(), v))
    return out


def parse_refix_doc(rcept_no: str, report_nm: str = "") -> dict:
    out = {"bd_tm": np.nan, "adj_before": np.nan, "adj_after": np.nan,
           "reason_siga": np.nan}
    try:
        text = get_document_xml(rcept_no)
    except Exception:
        text = ""
    if not text:
        return out
    clean = re.sub(r"\s+", " ", TAG_RE.sub(" ", text))
    clean = clean.replace("조정 전", "조정전").replace("조정 후", "조정후")

    m = TM_PAT.search(report_nm) or TM_PAT.search(clean[:3000])
    if m:
        out["bd_tm"] = float(m.group(1))
    out["reason_siga"] = bool(SIGA_PAT.search(clean))

    i_pre, i_post = clean.find("조정전"), clean.find("조정후")
    if i_pre == -1 or i_post == -1:
        return out
    start = min(i_pre, i_post)
    win = clean[start:start + 500]
    p_pre, p_post = win.find("조정전"), win.find("조정후")
    nums = _extract_nums(win)
    if not nums:
        return out

    if p_pre < p_post:
        between = [v for p, v in nums if p_pre < p < p_post]
        after = [v for p, v in nums if p > p_post]
        if between and after:
            # 인라인형: 조정전 2,000 ... 조정후 1,540
            out["adj_before"], out["adj_after"] = between[0], after[0]
        elif len(after) >= 2:
            # 표형(헤더행→값행): 조정전(헤더) 조정후(헤더) 2,000 1,540
            out["adj_before"], out["adj_after"] = after[0], after[1]
        elif len(after) == 1:
            out["adj_after"] = after[0]
    else:                                             # 드문 역순 표기
        between = [v for p, v in nums if p_post < p < p_pre]
        after = [v for p, v in nums if p > p_pre]
        if between and after:
            out["adj_after"], out["adj_before"] = between[0], after[0]
    return out


# ---------------------------------------------------------------- 3. 빌드
def build_dataset() -> pd.DataFrame:
    filings = scan_refix()
    print(f"[r2] 전환가액의조정 접수 {len(filings)}건 → 원문 파싱 시작")

    parsed = pd.DataFrame([parse_refix_doc(r.rcept_no, r.report_nm)
                           for r in filings.itertuples()], index=filings.index)
    df = pd.concat([filings, parsed], axis=1)
    df["event_dt"] = pd.to_datetime(df["rcept_dt"], format="%Y%m%d")
    n_ok = int(df["adj_after"].notna().sum())
    print(f"[r2] 조정후가액 파싱 성공 {n_ok}/{len(df)} ({n_ok/len(df):.0%})")

    # --- 1라운드 CB 조건 연결 (corp_code + 회차)
    cb = pd.read_parquet(C.OUT_DIR / "07_dataset.parquet")
    cb_cols = ["corp_code", "bd_tm", "stock_code", "cv_prc", "floor_prc",
               "amt_total", "has_call", "call_owner_side", "is_zero_zero",
               "cv_bgn", "cv_end", "maturity", "put_start_num",
               "event_dt", "corp_cls"]
    cb_cols = [c for c in cb_cols if c in cb.columns]
    cbl = cb[cb_cols].rename(columns={"event_dt": "issue_dt",
                                      "corp_cls": "corp_cls_issue"})
    cbl["bd_tm"] = pd.to_numeric(cbl["bd_tm"], errors="coerce")
    df = df.merge(cbl, on=["corp_code", "bd_tm"], how="left",
                  suffixes=("", "_cb"))
    df["matched"] = df["cv_prc"].notna().astype(int)
    print(f"[r2] 1라운드 CB 매칭 성공 {int(df['matched'].sum())}건 "
          f"(미매칭 {int((1-df['matched']).sum())}건: 2016년 이전 발행 등)")

    # --- 트리거 시점 피처
    df["pair_ok"] = (df["adj_before"].notna() & df["adj_after"].notna()).astype(int)
    df["refix_depth"] = df["adj_after"] / df["cv_prc"]
    df["at_floor"] = ((df["adj_after"] - df["floor_prc"]).abs()
                      / df["floor_prc"] < 0.01).astype("Int64")
    df["adj_pct"] = df["adj_after"] / df["adj_before"] - 1
    # [v2] 방향 분류는 전/후가 '모두' 파싱된 행에서만. 실패행은 <NA>로 남겨
    #      하향 표본에 섞이지 않게 한다 (v1의 치명 버그).
    df["is_up_refix"] = pd.array([np.nan] * len(df), dtype="Int64")
    ok = df["pair_ok"] == 1
    df.loc[ok, "is_up_refix"] = (df.loc[ok, "adj_pct"] > 0).astype("Int64")
    # [v2] 매칭된 CB의 조건과 교차 검증: 조정후가액은 최저조정가~최초전환가
    #      범위(±10%) 안이어야 정상. 밖이면 파싱 의심 플래그.
    with np.errstate(invalid="ignore"):
        df["adj_sane"] = np.where(
            df["cv_prc"].notna() & df["floor_prc"].notna()
            & df["adj_after"].notna(),
            ((df["adj_after"] >= df["floor_prc"] * 0.90)
             & (df["adj_after"] <= df["cv_prc"] * 1.10)).astype(float),
            np.nan)
    df = df.sort_values(["corp_code", "bd_tm", "event_dt"])
    df["refix_seq"] = df.groupby(["corp_code", "bd_tm"]).cumcount() + 1
    df["days_since_issue"] = (df["event_dt"] - df["issue_dt"]).dt.days
    # 풋 개시일 추정 = 발행일 + put_start_num(개월). 풋 임박도(개월)
    put_m = pd.to_numeric(df.get("put_start_num"), errors="coerce")
    df["put_open_dt"] = df["issue_dt"] + pd.to_timedelta(put_m * 30.4, unit="D")
    df["months_to_put"] = (df["put_open_dt"] - df["event_dt"]).dt.days / 30.4
    df["put_within_6m"] = ((df["months_to_put"] >= -1)
                           & (df["months_to_put"] <= 6)).astype("Int64")

    df.to_parquet(C.OUT_DIR / "r2_02_events.parquet", index=False)
    return df


# ---------------------------------------------------------------- 4. 재무 재연동
def attach_financials(df: pd.DataFrame) -> pd.DataFrame:
    """트리거일 직전 공시 재무로 압박 지표 재계산 (look-ahead 차단)."""
    rows = []
    for corp_code, g in df.groupby("corp_code"):
        try:
            plist = _periodic_filings(corp_code)
        except Exception:
            plist = pd.DataFrame()
        for idx, ev in g.iterrows():
            rec = {"_idx": idx}
            if not plist.empty:
                prior = plist[plist["rcept_dt"] < ev["event_dt"]]
                if len(prior):
                    latest = prior.sort_values("rcept_dt").iloc[-1]
                    reprt, year = _detect_reprt(latest["report_nm"])
                    if reprt and year:
                        rec.update(_pull_accounts(corp_code, year, reprt))
            rows.append(rec)
    fin = pd.DataFrame(rows).set_index("_idx")
    df = df.join(fin[[c for c in ("cash", "st_fin") if c in fin.columns]])
    df["cash_total"] = df[["cash", "st_fin"]].sum(axis=1, min_count=1)
    df["cb_to_cash_rf"] = df["amt_total"] / df["cash_total"]
    # (b)의 압박 스코어: 현금부족 × 풋임박 → '주가를 올려야만 하는' 정량화
    df["pressure"] = ((df["cb_to_cash_rf"] > 1).astype(float)
                      * df["put_within_6m"].fillna(0).astype(float))
    df.to_parquet(C.OUT_DIR / "r2_03_features.parquet", index=False)
    return df


# ---------------------------------------------------------------- 5. 라벨
def _one_label(ev: pd.Series) -> dict:
    out: dict = {}
    if pd.isna(ev.get("stock_code")):
        out["no_price_data"] = 1
        return out
    t0 = ev["event_dt"]
    ticker = str(ev["stock_code"]).zfill(6)
    bgn = (t0 - pd.Timedelta(days=10)).strftime("%Y%m%d")
    end = (t0 + pd.Timedelta(days=max(R_HORIZONS) + 30)).strftime("%Y%m%d")
    try:
        px = _prices(ticker, bgn, end)
    except Exception:
        px = pd.DataFrame()
    if px.empty or "종가" not in px.columns:
        out["no_price_data"] = 1
        return out
    out["no_price_data"] = 0
    close = px["종가"].astype(float)
    close = close[close > 0]
    pre = close[close.index < t0]
    post = close[close.index >= t0]
    if pre.empty or post.empty:
        return out
    p0 = float(pre.iloc[-1])
    out["px_at_refix"] = p0
    if ev.get("adj_after") and not pd.isna(ev["adj_after"]):
        out["px_over_newcv"] = p0 / ev["adj_after"]

    for h in R_HORIZONS:
        w = post[post.index <= t0 + pd.Timedelta(days=h)]
        if len(w) > 1:
            stk = float(w.iloc[-1]) / float(w.iloc[0]) - 1
            try:
                idx = _index(C.BENCH.get(ev.get("corp_cls"), "2001"), bgn, end)
                iw = idx[(idx.index >= w.index[0]) & (idx.index <= w.index[-1])]
                ir = float(iw.iloc[-1]) / float(iw.iloc[0]) - 1 if len(iw) > 1 else 0.0
            except Exception:
                ir = 0.0
            out[f"R{h}_excess"] = stk - ir
            out[f"R{h}"] = int(stk - ir > 0)

    # RCV: 조정후가액×1.3 을 180일 내 3영업일 이상 상회 (전환 유도 성공)
    if ev.get("adj_after") and not pd.isna(ev["adj_after"]):
        w = post[post.index <= t0 + pd.Timedelta(days=180)]
        target = ev["adj_after"] * RCV_MULT
        out["RCV_days"] = int((w >= target).sum())
        out["RCV"] = int(out["RCV_days"] >= RCV_MIN_DAYS)
    return out


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    lab = pd.DataFrame([_one_label(ev) for _, ev in df.iterrows()],
                       index=df.index)
    df = pd.concat([df, lab], axis=1)
    # 시간 분할 (트리거일 기준)
    df["split"] = "test"
    df.loc[df["event_dt"] <= RF_VALID_END, "split"] = "valid"
    df.loc[df["event_dt"] <= RF_TRAIN_END, "split"] = "train"
    df.to_parquet(C.OUT_DIR / "r2_04_dataset.parquet", index=False)
    print(df.groupby("split").agg(n=("rcept_no", "count"),
                                  R180_rate=("R180", "mean")).round(3))
    return df


def run() -> pd.DataFrame:
    df = build_dataset()
    df = attach_financials(df)
    df = build_labels(df)
    print(f"[r2] 완료: {len(df)}건 → {C.OUT_DIR / 'r2_04_dataset.parquet'}")
    return df
