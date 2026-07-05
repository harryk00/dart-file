"""4단계: 라벨링. (v3 — 원주가 사용으로 수정주가 왜곡 근본 해결)

배경
----
pykrx의 get_market_ohlcv는 수정주가(adjusted)를 반환한다. 발행 '이후'의
감자·액면병합이 발행 시점 주가를 수백 배 뻥튀기해(예: 오가닉티코스메틱
809원 → 404,500원) cv_premium과 L1을 완전히 왜곡했다.

해결: get_market_ohlcv_by_date(..., adjusted=False)로 '원주가'(그 시점 실제
명목가)를 받는다. 원주가는 CB 계약서의 전환가액과 동일 기준이므로 cv_premium·
L1이 계약 실질과 일치한다. (진단 00_pykrx_probe.py로 809원=정상 확인 완료)

주의
----
- 원주가 조회는 KRX 로그인 필요(KRX_ID/KRX_PW 환경변수).
- 기존 px_ 캐시(수정주가)는 반드시 삭제 후 재실행할 것. 안 그러면 옛 값 로드.
  labels 캐시는 접두어를 pxraw_ 로 바꿔 물리적으로 분리했다.
- 지수(get_index_ohlcv)는 수정주가 개념이 없어 그대로 사용, L2도 무영향.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

try:
    from pykrx import stock as krx
except ImportError:
    krx = None

_PX_CACHE: dict[str, pd.DataFrame] = {}
_IDX_CACHE: dict[str, pd.Series] = {}


def _prices(ticker: str, bgn: str, end: str) -> pd.DataFrame:
    """원주가(adjusted=False) OHLCV. 캐시 접두어 pxraw_ 로 수정주가와 분리."""
    key = f"{ticker}_{bgn}_{end}"
    if key not in _PX_CACHE:
        cp = C.CACHE_DIR / f"pxraw_{key}.parquet"
        if cp.exists():
            _PX_CACHE[key] = pd.read_parquet(cp)
        else:
            df = krx.get_market_ohlcv_by_date(bgn, end, ticker, adjusted=False)
            df.to_parquet(cp)
            _PX_CACHE[key] = df
    return _PX_CACHE[key]


def _mcap_at(ticker: str, t0: pd.Timestamp) -> float:
    """결의일 직전 시가총액(원). 캐시 접두어 mc_."""
    bgn = (t0 - pd.Timedelta(days=14)).strftime("%Y%m%d")
    end = (t0 - pd.Timedelta(days=1)).strftime("%Y%m%d")
    cp = C.CACHE_DIR / f"mc_{ticker}_{end}.parquet"
    try:
        if cp.exists():
            df = pd.read_parquet(cp)
        else:
            df = krx.get_market_cap_by_date(bgn, end, ticker)
            df.to_parquet(cp)
        if df.empty or "시가총액" not in df.columns:
            return np.nan
        return float(df["시가총액"].iloc[-1])
    except Exception:
        return np.nan


def _index(code: str, bgn: str, end: str) -> pd.Series:
    key = f"{code}_{bgn}_{end}"
    if key not in _IDX_CACHE:
        cp = C.CACHE_DIR / f"idx_{key}.parquet"
        if cp.exists():
            _IDX_CACHE[key] = pd.read_parquet(cp)["종가"]
        else:
            df = krx.get_index_ohlcv(bgn, end, code)
            df.to_parquet(cp)
            _IDX_CACHE[key] = df["종가"]
    return _IDX_CACHE[key]


def _one(ev: pd.Series) -> dict:
    t0: pd.Timestamp = ev["event_dt"]
    ticker = str(ev["stock_code"]).zfill(6)
    bgn = (t0 - pd.Timedelta(days=40)).strftime("%Y%m%d")
    end = (t0 + pd.Timedelta(days=C.L1_HORIZON_D + 30)).strftime("%Y%m%d")
    out: dict = {}
    try:
        px = _prices(ticker, bgn, end)                  # 원주가
    except Exception:
        px = pd.DataFrame()
    if px.empty or "종가" not in px.columns:
        out["no_price_data"] = 1
        return out
    out["no_price_data"] = 0
    close = px["종가"].astype(float)
    close = close[close > 0]
    if close.empty:
        out["no_price_data"] = 1
        return out

    # --- 결의 시점 피처 (원주가라 명목가 그대로, 왜곡 없음)
    pre = close[close.index < t0]
    if len(pre):
        p0 = float(pre.iloc[-1])
        out["px_at_event"] = p0
        out["cv_premium"] = ev["cv_prc"] / p0 - 1 if ev.get("cv_prc") else np.nan
        out["ret_pre20"] = p0 / float(pre.iloc[-21]) - 1 if len(pre) > 21 else np.nan

    # --- [v4/H5] 발행규모 / 시가총액
    mcap = _mcap_at(ticker, t0)
    out["mcap_at_event"] = mcap
    amt = ev.get("amt_total")
    out["amt_to_mcap"] = (float(amt) / mcap
                          if amt and mcap and mcap > 0 else np.nan)

    # --- 상폐/거래정지 추정
    horizon_end = t0 + pd.Timedelta(days=C.L1_HORIZON_D)
    obs_cap = min(horizon_end, pd.Timestamp.today() - pd.Timedelta(days=7))
    out["delisted_flag"] = int(close.index.max() < obs_cap - pd.Timedelta(days=90))

    post = close[close.index >= t0]
    if post.empty:
        return out

    # --- L1: 전환청구 가능 구간에서 원주가 기준 전환가×1.30 달성
    cv_bgn, cv_end = ev.get("cv_bgn"), ev.get("cv_end")
    if pd.notna(cv_bgn) and ev.get("cv_prc"):
        w_end = min(pd.Timestamp(cv_end) if pd.notna(cv_end) else horizon_end,
                    horizon_end)
        win = close[(close.index >= cv_bgn) & (close.index <= w_end)]
        target = ev["cv_prc"] * C.L1_TARGET_MULT           # 원주가끼리 비교
        out["L1_days_above"] = int((win >= target).sum())
        out["L1"] = int(out["L1_days_above"] >= C.L1_MIN_DAYS)
        above = win[win >= target]
        out["L1_first_hit_days"] = ((above.index[0] - t0).days
                                    if len(above) else np.nan)

    # --- L2: D+180 지수 대비 초과수익 (수익률이라 원/수정주가 무관, 정상)
    w2 = post[post.index <= t0 + pd.Timedelta(days=C.L2_HORIZON_D)]
    if len(w2) > 1:
        stk_ret = float(w2.iloc[-1]) / float(w2.iloc[0]) - 1
        try:
            idx = _index(C.BENCH.get(ev.get("corp_cls"), "2001"), bgn, end)
            i2 = idx[(idx.index >= w2.index[0]) & (idx.index <= w2.index[-1])]
            idx_ret = float(i2.iloc[-1]) / float(i2.iloc[0]) - 1 if len(i2) > 1 else 0.0
        except Exception:
            idx_ret = 0.0
        out["L2_excess"] = stk_ret - idx_ret
        out["L2"] = int(out["L2_excess"] > 0)

    # --- L3: 경로 (원주가 기준)
    w3 = post[post.index <= t0 + pd.Timedelta(days=C.L3_HORIZON_D)]
    if len(w3) > 1:
        base = float(w3.iloc[0])
        out["L3_mfe"] = float(w3.max()) / base - 1
        out["L3_mae"] = float(w3.min()) / base - 1
        if ev.get("floor_prc") and ev["floor_prc"] > 0:
            out["touched_floor"] = int((w3 <= ev["floor_prc"]).any())

    # --- L4: 이벤트 윈도우 — 전환청구 개시 D-90 진입 → D+90 청산, 지수 대비
    #     '공시 때 워치리스트 → 청구기간 임박 시 진입' 전략의 라벨
    if pd.notna(cv_bgn):
        w4_bgn = pd.Timestamp(cv_bgn) - pd.Timedelta(days=90)
        w4_end = pd.Timestamp(cv_bgn) + pd.Timedelta(days=90)
        w4 = close[(close.index >= w4_bgn) & (close.index <= w4_end)]
        if len(w4) > 20:                      # 윈도우 절반 이상 거래일 확보 시만
            stk4 = float(w4.iloc[-1]) / float(w4.iloc[0]) - 1
            try:
                idx = _index(C.BENCH.get(ev.get("corp_cls"), "2001"), bgn, end)
                i4 = idx[(idx.index >= w4.index[0]) & (idx.index <= w4.index[-1])]
                idx4 = float(i4.iloc[-1]) / float(i4.iloc[0]) - 1 if len(i4) > 1 else 0.0
            except Exception:
                idx4 = 0.0
            out["L4_excess"] = stk4 - idx4
            out["L4"] = int(out["L4_excess"] > 0)
    return out


def build(features: pd.DataFrame) -> pd.DataFrame:
    if krx is None:
        raise ImportError("pip install pykrx 필요")
    lab = pd.DataFrame([_one(ev) for _, ev in features.iterrows()],
                       index=features.index)
    df = pd.concat([features, lab], axis=1)
    df.to_parquet(C.OUT_DIR / "06_labeled.parquet", index=False)
    return df
