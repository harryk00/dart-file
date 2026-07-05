"""2단계: 발행 '결의 시점' 피처 생성. (v4 — 가설 확장판)

v4 추가 피처
------------
[원문 파싱 확장 — 콜/풋과 같은 다운로드에서 함께 추출]
  n_union          : 원문 내 투자조합 언급 수 (조합 쪼개기 딜 프록시)
  owner_subscriber : 최대주주/대표이사/특수관계인의 직접 인수 정황
  has_lockup       : 보호예수(락업) 언급 여부
[공시 이력 피처 — 추가 API 콜 없음]
  prior_cb_24m     : 직전 24개월 내 동일회사 CB 발행 횟수 (상습 발행)
  is_first_cb      : 수집기간 내 첫 발행 여부
  multi_tranche    : 같은 날 복수 회차 동시 발행 여부

look-ahead 금지 원칙 유지: 발행 결의 시점에 알 수 있던 정보만 피처화.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C
from .dart_client import get_document_xml

NUMERIC_FIELDS = {
    "bd_fta": "amt_total",
    "bd_intr_ex": "coupon",
    "bd_intr_sf": "ytm",
    "cv_prc": "cv_prc",
    "act_mktprcfl_cvprc_lwtrsprc": "floor_prc",
    "cvisstk_cnt": "cv_shares",
    "cvisstk_tisstk_vs": "dilution_pct_disclosed",
    "fdpp_fclt": "use_facility",
    "fdpp_bsninh": "use_bizacq",
    "fdpp_op": "use_working",
    "fdpp_dtrp": "use_debt",
    "fdpp_ocsa": "use_otherstock",
    "fdpp_etc": "use_etc",
}
DATE_FIELDS = {"cvrqpd_bgd": "cv_bgn", "cvrqpd_edd": "cv_end",
               "bd_mtd": "maturity", "pymd": "pay_dt"}


def _num(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = re.sub(r"[,\s원%]", "", str(x))
    if s in ("", "-", "해당없음", "해당사항없음"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _date(x):
    if x is None:
        return pd.NaT
    s = re.sub(r"[^\d]", "", str(x))
    if len(s) < 8:
        return pd.NaT
    return pd.to_datetime(s[:8], format="%Y%m%d", errors="coerce")


# ------------------------------------------------------------------ 원문 파싱
CALL_PAT = re.compile(r"(매도청구권|콜옵션|call\s*option)", re.I)
CALL_OWNER_PAT = re.compile(r"(최대주주|발행회사|회사)[^.\n]{0,40}(매도청구|콜)")
CALL_LIMIT_PAT = re.compile(r"(?:매도청구|콜)[^%\n]{0,120}?(\d{1,3}(?:\.\d+)?)\s*%")
PUT_PAT = re.compile(r"(조기상환청구권|풋옵션|put\s*option)", re.I)
PUT_START_PAT = re.compile(
    r"조기상환[^\n]{0,80}?(?:발행일|납입일)[^\n]{0,20}?(\d{1,2})\s*(?:년|개월)")
UNION_PAT = re.compile(r"[가-힣\w]+조합(?:\s*제?\s*\d+\s*호)?")
OWNER_SUB_PAT = re.compile(
    r"(최대주주|대표이사|특수관계인)[^\n]{0,80}?(배정|인수|청약|취득)")
LOCKUP_PAT = re.compile(r"(보호예수|의무보유|전매제한)")


def parse_doc(rcept_no: str) -> dict:
    """공시 원문에서 콜/풋 + 인수자 구성 정보 추출."""
    empty = {"has_call": np.nan, "has_put": np.nan, "call_owner_side": np.nan,
             "call_limit_pct": np.nan, "put_start_num": np.nan,
             "n_union": np.nan, "owner_subscriber": np.nan, "has_lockup": np.nan}
    try:
        text = get_document_xml(rcept_no)
    except Exception:
        return empty
    if not text:
        return empty
    has_call = bool(CALL_PAT.search(text))
    has_put = bool(PUT_PAT.search(text))
    m_lim = CALL_LIMIT_PAT.search(text) if has_call else None
    m_put = PUT_START_PAT.search(text) if has_put else None
    return {
        "has_call": has_call,
        "has_put": has_put,
        "call_owner_side": bool(CALL_OWNER_PAT.search(text)) if has_call else False,
        "call_limit_pct": _num(m_lim.group(1)) if m_lim else np.nan,
        "put_start_num": _num(m_put.group(1)) if m_put else np.nan,
        "n_union": min(len({re.sub(r"\s", "", m)
                            for m in UNION_PAT.findall(text)}), 30),
        "owner_subscriber": bool(OWNER_SUB_PAT.search(text)),
        "has_lockup": bool(LOCKUP_PAT.search(text)),
    }


# ------------------------------------------------------------------ 피처 빌드
def build(events: pd.DataFrame, parse_docs: bool = True) -> pd.DataFrame:
    df = events.copy()

    for src, dst in NUMERIC_FIELDS.items():
        df[dst] = df[src].map(_num) if src in df.columns else np.nan
    for src, dst in DATE_FIELDS.items():
        df[dst] = df[src].map(_date) if src in df.columns else pd.NaT

    # --- 조건 피처
    df["is_zero_zero"] = ((df["coupon"].fillna(0) == 0) &
                          (df["ytm"].fillna(0) == 0)).astype(int)
    df["refix_floor_ratio"] = df["floor_prc"] / df["cv_prc"]
    df["refix_below_70"] = (df["refix_floor_ratio"] < 0.699).astype("Int64")
    df["days_to_cv_open"] = (df["cv_bgn"] - df["event_dt"]).dt.days
    df["cv_window_days"] = (df["cv_end"] - df["cv_bgn"]).dt.days
    df["tenor_days"] = (df["maturity"] - df["pay_dt"]).dt.days

    # --- 자금 목적 비중
    uses = ["use_facility", "use_bizacq", "use_working",
            "use_debt", "use_otherstock", "use_etc"]
    tot = df[uses].sum(axis=1)
    for u in uses:
        df[u + "_w"] = df[u] / tot.replace(0, np.nan)
    df["use_debt_heavy"] = (df["use_debt_w"].fillna(0) > 0.5).astype(int)
    df["use_ma_heavy"] = ((df["use_otherstock_w"].fillna(0)
                           + df["use_bizacq_w"].fillna(0)) > 0.5).astype(int)

    # --- 희석률
    df["dilution_calc"] = np.where(df["cv_prc"] > 0,
                                   df["amt_total"] / df["cv_prc"], np.nan)
    df["dilution_max_shares"] = np.where(df["floor_prc"] > 0,
                                         df["amt_total"] / df["floor_prc"], np.nan)
    df["shares_out_est"] = df["cv_shares"] / (df["dilution_pct_disclosed"] / 100)
    df["dilution_now"] = df["cv_shares"] / df["shares_out_est"]
    df["dilution_max"] = df["dilution_max_shares"] / df["shares_out_est"]

    # --- 이력 피처 (v4): 상습 발행 / 첫 발행 / 동시 다회차
    df = df.sort_values(["corp_code", "event_dt"]).reset_index(drop=True)
    cum_dil, prior_cnt = [], []
    for _, g in df.groupby("corp_code"):
        for i in range(len(g)):
            t = g.iloc[i]["event_dt"]
            lo = t - pd.Timedelta(days=730)
            win = g[(g["event_dt"] >= lo) & (g["event_dt"] <= t)]
            cum_dil.append(win["dilution_max"].sum())
            prior_cnt.append(int((win["event_dt"] < t).sum()))
    df["cum_dilution_24m"] = cum_dil
    df["prior_cb_24m"] = prior_cnt
    df["is_first_cb"] = (df.groupby("corp_code").cumcount() == 0).astype(int)
    df["multi_tranche"] = (df.groupby(["corp_code", "event_dt"])["rcept_no"]
                             .transform("count") > 1).astype(int)

    # --- 레짐
    df["post_refix_rule"] = (
        df["event_dt"] >= pd.Timestamp(C.REFIX_REGIME_DATE)).astype(int)

    # --- 원문 파싱 (v4: 콜/풋 + 인수자)
    if parse_docs:
        cp = pd.DataFrame([parse_doc(r) for r in df["rcept_no"]], index=df.index)
        df = pd.concat([df, cp], axis=1)
        df["call_and_zero"] = ((df["has_call"] == True) &            # noqa: E712
                               (df["is_zero_zero"] == 1)).astype(int)
        df["union_heavy"] = (df["n_union"].fillna(0) >= 3).astype(int)

    df.to_parquet(C.OUT_DIR / "04_features.parquet", index=False)
    return df
