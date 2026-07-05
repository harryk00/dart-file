"""10_strata.py — 층화표 v2: 모든 표에 L1·L2 병기 (기계적 결합 필터링).

L1(전환가×1.3 달성)은 전환가액과 얽힌 기계적 결합이 있으므로,
전환가와 무관하게 정의된 L2(180일 지수 대비 초과수익)를 항상 병기한다.
판정 규칙: L1과 L2가 같은 방향 → 진짜 후보 / L1만 좋음 → 기계적 결합 의심.

사용법:
    python 10_strata.py
    python 10_strata.py --split train
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/out")
pd.set_option("display.width", 160)
REPORT: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    REPORT.append(str(msg))


TGT, TGT_XS = "L2", "L2_excess"     # --target 옵션으로 교체됨


def rate_table(df: pd.DataFrame, by, label: str) -> None:
    """구간별 n / L1 / L1리프트 / 타깃 / 타깃 초과수익 중앙값."""
    base1 = df["L1"].mean()
    base2 = df[TGT].mean() if TGT in df.columns else np.nan
    agg = {"n": ("L1", "size"), "L1": ("L1", "mean")}
    if TGT in df.columns:
        agg[TGT] = (TGT, "mean")
        agg[TGT + "xs_med"] = (TGT_XS, "median")
    g = (df.groupby(by, dropna=False, observed=True).agg(**agg))
    g["L1_lift"] = g["L1"] / base1
    if TGT in g.columns:
        g[TGT + "_lift"] = g[TGT] / base2
    g = g[g["n"] >= 30].round(3)
    order = [c for c in ["n", "L1", "L1_lift", TGT, TGT + "_lift", TGT + "xs_med"]
             if c in g.columns]
    log(f"\n### {label}  (base: L1={base1:.3f}, {TGT}={base2:.3f})")
    log(g[order].to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, choices=["train", "valid", "test"])
    ap.add_argument("--target", default="L2", choices=["L2", "L4"])
    args = ap.parse_args()
    global TGT, TGT_XS
    TGT, TGT_XS = args.target, args.target + "_excess"

    df = pd.read_parquet(OUT / "07_dataset.parquet")

    log("=" * 70)
    log("0. 표본 건전성 진단")
    log("=" * 70)
    terms = pd.read_parquet(OUT / "02_terms_raw.parquet")
    events = pd.read_parquet(OUT / "03_events.parquet")
    lost = terms["corp_code"].nunique() - events["corp_code"].nunique()
    log(f"조건조회 고유회사: {terms['corp_code'].nunique()}  "
        f"→ 최종 고유회사: {events['corp_code'].nunique()}  (매핑탈락 {lost}곳)")
    if "no_price_data" in df.columns:
        npd = int(df["no_price_data"].fillna(0).sum())
        log(f"no_price_data=1: {npd}건 ({npd/len(df):.1%})")
    log(f"delisted_flag=1: {int(df['delisted_flag'].fillna(0).sum())}건 "
        f"({df['delisted_flag'].fillna(0).mean():.1%})")

    n0 = len(df)
    df = df[df["L1"].notna() & df[TGT].notna()].copy()
    log(f"L1·{TGT} 산출 표본: {len(df)}/{n0}")
    if args.split:
        df = df[df["split"] == args.split]
        log(f"[{args.split}] 구간만: {len(df)}건")

    # 구간화
    df["cv_prem_bin"] = pd.cut(df["cv_premium"],
                               [-np.inf, -0.05, 0.02, 0.10, np.inf],
                               labels=["할인(<-5%)", "시가근접", "5~10%할증", ">10%할증"])
    df["dil_bin"] = pd.cut(df["dilution_max"],
                           [0, 0.05, 0.10, 0.20, np.inf],
                           labels=["<5%", "5~10%", "10~20%", ">20%"])
    df["cash_bin"] = pd.cut(df["cb_to_cash"],
                            [0, 0.5, 1.0, 2.0, np.inf],
                            labels=["여유(<0.5)", "0.5~1", "1~2", "압박(>2)"])
    df["pre20_bin"] = pd.cut(df["ret_pre20"],
                             [-np.inf, -0.05, 0.05, np.inf],
                             labels=["결의전 하락", "보합", "결의전 상승"])
    if "amt_to_mcap" in df.columns:
        df["amt_mcap_bin"] = pd.cut(df["amt_to_mcap"],
                                    [0, 0.03, 0.07, 0.15, np.inf],
                                    labels=["<3%", "3~7%", "7~15%", ">15%"])
    if "days_to_cv_open" in df.columns:
        df["cvopen_bin"] = pd.cut(df["days_to_cv_open"],
                                  [0, 200, 400, np.inf],
                                  labels=["<200일", "표준(1년±)", ">400일"])
    if "prior_cb_24m" in df.columns:
        df["prior_bin"] = pd.cut(df["prior_cb_24m"].astype(float),
                                 [-0.5, 0.5, 1.5, np.inf],
                                 labels=["첫/단독", "1회", "2회+"])
    if "n_union" in df.columns:
        df["union_bin"] = pd.cut(df["n_union"].astype(float),
                                 [-0.5, 0.5, 2.5, np.inf],
                                 labels=["0", "1~2", "3+"])

    log("\n" + "=" * 70)
    log("1. 단변량 층화표 — L1·L2 병기 (같은 방향이어야 진짜)")
    log("=" * 70)
    rate_table(df, "is_zero_zero", "표면0%/만기0% (zero-zero)")
    rate_table(df, "cv_prem_bin", "전환가액 vs 결의일 주가 (원주가 기준)")
    rate_table(df, "dil_bin", "최대 잠재 희석률")
    rate_table(df, "cash_bin", "CB총액/현금성자산")
    rate_table(df, "use_debt_heavy", "채무상환 목적 50% 초과")
    rate_table(df, "refix_below_70", "리픽싱 하한 70% 미만")
    rate_table(df, "post_refix_rule", "상향 리픽싱 의무화 이후")
    rate_table(df, "pre20_bin", "결의 전 20영업일 수익률")
    # --- v3 신규 가설 (H1~H5)
    if "n_corrections" in df.columns:
        rate_table(df, df["n_corrections"].clip(upper=2), "H1. 정정공시 횟수(0/1/2+)")
    if "cvopen_bin" in df.columns:
        rate_table(df, "cvopen_bin", "H2. 전환청구 개시까지 기간")
    if "use_ma_heavy" in df.columns:
        rate_table(df, "use_ma_heavy", "H3. M&A 목적(타법인취득+영업양수) 50% 초과")
    if "prior_bin" in df.columns:
        rate_table(df, "prior_bin", "H4. 직전 24개월 발행 이력 (상습 발행)")
    if "multi_tranche" in df.columns:
        rate_table(df, "multi_tranche", "H4b. 같은 날 복수 회차 동시 발행")
    if "amt_mcap_bin" in df.columns:
        rate_table(df, "amt_mcap_bin", "H5. 발행규모/시가총액")

    # --- 원문 파싱 가설 (콜/풋 + H6 인수자)
    if "has_call" in df.columns and df["has_call"].notna().any():
        rate_table(df, "has_call", "콜옵션 존재")
        rate_table(df, "call_owner_side", "콜옵션 최대주주/회사 귀속")
        if "call_and_zero" in df.columns:
            rate_table(df, "call_and_zero", "콜옵션 × zero-zero 결합")
        if "union_bin" in df.columns:
            rate_table(df, "union_bin", "H6a. 인수 투자조합 수")
        if "owner_subscriber" in df.columns:
            rate_table(df, "owner_subscriber", "H6b. 최대주주/특수관계인 직접 인수")
        if "has_lockup" in df.columns:
            rate_table(df, "has_lockup", "H6c. 보호예수 언급")
    else:
        log("\n(콜/풋·인수자 피처 없음 — 원문 파싱 미실행)")

    log("\n⚠️  다중비교 경고: 표가 15개+ 이므로 우연한 리프트가 반드시 존재.")
    log("   채택 기준: L2 방향 일치 + 레짐 전후 안정 + n>=200 동시 충족,")
    log("   최종 확인은 test 구간에서 1회만 (python 10_strata.py --split test).")

    log("\n" + "=" * 70)
    log("2. 교차 층화표 (셀: L2 달성률 — 기계적 결합 없는 지표 기준)")
    log("=" * 70)
    for cols, name in [(("dil_bin", "is_zero_zero"), "희석률 × zero-zero"),
                       (("cash_bin", "is_zero_zero"), "재무압박 × zero-zero"),
                       (("post_refix_rule", "cv_prem_bin"), "레짐 × 전환가 구간")]:
        log(f"\n### {name}")
        pt = df.pivot_table(index=cols[0], columns=cols[1],
                            values=TGT, aggfunc=["mean", "size"], observed=True)
        log(pt.round(3).to_string())

    log("\n" + "=" * 70)
    log("3. 결측 진단 — 희석률 NaN의 정체")
    log("=" * 70)
    df["dil_missing"] = df["dilution_max"].isna().astype(int)
    rate_table(df, "dil_missing", "희석률 결측 여부 (결측=정보성인지 확인)")
    miss_by_year = (df.assign(yr=df["event_dt"].dt.year)
                      .groupby("yr")["dil_missing"].mean().round(2))
    log("\n연도별 희석률 결측 비율(필드 스키마 변화 추적):")
    log(miss_by_year.to_string())

    (OUT / "strata_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log(f"\n저장: {OUT / 'strata_report.txt'}")


if __name__ == "__main__":
    main()
