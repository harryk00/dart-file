"""45_rate_cashflow.py — 질문3(개정): 이자율 스펙트럼 × 자금흐름.

개정 포인트 (사용자 피드백: 0%만이 아니라 다양한 이자율 조합)
--------------------------------------------------------------
1) 표면이자율(coupon)과 만기이자율(YTM)을 '각각' 층화 — 둘은 다른 정보.
2) 고정 경계 대신 데이터 분위수(quartile)로 구간화 → '어느 수준'을 데이터가 답.
3) 표면×만기 '스프레드 유형'을 딜 성격 축으로:
     zero_zero        표면0·만기0        → 순수 전환 베팅
     coupon0_ytm_pos  표면0·만기>0        → 전환 전제('만기 안 감')  ★핵심 가설
     both_pos_flat    표면≈만기 (차<1%p)  → 순수 차입
     both_pos_step    만기>표면+1%p       → 상환 우대 차입
4) 연속값 상관(로지스틱 단일계수 부호)도 병기 — 구간화 손실 보완.
5) 이자율 × 자금흐름(영업CF 부호·런웨이) 2차원 유지.
라벨 L2/L4 병기, test 확인.

사용법: python 45_rate_cashflow.py   (API 호출 없음)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/out")
pd.set_option("display.width", 170)
REPORT: list[str] = []


def log(m=""):
    print(m); REPORT.append(str(m))


def qbin(s: pd.Series, labels) -> pd.Series:
    """양수만 분위수 구간화(0은 별도 취급되므로 제외하고 경계 산출)."""
    pos = s[s > 0]
    if pos.empty:
        return pd.Series(index=s.index, dtype="object")
    qs = np.unique(pos.quantile([0, .25, .5, .75, 1.0]).values)
    if len(qs) < 3:                        # 구간 2개 미만이면 포기
        return pd.Series(index=s.index, dtype="object")
    n_bins = len(qs) - 1
    return pd.cut(s, bins=qs, labels=list(labels)[:n_bins], include_lowest=True)


def table(d, by, label, tgt="L2"):
    d = d[d[tgt].notna()]
    if d.empty:
        return
    b = d[tgt].mean()
    g = (d.groupby(by, observed=True, dropna=False)
           .agg(n=(tgt, "size"), rate=(tgt, "mean"),
                xs_med=(tgt + "_excess", "median"),
                xs_mean=(tgt + "_excess", "mean")))
    g["lift"] = g["rate"] / b
    g = g[g["n"] >= 30].round(3)
    if len(g):
        log(f"\n### {label} [{tgt}]  (base={b:.3f})")
        log(g.to_string())


def main() -> None:
    df = pd.read_parquet(OUT / "07_dataset.parquet")
    cpn = pd.to_numeric(df["coupon"], errors="coerce")
    ytm = pd.to_numeric(df["ytm"], errors="coerce")
    df["coupon_v"], df["ytm_v"] = cpn, ytm
    df["spread"] = ytm - cpn

    # --- 분위수 구간 (0 별도 + 양수 3분위)
    log("이자율 분포(관측):")
    log(f"  표면 coupon: {cpn.describe()[['min','25%','50%','75%','max']].round(2).to_dict()}")
    log(f"  만기 ytm   : {ytm.describe()[['min','25%','50%','75%','max']].round(2).to_dict()}")
    log(f"  0% 비율    : 표면 {(cpn==0).mean():.0%} / 만기 {(ytm==0).mean():.0%}")

    lab4 = ["저", "중저", "중고", "고"]
    df["coupon_bin"] = np.where(cpn == 0, "0%",
                                qbin(cpn, lab4).astype("object"))
    df["ytm_bin"] = np.where(ytm == 0, "0%", qbin(ytm, lab4).astype("object"))
    df["coupon_bin"] = pd.Categorical(df["coupon_bin"], ["0%"] + lab4)
    df["ytm_bin"] = pd.Categorical(df["ytm_bin"], ["0%"] + lab4)

    # --- 스프레드 유형
    df["deal_type"] = np.select(
        [(cpn == 0) & (ytm == 0),
         (cpn == 0) & (ytm > 0),
         (df["spread"].abs() <= 1),
         (df["spread"] > 1)],
        ["전환베팅(0/0)", "전환전제(표0·만+)", "차입(표≈만)", "상환우대(만>표)"],
        default="기타")

    # --- 자금흐름
    df["cfo_sign"] = np.where(df["cfo"].isna(), "결측",
                              np.where(df["cfo"] < 0, "적자", "흑자"))
    burn = df["cfo"].where(df["cfo"] < 0).abs()
    df["runway_x"] = (df["cash_total"].fillna(0) + df["amt_total"]) / burn
    df["runway_bin"] = pd.cut(df["runway_x"], [0, 2, 5, 10, np.inf],
                              labels=["짧음(<2x)", "2~5x", "5~10x", "여유(10x+)"])

    for tgt in ("L2", "L4"):
        log("\n" + "#" * 70)
        log(f"# 라벨 {tgt}")
        log("#" * 70)
        log("\n--- (1) 표면·만기 이자율 각각의 수준별 ---")
        table(df, "coupon_bin", "표면이자율 수준", tgt)
        table(df, "ytm_bin", "만기이자율 수준", tgt)

        log("\n--- (2) 스프레드 유형(딜 성격) ---")
        table(df, "deal_type", "표면×만기 조합 유형", tgt)

        log("\n--- (3) 표면 × 만기 교차 (셀: 달성률/n) ---")
        pt = (df[df[tgt].notna()]
              .pivot_table(index="coupon_bin", columns="ytm_bin", values=tgt,
                           aggfunc=["mean", "size"], observed=True))
        log(pt.round(3).to_string())

        log("\n--- (4) 딜유형 × 자금흐름 ---")
        pt2 = (df[df[tgt].notna()]
               .pivot_table(index="deal_type", columns="cfo_sign", values=tgt,
                            aggfunc=["mean", "size"], observed=True))
        log(pt2.round(3).to_string())
        table(df[df["cfo_sign"] == "적자"], "runway_bin",
              "적자기업: CB가 사준 런웨이 배수", tgt)

        log("\n--- (5) 연속값 방향성 (로지스틱 단일변수 계수 부호) ---")
        _logit_dir(df, tgt)

    log("\n" + "=" * 70)
    log("test 확인: 딜유형 × 자금흐름")
    log("=" * 70)
    te = df[df["split"] == "test"]
    for tgt in ("L2", "L4"):
        table(te, "deal_type", f"[test] 딜 유형", tgt)

    log("\n판정: L2·L4 방향 일치 + test 유지 + n>=200. "
        "런웨이는 보고서기간 혼재로 서열 해석.")
    (OUT / "q3_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log(f"\n저장: {OUT / 'q3_report.txt'}")


def _logit_dir(df, tgt):
    """coupon/ytm/spread 각각의 단일 로지스틱 계수 부호와 유의성(근사)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        log("  (sklearn 없음 — 상관계수로 대체)")
        for c in ("coupon_v", "ytm_v", "spread"):
            d = df[[c, tgt]].dropna()
            if len(d) > 100:
                r = np.corrcoef(d[c], d[tgt])[0, 1]
                log(f"  {c}: corr={r:+.3f}")
        return
    for c in ("coupon_v", "ytm_v", "spread"):
        d = df[[c, tgt]].dropna()
        if len(d) < 200:
            continue
        X = ((d[[c]] - d[[c]].mean()) / d[[c]].std()).values
        y = d[tgt].astype(int).values
        if len(np.unique(y)) < 2:
            continue
        m = LogisticRegression().fit(X, y)
        coef = float(m.coef_[0][0])
        arrow = "↑상승연관" if coef > 0 else "↓하락연관"
        log(f"  {c:9s}: 표준화계수 {coef:+.3f}  {arrow}")


if __name__ == "__main__":
    main()
