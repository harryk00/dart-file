"""20_model.py — v2: 실전 투입 판정용 모델 학습 + 홀드아웃 백테스트.

공정: train 학습 → valid로 조기종료·컷오프 결정 → test(미접촉 기간)에서
      '컷오프 통과 종목만 매수했다면'의 성과를 초과수익률로 백테스트.

실전 투입 판정 기준 (사전 고정 — 결과 보고 바꾸지 말 것):
  [B1] test 시그널 평균 L2_excess > 0        (절대수익 방어)
  [B2] test 시그널 평균 - test 전체 평균 > +10%p (베이스 대비 우위)
  [B3] test 시그널 수 >= 30건                 (통계적 최소 표본)
셋 중 하나라도 미달이면 실전 보류.

매매 룰 가정(v1): 공시 후 진입, 180일 보유 (L2_excess와 동일 정의).
  ※ 진입 시점·보유기간을 바꾸려면 labels.py의 L2 정의를 함께 바꿔 재라벨링.

사용법:
    pip install lightgbm shap scikit-learn matplotlib
    python 20_model.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path("data/out")

# 발행 결의 시점(공시 당일)에 계산 가능한 피처만 — 실전 스코어러와 동일 집합
FEATURES = [
    # 조건
    "coupon", "ytm", "is_zero_zero", "cv_premium",
    "refix_floor_ratio", "refix_below_70",
    "days_to_cv_open", "cv_window_days", "tenor_days",
    "use_working_w", "use_debt_w", "use_otherstock_w",
    "use_debt_heavy", "use_ma_heavy",
    # 희석·규모
    "dilution_pct_disclosed", "dilution_now", "dilution_max",
    "cum_dilution_24m", "amt_to_mcap", "mcap_at_event",
    # 이력
    "prior_cb_24m", "is_first_cb", "multi_tranche", "n_corrections",
    # 재무(결의 전 공시분)
    "cb_to_cash", "impairment", "is_impaired", "cfo_negative", "fin_lag_days",
    # 시세(결의 전)
    "ret_pre20",
    # 레짐
    "post_refix_rule",
    # 원문 파싱
    "has_call", "call_owner_side", "call_limit_pct", "has_put",
    "put_start_num", "call_and_zero",
    "n_union", "union_heavy", "owner_subscriber", "has_lockup",
]
DEFAULT_TARGET = "L2"          # L2: 공시일 진입 / L4: 청구개시 D-90 진입
BT_PRECISION_GOAL = 0.55      # valid에서 목표 정밀도(초과수익 확률)
REPORT: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    REPORT.append(str(msg))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=DEFAULT_TARGET, choices=["L2", "L4"],
                    help="L2=공시일 진입 180일 / L4=청구개시 D-90 진입 D+90 청산")
    args = ap.parse_args()
    global TARGET, XS
    TARGET, XS = args.target, args.target + "_excess"
    df = pd.read_parquet(OUT / "07_dataset.parquet")
    df = df[df[TARGET].notna() & df[XS].notna()].copy()
    for c in FEATURES:
        if c in df.columns and df[c].dtype == bool:
            df[c] = df[c].astype(float)

    feats = [f for f in FEATURES if f in df.columns
             and df[f].notna().mean() > 0.3]
    log(f"타깃={TARGET} ({XS}) / 사용 피처 {len(feats)}개")
    log(str(feats))

    X = df[feats].astype(float)
    y = df[TARGET].astype(int)
    tr, va, te = (df["split"] == s for s in ("train", "valid", "test"))
    log(f"\ntrain={tr.sum()}  valid={va.sum()}  test={te.sum()}")

    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03,
        num_leaves=15, min_child_samples=40,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1,
    )
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    log(f"best_iteration = {model.best_iteration_}")

    def _eval(mask, name):
        p = model.predict_proba(X[mask])[:, 1]
        log(f"[{name}] base={y[mask].mean():.3f}  "
            f"PR-AUC={average_precision_score(y[mask], p):.3f}  "
            f"ROC-AUC={roc_auc_score(y[mask], p):.3f}")
        return p

    log("")
    _eval(tr, "train")
    p_va = _eval(va, "valid")

    # ---- 컷오프: valid에서 '통과군 평균 초과수익 최대' 관점으로 탐색
    log("\nvalid 컷오프 스캔 (통과군 평균 L2_excess / 정밀도 / 건수):")
    xs_va = df.loc[va, XS].to_numpy()
    best_cut, best_score = 0.5, -np.inf
    for cut in np.arange(0.35, 0.71, 0.05):
        pick = p_va >= cut
        if pick.sum() < 20:
            continue
        mean_xs = xs_va[pick].mean()
        prec = y[va][pick].mean()
        log(f"  cut={cut:.2f}: n={int(pick.sum()):4d}  "
            f"평균초과수익={mean_xs:+.3f}  P(초과수익)={prec:.3f}")
        if prec >= BT_PRECISION_GOAL and mean_xs > best_score:
            best_cut, best_score = float(cut), mean_xs
    if best_score == -np.inf:                     # 목표 정밀도 미달 시 차선
        best_cut = 0.55
        log(f"  (정밀도 {BT_PRECISION_GOAL} 달성 컷오프 없음 → 기본 {best_cut})")
    log(f"채택 컷오프 = {best_cut:.2f}")

    # ---------------------------------------------------- 홀드아웃 백테스트
    log("\n" + "=" * 66)
    log("홀드아웃 백테스트 (test: 모델 미접촉 기간, 1회만 평가)")
    log("=" * 66)
    p_te = _eval(te, "test")
    xs_te = df.loc[te, XS].to_numpy()
    pick = p_te >= best_cut

    n_sig = int(pick.sum())
    base_xs = xs_te.mean()
    log(f"\n전체 test CB {int(te.sum())}건: 평균 초과수익 {base_xs:+.3f} "
        f"(중앙값 {np.median(xs_te):+.3f})")
    if n_sig:
        sig_xs = xs_te[pick]
        log(f"시그널 {n_sig}건: 평균 초과수익 {sig_xs.mean():+.3f} "
            f"(중앙값 {np.median(sig_xs):+.3f})  "
            f"승률(지수 대비 +){(sig_xs > 0).mean():.3f}")
        # 연도(반기)별 안정성
        sub = df.loc[te].loc[pick]
        by_half = (sub.assign(half=sub["event_dt"].dt.to_period("Q"))
                     .groupby("half")[XS]
                     .agg(["count", "mean"]).round(3))
        log("\n분기별 시그널 성과(집중 리스크 확인):")
        log(by_half.to_string())

        # ---- 사전 고정 판정
        b1 = sig_xs.mean() > 0
        b2 = (sig_xs.mean() - base_xs) > 0.10
        b3 = n_sig >= 30
        log(f"\n[판정] B1 절대수익>0: {'통과' if b1 else '미달'} "
            f"({sig_xs.mean():+.3f})")
        log(f"[판정] B2 베이스+10%p: {'통과' if b2 else '미달'} "
            f"(격차 {sig_xs.mean()-base_xs:+.3f})")
        log(f"[판정] B3 표본>=30: {'통과' if b3 else '미달'} (n={n_sig})")
        log(f"\n>>> 실전 투입: {'가능 (소액 파일럿 권장)' if (b1 and b2 and b3) else '보류'}")
    else:
        log("시그널 0건 — 컷오프가 test에서 아무것도 통과 못 함. 실전 보류.")

    # ---------------------------------------------------- SHAP
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(X[va])
        sv = sv[1] if isinstance(sv, list) else sv
        shap.summary_plot(sv, X[va], show=False, max_display=20)
        plt.tight_layout(); plt.savefig(OUT / "shap_summary.png", dpi=150); plt.close()
        imp = (pd.Series(np.abs(sv).mean(0), index=feats)
                 .sort_values(ascending=False).head(15))
        log("\nSHAP 중요도 상위 15 (valid) — 실전 체크리스트 후보:")
        log(imp.round(4).to_string())
        log(f"플롯: {OUT/'shap_summary.png'}")
    except ImportError:
        log("\n(shap 미설치 — pip install shap)")

    (OUT / "model_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log(f"\n저장: {OUT / 'model_report.txt'}")


if __name__ == "__main__":
    main()
