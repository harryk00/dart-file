"""05_diagnose.py — '할인 90%' 진단: 수동 검증 표본 + L2 교차검증.

사용법: python 05_diagnose.py   (cb_study 폴더에서)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/out")
pd.set_option("display.width", 160)

df = pd.read_parquet(OUT / "07_dataset.parquet")
df = df[df["L1"].notna()].copy()

# ------------------------------------------------- 1. 수동 검증 표본 15건
# 할인 폭이 큰 순서. 이 회사들을 DART/차트에서 직접 확인:
#   발행일 '이후' 감자·액면병합 공시가 있으면 → 수정주가 왜곡(버그 B) 확정
disc = (df[df["cv_premium"] < -0.05]
        .sort_values("cv_premium")
        .head(15))
cols = [c for c in ["corp_name", "stock_code", "event_dt",
                    "cv_prc", "px_at_event", "cv_premium", "L1"] if c in disc.columns]
print("=" * 70)
print("1. 할인 상위 15건 — DART에서 발행일 이후 감자/액면병합 여부 확인용")
print("=" * 70)
print(disc[cols].to_string(index=False))

# ------------------------------------------------- 2. L2 교차검증
# L2(지수 대비 초과수익)는 전환가와 무관하게 정의 → 기계적 결합 판별기
df["cv_prem_bin"] = pd.cut(df["cv_premium"],
                           [-np.inf, -0.05, 0.02, 0.10, np.inf],
                           labels=["할인(<-5%)", "시가근접", "5~10%할증", ">10%할증"])
print("\n" + "=" * 70)
print("2. 전환가 구간별 L1 vs L2 비교 — L2에서도 우위가 유지되는가?")
print("=" * 70)
t = (df.groupby("cv_prem_bin", observed=True)
       .agg(n=("L1", "size"),
            L1_rate=("L1", "mean"),
            L2_rate=("L2", "mean"),
            L2_excess_med=("L2_excess", "median"))
       .round(3))
print(t.to_string())
print("""
해석:
  - L1은 높은데 L2가 평범/열위 → 할인 구간 우위는 기계적 결합(가짜)
  - L2에서도 뚜렷한 우위 → 경제적 실체 있음 (버그 수정 후 재확인 필요)
""")

# ------------------------------------------------- 3. zero-zero도 L2로 재확인
print("=" * 70)
print("3. zero-zero 역설의 L2 재확인")
print("=" * 70)
t2 = (df.groupby("is_zero_zero", observed=True)
        .agg(n=("L1", "size"), L1_rate=("L1", "mean"),
             L2_rate=("L2", "mean"), L2_excess_med=("L2_excess", "median"))
        .round(3))
print(t2.to_string())
