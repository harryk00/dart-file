"""5단계: 클러스터링 + 시간 분할.

- 같은 회사의 발행 결의가 CLUSTER_GAP_DAYS(180일) 이내로 이어지면
  하나의 클러스터로 묶는다 (이벤트 윈도우 중첩 → 표본 독립성 훼손 방지)
- 분할은 클러스터의 '최초 결의일' 기준으로 하며, 클러스터 전체가
  하나의 분할에만 속한다 (동일 회사 주가 흐름이 train/test에 동시
  들어가는 누수 차단)

TRAIN : ~2021-12-31   (리픽싱 규제 전 구간 포함)
VALID : ~2023-06-30   (규제 후. 튜닝/컷오프 결정)
TEST  : ~2024-06-30   (최종 홀드아웃. 단 1회만 평가)
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def assign(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["corp_code", "event_dt"]).copy()

    # --- 클러스터 id
    gap = df.groupby("corp_code")["event_dt"].diff().dt.days
    new_cluster = (gap.isna()) | (gap > C.CLUSTER_GAP_DAYS)
    df["cluster_id"] = (
        df["corp_code"].astype(str) + "_" +
        new_cluster.groupby(df["corp_code"]).cumsum().astype(int).astype(str)
    )

    # --- 클러스터 최초 결의일 기준 분할
    first_dt = df.groupby("cluster_id")["event_dt"].transform("min")
    train_end = pd.Timestamp(C.SPLIT_TRAIN_END)
    valid_end = pd.Timestamp(C.SPLIT_VALID_END)
    df["split"] = "test"
    df.loc[first_dt <= valid_end, "split"] = "valid"
    df.loc[first_dt <= train_end, "split"] = "train"

    df.to_parquet(C.OUT_DIR / "07_dataset.parquet", index=False)
    summary = (df.groupby("split")
                 .agg(n=("rcept_no", "count"),
                      firms=("corp_code", "nunique"),
                      L1_rate=("L1", "mean"),
                      delist_rate=("delisted_flag", "mean"))
                 .round(3))
    print(summary)
    return df
