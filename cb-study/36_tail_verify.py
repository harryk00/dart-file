"""36_tail_verify.py — 꼬리 수익의 진위 검증 (감자·병합 착시 제거).

원리: 국내 가격제한폭은 ±30%. 관측창 내 일간 종가 변동이 +35% 초과 또는
-35% 미만이면 시장 수익이 아니라 자본 이벤트(감자/병합/분할)의 명목가 점프다.
이런 이벤트가 낀 표본을 플래그하고, 제외한 통계를 재계산한다.

캐시된 원주가 parquet만 읽으므로 API 호출 없음.
사용법: python 36_tail_verify.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/out")
CACHE = Path("data/cache")
UP_LIM, DN_LIM = 1.35, 0.65          # 일간 ±35% 밖 = 자본 이벤트
pd.set_option("display.width", 160)


def max_daily_move(ev: pd.Series) -> dict:
    """관측창(트리거~180일) 내 최대 일간 변동비. 캐시 miss면 NaN."""
    t0 = pd.Timestamp(ev["event_dt"])
    ticker = str(ev["stock_code"]).zfill(6)
    bgn = (t0 - pd.Timedelta(days=10)).strftime("%Y%m%d")
    end = (t0 + pd.Timedelta(days=210)).strftime("%Y%m%d")
    cp = CACHE / f"pxraw_{ticker}_{bgn}_{end}.parquet"
    if not cp.exists():
        return {"max_ratio": np.nan, "min_ratio": np.nan}
    px = pd.read_parquet(cp)
    if px.empty or "종가" not in px.columns:
        return {"max_ratio": np.nan, "min_ratio": np.nan}
    close = px["종가"].astype(float)
    close = close[(close > 0) & (close.index >= t0)
                  & (close.index <= t0 + pd.Timedelta(days=180))]
    if len(close) < 3:
        return {"max_ratio": np.nan, "min_ratio": np.nan}
    ratio = close / close.shift(1)
    return {"max_ratio": float(ratio.max()), "min_ratio": float(ratio.min())}


def stats(g: pd.DataFrame, name: str) -> None:
    xs = g["R180_excess"].dropna()
    if xs.empty:
        print(f"[{name}] 표본 없음")
        return
    print(f"[{name}] n={len(xs)}  평균 {xs.mean():+.3f}  "
          f"중앙 {xs.median():+.3f}  승률 {(xs > 0).mean():.2f}  "
          f"상위5 {[f'{v:+.0%}' for v in xs.nlargest(5).tolist()]}")


def main() -> None:
    df = pd.read_parquet(OUT / "r2_04_dataset.parquet")
    df = df[(df["is_up_refix"] == 0) & df["R180_excess"].notna()
            & (df["at_floor"] == 1)].copy()
    key = [c for c in ("corp_code", "bd_tm") if c in df.columns]
    df = df.sort_values("event_dt").groupby(key, dropna=False).head(1)   # dedup
    print(f"검증 대상: 첫바닥 이벤트 {len(df)}건")

    moves = pd.DataFrame([max_daily_move(ev) for _, ev in df.iterrows()],
                         index=df.index)
    df = pd.concat([df, moves], axis=1)
    n_na = int(df["max_ratio"].isna().sum())
    df["capital_event"] = ((df["max_ratio"] > UP_LIM)
                           | (df["min_ratio"] < DN_LIM)).fillna(False)
    n_flag = int(df["capital_event"].sum())
    print(f"캐시 미확인 {n_na}건 / 자본이벤트 의심(일간 ±35% 초과) {n_flag}건 "
          f"({n_flag / max(len(df), 1):.0%})")

    print("\n" + "=" * 72)
    print("A. 자본이벤트 의심 상위 사례 (수동 확인용)")
    print("=" * 72)
    cols = [c for c in ("corp_name", "stock_code", "event_dt",
                        "R180_excess", "max_ratio") if c in df.columns]
    sus = df[df["capital_event"]].sort_values("R180_excess", ascending=False)
    if len(sus):
        print(sus[cols].head(12).to_string(index=False))
    else:
        print("(없음)")

    print("\n" + "=" * 72)
    print("B. 착시 제거 후 재판정 (dedup + capital_event 제외)")
    print("=" * 72)
    clean = df[~df["capital_event"]]
    for sub, nm in [
        (clean, "첫바닥 전체(클린)"),
        (clean[clean["has_call"] == True], "첫바닥×콜(클린)"),          # noqa: E712
        (clean[clean["call_owner_side"] == True], "첫바닥×콜귀속(클린)"),  # noqa: E712
    ]:
        stats(sub, nm + " | 전구간")
        stats(sub[sub["split"] == "test"], nm + " | test")
        print("-" * 72)

    print("판정: test 클린 평균이 여전히 양수 → 시그널 확정, 스코어러 제작 진행")
    print("      음수로 반전 → 꼬리가 감자 착시였음, 시그널 기각")


if __name__ == "__main__":
    main()
