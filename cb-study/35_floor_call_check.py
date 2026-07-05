"""35_floor_call_check.py — 채택 시그널(바닥 리픽싱 × 콜)의 실탄 투입 전 정밀 검증.

확인 1: 평균 초과수익 — 중앙값이 음수여도 평균이 양수면(우측 꼬리)
        포트폴리오 전략으로 성립. 상위 수익 집중도도 함께.
확인 2: 회사 중복 — 같은 CB가 반복 바닥 리픽싱하면 이벤트가 중복 계산됨.
        (회사, 회차)당 '첫 바닥 도달'만 남긴 dedup 버전을 병기.

사용법: python 35_floor_call_check.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/out")
pd.set_option("display.width", 160)


def stats(g: pd.DataFrame, name: str) -> None:
    if g.empty:
        print(f"[{name}] 표본 없음")
        return
    xs60, xs180 = g["R60_excess"].dropna(), g["R180_excess"].dropna()
    firms = g["corp_code"].nunique() if "corp_code" in g.columns else np.nan
    top = xs180.nlargest(max(1, len(xs180) // 10)).sum()
    tot_pos = xs180[xs180 > 0].sum()
    print(f"[{name}] 이벤트 {len(g)}건 / 고유회사 {firms}곳")
    print(f"    R60 : 평균 {xs60.mean():+.3f}  중앙 {xs60.median():+.3f}  "
          f"승률 {(xs60 > 0).mean():.2f}")
    print(f"    R180: 평균 {xs180.mean():+.3f}  중앙 {xs180.median():+.3f}  "
          f"승률 {(xs180 > 0).mean():.2f}")
    if tot_pos > 0:
        print(f"    상위 10% 이벤트가 양(+)수익 총합의 {top / tot_pos:.0%} 차지")
    print(f"    최상위 5건 초과수익: "
          f"{[f'{v:+.0%}' for v in xs180.nlargest(5).tolist()]}")
    print(f"    최하위 5건 초과수익: "
          f"{[f'{v:+.0%}' for v in xs180.nsmallest(5).tolist()]}")


def main() -> None:
    df = pd.read_parquet(OUT / "r2_04_dataset.parquet")
    df = df[(df["is_up_refix"] == 0) & df["R180_excess"].notna()].copy()

    cell = df[(df["at_floor"] == 1)]
    cell_call = cell[cell["has_call"] == True]                     # noqa: E712
    cell_owner = cell[cell["call_owner_side"] == True]             # noqa: E712

    print("=" * 72)
    print("A. 이벤트 단위 (반복 리픽싱 포함) — 구간별")
    print("=" * 72)
    for sub, nm in [(cell, "바닥 전체"), (cell_call, "바닥×콜"),
                    (cell_owner, "바닥×콜귀속")]:
        print("-" * 72)
        for s in ("train", "valid", "test"):
            stats(sub[sub["split"] == s], f"{nm} | {s}")

    print("\n" + "=" * 72)
    print("B. (회사,회차)당 '첫 바닥 도달'만 — 중복 제거 버전")
    print("=" * 72)
    key = [c for c in ("corp_code", "bd_tm") if c in df.columns]
    first = (cell.sort_values("event_dt").groupby(key, dropna=False)
                 .head(1))
    first_call = first[first["has_call"] == True]                  # noqa: E712
    first_owner = first[first["call_owner_side"] == True]          # noqa: E712
    for sub, nm in [(first, "첫바닥 전체"), (first_call, "첫바닥×콜"),
                    (first_owner, "첫바닥×콜귀속")]:
        print("-" * 72)
        stats(sub, f"{nm} | 전구간")
        stats(sub[sub["split"] == "test"], f"{nm} | test")

    print("\n" + "=" * 72)
    print("C. 회사별 이벤트 집중도 (바닥×콜귀속 셀)")
    print("=" * 72)
    if "corp_code" in cell_owner.columns and len(cell_owner):
        cnt = cell_owner.groupby("corp_code").size().sort_values(ascending=False)
        print(f"이벤트 {len(cell_owner)}건이 {len(cnt)}개 회사에 분포")
        print(f"상위 10개 회사가 {cnt.head(10).sum()}건 "
              f"({cnt.head(10).sum() / len(cell_owner):.0%}) 차지")
        print("이벤트 수 분포:", cnt.describe()[["mean", "50%", "max"]].round(1).to_dict())

    print("\n판정 가이드:")
    print("  - dedup 후에도 test 평균 R180_excess > 0 → 실전 파일럿 가능")
    print("  - 평균은 +인데 상위 10%가 수익의 80%+ → 분산 필수(균등 소액 다수 진입)")
    print("  - dedup 후 평균 음수 → 시그널은 '회피 완화'용 정보로만 사용")


if __name__ == "__main__":
    main()
