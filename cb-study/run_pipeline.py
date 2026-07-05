"""CB 발행공시 후향 연구 — 수집 파이프라인 실행 스크립트.

사용법:
    export DART_API_KEY=발급받은키
    pip install pandas pyarrow requests pykrx
    python run_pipeline.py                # 전체 실행
    python run_pipeline.py --stage labels # 특정 단계부터 재개(캐시 재사용)

단계:
    collect  → data/out/03_events.parquet        (코스피+코스닥 CB 발행결정 전수)
    features → data/out/05_features_fin.parquet  (조건+원문 콜/풋+재무 피처)
    labels   → data/out/06_labeled.parquet       (L1/L2/L3 + 상폐 플래그)
    split    → data/out/07_dataset.parquet       (train/valid/test 분할 완료)
"""
from __future__ import annotations

import argparse

import pandas as pd

from cb_study import collect, features, financials, labels, split
from cb_study import config as C

STAGES = ["collect", "features", "labels", "split"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES, default="collect",
                    help="이 단계부터 실행 (이전 산출물은 parquet에서 로드)")
    ap.add_argument("--no-docs", action="store_true",
                    help="원문 콜/풋 파싱 생략 (빠른 1차 수집용)")
    args = ap.parse_args()
    start = STAGES.index(args.stage)

    if not C.DART_API_KEY:
        raise SystemExit("환경변수 DART_API_KEY 를 설정하세요.")

    ev = fe = lb = None
    if start <= 0:
        ev = collect.run()
    if start <= 1:
        ev = ev if ev is not None else pd.read_parquet(C.OUT_DIR / "03_events.parquet")
        fe = features.build(ev, parse_docs=not args.no_docs)
        fe = financials.attach(fe)
    if start <= 2:
        fe = fe if fe is not None else pd.read_parquet(
            C.OUT_DIR / "05_features_fin.parquet")
        lb = labels.build(fe)
    lb = lb if lb is not None else pd.read_parquet(C.OUT_DIR / "06_labeled.parquet")
    ds = split.assign(lb)

    print(f"\n완료: {len(ds)}건 → {C.OUT_DIR / '07_dataset.parquet'}")
    print("다음 단계 제안: 단변량 층화표(조건 조합별 L1 달성률) → LightGBM+SHAP")


if __name__ == "__main__":
    main()
