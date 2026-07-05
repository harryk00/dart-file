"""30_refix_strata.py — 2라운드 층화표: 리픽싱 깊이·바닥도달·재무압박 vs 반등.

지표 병기 원칙 (1라운드 교훈):
  R180_excess : 지수 대비 초과수익 — 정직한 판정 기준
  RCV         : 조정후가액×1.3 도달 — 메커니즘(전환 유도) 지표, 기계적 결합 주의

사용법: python 30_refix_strata.py [--split train]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/out")
pd.set_option("display.width", 160)
REPORT: list[str] = []


def log(m=""):
    print(m); REPORT.append(str(m))


def table(df, by, label):
    b60, b180 = df["R60"].mean(), df["R180"].mean()
    g = (df.groupby(by, dropna=False, observed=True)
           .agg(n=("R180", "size"),
                R60=("R60", "mean"),
                R180=("R180", "mean"),
                R180xs_med=("R180_excess", "median"),
                RCV=("RCV", "mean")))
    g["R180_lift"] = g["R180"] / b180
    g = g[g["n"] >= 30].round(3)
    log(f"\n### {label}  (base: R60={b60:.3f}, R180={b180:.3f})")
    log(g[["n", "R60", "R180", "R180_lift", "R180xs_med", "RCV"]].to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, choices=["train", "valid", "test"])
    args = ap.parse_args()

    df = pd.read_parquet(OUT / "r2_04_dataset.parquet")
    log("=" * 70)
    log("0. 표본·파싱 진단")
    log("=" * 70)
    n_all = len(df)
    n_pair = int(df.get("pair_ok", pd.Series(dtype=float)).fillna(0).sum())
    n_dn = int((df["is_up_refix"] == 0).sum())
    n_up = int((df["is_up_refix"] == 1).sum())
    log(f"전체 트리거 {n_all}건 / 전·후 페어 파싱 {n_pair}건 ({n_pair/n_all:.0%})")
    log(f"방향 확정: 하향 {n_dn}건 / 상향 {n_up}건 / 미확정 {n_all-n_dn-n_up}건")
    log(f"CB조건 매칭 {int(df['matched'].sum())}건")
    if "reason_siga" in df.columns:
        log(f"조정사유 '시가하락' 언급 {int(df['reason_siga'].fillna(False).sum())}건")
    if "adj_sane" in df.columns:
        sane = df["adj_sane"].dropna()
        if len(sane):
            log(f"교차검증(조정후가액이 최저가~최초가 범위 내): "
                f"{sane.mean():.0%} 정상 (검증가능 {len(sane)}건)")
            if sane.mean() < 0.85:
                log("⚠️  범위 밖 비율이 높음 — 파싱 재점검 필요. 아래 표 신뢰 유보.")

    # [v2] 분석 표본 = '하향 확정'(전·후 모두 파싱 & adj_pct<0) + 라벨 존재.
    #      v1은 파싱 실패를 하향으로 오분류해 표본의 98%가 껍데기였음.
    df = df[(df["is_up_refix"] == 0) & df["R180"].notna()].copy()
    if args.split:
        df = df[df["split"] == args.split]
        log(f"[{args.split}] 구간: {len(df)}건")
    log(f"분석 표본(하향 확정) {len(df)}건")

    # 구간화
    df["depth_bin"] = pd.cut(df["refix_depth"],
                             [0, 0.72, 0.80, 0.90, 1.01],
                             labels=["바닥권(≤72%)", "72~80%", "80~90%", "90%+"])
    df["pxcv_bin"] = pd.cut(df["px_over_newcv"],
                            [0, 0.95, 1.05, 1.2, np.inf],
                            labels=["주가<전환가", "전환가 근접", "1.05~1.2", ">1.2"])
    df["seq_bin"] = pd.cut(df["refix_seq"].astype(float),
                           [0, 1.5, 3.5, np.inf], labels=["1회차", "2~3회차", "4회+"])
    df["cash_bin"] = pd.cut(df["cb_to_cash_rf"],
                            [0, 0.5, 1, 2, np.inf],
                            labels=["여유", "0.5~1", "1~2", "압박(>2)"])

    log("\n" + "=" * 70)
    log("1. 질문(a): 어느 전환가액 수준에서 반등이 오는가")
    log("=" * 70)
    if "reason_siga" in df.columns:
        table(df, df["reason_siga"].fillna(False),
              "조정사유: 시가하락 언급 여부 (True=순수 리픽싱)")
    table(df, "depth_bin", "리픽싱 깊이 (조정후/최초 전환가)")
    table(df, "at_floor", "최저조정가 도달 (버퍼 소진)")
    table(df, "pxcv_bin", "주가/조정후전환가 (새 전환가 대비 주가 위치)")
    table(df, "seq_bin", "몇 번째 조정인가")

    log("\n" + "=" * 70)
    log("2. 질문(b): 회사가 '올려야만 하는' 재무 지점")
    log("=" * 70)
    table(df, "cash_bin", "CB총액/현금 (조정일 기준 재계산)")
    table(df, "put_within_6m", "풋옵션 개시 6개월 내 임박")
    table(df, "pressure", "압박 스코어 (현금부족 × 풋임박)")

    log("\n" + "=" * 70)
    log("3. 본명 가설: 바닥 리픽싱 × 콜옵션")
    log("=" * 70)
    if "has_call" in df.columns:
        sub = df[df["at_floor"] == 1]
        if len(sub) >= 60:
            table(sub, "has_call", "at_floor=1 표본에서 콜옵션 유무")
            table(sub, "call_owner_side", "at_floor=1 표본에서 콜 귀속")
        pt = df.pivot_table(index="at_floor", columns="has_call",
                            values="R180_excess", aggfunc=["median", "size"],
                            observed=True)
        log("\n### at_floor × has_call (셀: R180 초과수익 중앙값 / n)")
        log(pt.round(3).to_string())

    log("\n⚠️  판정 원칙: R180 방향 + 구간(train/test) 안정 + n>=200.")
    (OUT / "refix_strata_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log(f"\n저장: {OUT / 'refix_strata_report.txt'}")


if __name__ == "__main__":
    main()
