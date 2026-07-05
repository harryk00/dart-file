"""60_short_drift.py — 검증③: 공시 직후 시장 반응 → 단기 드리프트.

가설: 장기(180d+) 보유는 전부 진다(확정). 남은 시간축은 단기.
      공시 첫 거래일의 '시장 반응'(수익률·거래량 폭증)이 이후 2~12주
      드리프트를 가르는가?

look-ahead 차단: 조건 = 공시 첫 거래일(D0) 종가·거래량 → 진입 = D+1 종가.
결과: D+1 진입 → +20영업일 / +60영업일 청산 절대수익.
판정: 클린 test 평균 > 0 + 반응 구간별 단조성.

사용법: python 60_short_drift.py   (캐시만 사용, API 없음)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/out")
CACHE = Path("data/cache")
UP_LIM, DN_LIM = 1.35, 0.65
pd.set_option("display.width", 170)
REPORT: list[str] = []


def log(m=""):
    print(m); REPORT.append(str(m))


def one(ev) -> dict:
    try:
        tk = str(ev["stock_code"]).split(".")[0].zfill(6)
        t0 = pd.Timestamp(ev["event_dt"])
        bgn = (t0 - pd.Timedelta(days=40)).strftime("%Y%m%d")
        end = (t0 + pd.Timedelta(days=750)).strftime("%Y%m%d")
        cp = CACHE / f"pxraw_{tk}_{bgn}_{end}.parquet"
        if not cp.exists():
            return {}
        px = pd.read_parquet(cp)
        if "종가" not in px.columns:
            return {}
        close = px["종가"].astype(float)
        vol = px["거래량"].astype(float) if "거래량" in px.columns else None
        close = close[close > 0]
        pre = close[close.index < t0]
        post = close[close.index >= t0]
        if len(pre) < 21 or len(post) < 65:
            return {}
        # --- D0 반응 (조건)
        react = float(post.iloc[0]) / float(pre.iloc[-1]) - 1
        volx = np.nan
        if vol is not None:
            v_pre = vol[vol.index < t0].tail(20)
            v0 = vol[vol.index >= t0]
            if len(v_pre) >= 10 and len(v0) and v_pre.mean() > 0:
                volx = float(v0.iloc[0]) / float(v_pre.mean())
        # --- D+1 진입 → +20 / +60 영업일 청산 (절대수익)
        entry = float(post.iloc[1])
        out = {"react": react, "volx": volx, "year": int(t0.year)}
        for h in (20, 60):
            if len(post) > 1 + h:
                w = post.iloc[1:2 + h]
                ratio = (w / w.shift(1)).dropna()
                out[f"ret{h}"] = float(w.iloc[-1]) / entry - 1
                out[f"cap{h}"] = bool((ratio.max() > UP_LIM)
                                      or (ratio.min() < DN_LIM))
        return out
    except Exception:
        return {}


def table(d, by, col, label):
    d = d[d[col].notna() & ~d[f"cap{col[3:]}"].fillna(False)]
    if d.empty:
        return
    g = (d.groupby(by, observed=True, dropna=False)[col]
           .agg(n="size", mean="mean", med="median",
                win=lambda s: (s > 0).mean()))
    g = g[g["n"] >= 30].round(3)
    if len(g):
        log(f"\n### {label} → {col} (클린)")
        log(g.to_string())


def main():
    df = pd.read_parquet(OUT / "07_dataset.parquet")
    df = df[df["stock_code"].notna()].copy()
    lab = pd.DataFrame([one(ev) for _, ev in df.iterrows()], index=df.index)
    df = pd.concat([df, lab], axis=1)
    df = df[df["react"].notna()]
    log(f"단기반응 계산 {len(df)}건 / 거래량 확보 {int(df['volx'].notna().sum())}건")

    df["react_bin"] = pd.cut(df["react"],
                             [-np.inf, -0.05, -0.01, 0.01, 0.05, 0.15, np.inf],
                             labels=["급락(<-5%)", "하락", "중립", "상승",
                                     "급등(5~15%)", "폭등(>15%)"])
    df["volx_bin"] = pd.cut(df["volx"], [0, 1, 3, 10, np.inf],
                            labels=["평소이하", "1~3x", "3~10x", "폭증(10x+)"])

    for split in (None, "test"):
        d = df if split is None else df[df["split"] == split]
        log("\n" + "=" * 70)
        log(f"[{'전구간' if split is None else 'test'}] n={len(d)}")
        log("=" * 70)
        table(d, "react_bin", "ret20", "D0 주가반응별 → 20영업일 드리프트")
        table(d, "react_bin", "ret60", "D0 주가반응별 → 60영업일 드리프트")
        table(d, "volx_bin", "ret20", "D0 거래량배율별 → 20영업일")
        # 반응 × 거래량 결합 (급등+폭증 = '시장이 확신한' 공시)
        d2 = d[d["ret20"].notna() & ~d["cap20"].fillna(False)]
        if len(d2) > 200:
            pt = d2.pivot_table(index="react_bin", columns="volx_bin",
                                values="ret20", aggfunc=["mean", "size"],
                                observed=True)
            log("\n### 반응 × 거래량 (셀: 20일 평균수익 / n)")
            log(pt.round(3).to_string())

    log("\n판정: 클린 test 평균>0 + react 구간 단조성. 비용버퍼 왕복 ~2%p 감안.")
    (OUT / "drift_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log(f"저장: {OUT / 'drift_report.txt'}")


if __name__ == "__main__":
    main()
