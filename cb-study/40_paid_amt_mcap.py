"""40_paid_amt_mcap.py — 질문1: '실제 납입 완료' 앵커 검증.

설계
----
- 트리거: 최종 납입일(정정 반영). 02_terms_raw(정정 포함 전체 행)에서
  (회사,회차)별 최신 접수의 납입일을 최종값으로 채택.
  ※ 납입일 앵커에서는 정정 정보 사용이 look-ahead가 아님(이미 공시된 뒤).
- 피처: 권면총액/납입직전 시가총액, 납입지연(최초 공시 납입일 대비 일수),
  납입일 변경 횟수.
- 라벨: P60/P180 = 납입일 다음 거래일 진입, 60/180일 지수 대비 초과수익.
- 판정: 기존 원칙 유지 (초과수익 방향 + train/test 안정 + n>=200)

사용법: KRX 로그인 env 설정 후  python 40_paid_amt_mcap.py
        (시총 조회 일부 신규 — 30분 안팎)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from cb_study import config as C
from cb_study.labels import _prices, _index, _mcap_at

OUT = Path("data/out")
pd.set_option("display.width", 160)
REPORT: list[str] = []


def log(m=""):
    print(m); REPORT.append(str(m))


def _date(x):
    if x is None:
        return pd.NaT
    s = re.sub(r"[^\d]", "", str(x))
    return pd.to_datetime(s[:8], format="%Y%m%d", errors="coerce") if len(s) >= 8 else pd.NaT


def build() -> pd.DataFrame:
    raw = pd.read_parquet(OUT / "02_terms_raw.parquet")
    fil = pd.read_parquet(OUT / "01_filings.parquet")[["rcept_no", "rcept_dt"]]
    raw = raw.merge(fil, on="rcept_no", how="left")
    raw["rcept_dt"] = pd.to_datetime(raw["rcept_dt"], format="%Y%m%d")
    raw["pymd_dt"] = raw["pymd"].map(_date) if "pymd" in raw.columns else pd.NaT
    raw["bd_tm_n"] = pd.to_numeric(raw.get("bd_tm"), errors="coerce")

    g = raw.dropna(subset=["pymd_dt"]).sort_values("rcept_dt")
    agg = g.groupby(["corp_code", "bd_tm_n"]).agg(
        pay_first=("pymd_dt", "first"),
        pay_final=("pymd_dt", "last"),
        n_filings=("rcept_no", "size"),
    ).reset_index()
    agg["pay_delay_days"] = (agg["pay_final"] - agg["pay_first"]).dt.days
    agg["n_pay_changes"] = (g.groupby(["corp_code", "bd_tm_n"])["pymd_dt"]
                              .nunique().values - 1)

    ds = pd.read_parquet(OUT / "07_dataset.parquet")
    ds["bd_tm_n"] = pd.to_numeric(ds["bd_tm"], errors="coerce")
    keep = ["corp_code", "bd_tm_n", "stock_code", "corp_cls", "amt_total",
            "event_dt", "split", "delisted_flag"]
    df = ds[[c for c in keep if c in ds.columns]].merge(
        agg, on=["corp_code", "bd_tm_n"], how="inner")
    df = df[df["pay_final"].notna() & df["stock_code"].notna()].copy()
    log(f"[q1] 납입일 확보 이벤트 {len(df)}건 "
        f"(납입 변경 있음 {int((df['n_pay_changes']>0).sum())}건)")

    rows = []
    for _, ev in df.iterrows():
        rows.append(_label(ev))
    lab = pd.DataFrame(rows, index=df.index)
    df = pd.concat([df, lab], axis=1)
    df["amt_to_mcap_paid"] = df["amt_total"] / df["mcap_pre_pay"]
    df.to_parquet(OUT / "q1_paid_dataset.parquet", index=False)
    return df


def _label(ev) -> dict:
    out: dict = {}
    t0 = ev["pay_final"]
    ticker = str(ev["stock_code"]).zfill(6)
    # 1라운드 라벨과 동일 캐시 윈도우 재사용 (event_dt 기준 -40d ~ +750d)
    e0 = ev["event_dt"]
    bgn = (e0 - pd.Timedelta(days=40)).strftime("%Y%m%d")
    end = (e0 + pd.Timedelta(days=750)).strftime("%Y%m%d")
    try:
        px = _prices(ticker, bgn, end)
        close = px["종가"].astype(float)
        close = close[close > 0]
    except Exception:
        out["no_price_data"] = 1
        return out
    out["no_price_data"] = 0
    out["mcap_pre_pay"] = _mcap_at(ticker, t0)
    post = close[close.index > t0]
    for h in (60, 180):
        w = post[post.index <= t0 + pd.Timedelta(days=h)]
        if len(w) > 1:
            stk = float(w.iloc[-1]) / float(w.iloc[0]) - 1
            try:
                idx = _index(C.BENCH.get(ev.get("corp_cls"), "2001"), bgn, end)
                iw = idx[(idx.index >= w.index[0]) & (idx.index <= w.index[-1])]
                ir = float(iw.iloc[-1]) / float(iw.iloc[0]) - 1 if len(iw) > 1 else 0.0
            except Exception:
                ir = 0.0
            out[f"P{h}_excess"] = stk - ir
            out[f"P{h}"] = int(stk - ir > 0)
    return out


def strata(df: pd.DataFrame) -> None:
    df = df[df["P180_excess"].notna()].copy()
    df["ratio_bin"] = pd.cut(df["amt_to_mcap_paid"],
                             [0, 0.03, 0.07, 0.15, 0.30, np.inf],
                             labels=["<3%", "3~7%", "7~15%", "15~30%", ">30%"])
    df["delay_bin"] = pd.cut(df["pay_delay_days"].fillna(0),
                             [-1, 0, 30, np.inf],
                             labels=["지연없음", "1~30일", "30일+"])

    def table(d, by, label):
        b = d["P180"].mean()
        g = (d.groupby(by, observed=True, dropna=False)
               .agg(n=("P180", "size"), P60=("P60", "mean"),
                    P180=("P180", "mean"),
                    P180xs_med=("P180_excess", "median"),
                    P180xs_mean=("P180_excess", "mean")))
        g["lift"] = g["P180"] / b
        g = g[g["n"] >= 30].round(3)
        log(f"\n### {label}  (base P180={b:.3f})")
        log(g.to_string())

    for split in (None, "test"):
        d = df if split is None else df[df["split"] == split]
        log("\n" + "=" * 66)
        log(f"[{'전구간' if split is None else split}] n={len(d)}")
        log("=" * 66)
        table(d, "ratio_bin", "권면총액 / 납입직전 시가총액")
        table(d, "delay_bin", "납입 지연")
        table(d, "n_pay_changes", "납입일 변경 횟수")

    (OUT / "q1_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log(f"\n저장: {OUT / 'q1_report.txt'}")


if __name__ == "__main__":
    strata(build())
