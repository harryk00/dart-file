"""61_combo_filings.py — 검증②: 동반 공시 콤보.

가설: CB 발행 '조건'이 아니라 전후 ±10일의 다른 공시와의 '조합'이 신호.
      특히 최대주주변경+CB(무자본 M&A 시그니처), 유상증자 동시, BW 동시,
      감자 직후 CB.

수집: 이벤트별 corp_code 한정 list API 스캔(±10일). 캐시되므로 재실행 무료.
결과: 발행공시일 진입 180일 절대수익(클린), 콤보별 층화 + test.

사용법: export DART_API_KEY=... 후  python 61_combo_filings.py
        (~4천 콜, 첫 실행 20~30분)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cb_study.dart_client import get_json

OUT = Path("data/out")
CACHE = Path("data/cache")
UP_LIM, DN_LIM = 1.35, 0.65
pd.set_option("display.width", 170)
REPORT: list[str] = []

KEYWORDS = {
    "co_owner_chg": ["최대주주변경", "최대주주 변경"],
    "co_rights":    ["유상증자결정"],
    "co_bonus":     ["무상증자결정"],
    "co_capred":    ["감자결정"],
    "co_bw":        ["신주인수권부사채권발행결정"],
    "co_acq":       ["타법인주식및출자증권양수결정", "영업양수결정"],
    "co_inquiry":   ["조회공시"],
}


def log(m=""):
    print(m); REPORT.append(str(m))


def co_filings(ev) -> dict:
    t0 = pd.Timestamp(ev["event_dt"])
    out = {k: 0 for k in KEYWORDS}
    out["n_co"] = 0
    try:
        data = get_json("list", corp_code=ev["corp_code"],
                        bgn_de=(t0 - pd.Timedelta(days=10)).strftime("%Y%m%d"),
                        end_de=(t0 + pd.Timedelta(days=10)).strftime("%Y%m%d"),
                        page_no=1, page_count=100)
    except Exception:
        return out
    for it in data.get("list", []) or []:
        nm = it.get("report_nm", "")
        if "전환사채권발행결정" in nm:
            continue
        out["n_co"] += 1
        for key, kws in KEYWORDS.items():
            if any(k in nm for k in kws):
                out[key] = 1
    return out


def abs180(ev) -> dict:
    try:
        tk = str(ev["stock_code"]).split(".")[0].zfill(6)
        t0 = pd.Timestamp(ev["event_dt"])
        bgn = (t0 - pd.Timedelta(days=40)).strftime("%Y%m%d")
        end = (t0 + pd.Timedelta(days=750)).strftime("%Y%m%d")
        cp = CACHE / f"pxraw_{tk}_{bgn}_{end}.parquet"
        if not cp.exists():
            return {}
        close = pd.read_parquet(cp)["종가"].astype(float)
        close = close[close > 0]
        w = close[(close.index >= t0)
                  & (close.index <= t0 + pd.Timedelta(days=180))]
        if len(w) < 5:
            return {}
        r = (w / w.shift(1)).dropna()
        return {"abs_ret": float(w.iloc[-1]) / float(w.iloc[0]) - 1,
                "cap": bool((r.max() > UP_LIM) or (r.min() < DN_LIM))}
    except Exception:
        return {}


def table(d, by, label):
    d = d[d["abs_ret"].notna() & ~d["cap"].fillna(False)]
    if d.empty:
        return
    g = (d.groupby(by, observed=True, dropna=False)["abs_ret"]
           .agg(n="size", mean="mean", med="median",
                win=lambda s: (s > 0).mean()))
    g = g[g["n"] >= 20].round(3)          # 콤보는 희귀해 완화(단 해석 보수적으로)
    if len(g):
        log(f"\n### {label} (180일 절대수익, 클린)")
        log(g.to_string())


def main():
    df = pd.read_parquet(OUT / "07_dataset.parquet")
    df = df[df["stock_code"].notna()].copy()
    log(f"대상 {len(df)}건 — 동반공시 스캔 시작 (캐시되면 재실행 무료)")
    co = []
    for i, (_, ev) in enumerate(df.iterrows()):
        co.append(co_filings(ev))
        if (i + 1) % 300 == 0:
            print(f"  ... {i+1}/{len(df)}")
    df = pd.concat([df, pd.DataFrame(co, index=df.index)], axis=1)
    lab = pd.DataFrame([abs180(ev) for _, ev in df.iterrows()], index=df.index)
    df = pd.concat([df, lab], axis=1)
    df.to_parquet(OUT / "combo_dataset.parquet", index=False)

    for k in KEYWORDS:
        log(f"{k}: {int(df[k].sum())}건 동반")

    for split in (None, "test"):
        d = df if split is None else df[df["split"] == split]
        log("\n" + "=" * 70)
        log(f"[{'전구간' if split is None else 'test'}] n={len(d)}")
        log("=" * 70)
        for k in KEYWORDS:
            table(d, k, f"동반공시: {k}")
        d["n_co_bin"] = pd.cut(d["n_co"], [-1, 0, 2, np.inf],
                               labels=["없음", "1~2건", "3건+"])
        table(d, "n_co_bin", "±10일 내 기타 공시 건수")

    log("\n판정: 클린 test 평균>0 + 전구간 방향 일치 + n>=20(보수 해석).")
    (OUT / "combo_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log(f"저장: {OUT / 'combo_report.txt'}")


if __name__ == "__main__":
    main()
