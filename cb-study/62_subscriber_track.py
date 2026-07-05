"""62_subscriber_track.py — 검증①: 인수자 트랙레코드 (스마트머니 추적).

가설: 조건이 아니라 '누가 샀는가'. 발행 공시 원문의 대상자 명단에서
      기관형 인수자(조합·운용사·캐피탈 등)를 추출하고, 각 인수자의
      '과거 참여 딜들의 이후 성과'를 이력 점수로 만들어, 점수 높은
      인수자가 들어온 새 딜이 실제로 더 오르는지 검증.

look-ahead 차단: 인수자의 과거 딜 성과는 그 딜의 180일 관측창이
      '현재 딜 공시일 이전에 완전히 닫힌 것'만 집계 (240일 버퍼).

전제: features 단계를 --no-docs 없이 실행해 원문 캐시(doc_*.txt) 존재.
사용법: python 62_subscriber_track.py   (캐시만 사용, API 없음)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/out")
CACHE = Path("data/cache")
UP_LIM, DN_LIM = 1.35, 0.65
MIN_PRIOR = 2                 # 점수 산정에 필요한 인수자 과거 딜 최소 수
pd.set_option("display.width", 170)
REPORT: list[str] = []

# 기관형 인수자 접미사 (개인 배제). v2: 접미사 앞에 고유명 접두어 2자+ 필수,
# bare '조합'은 제외(업무집행조합/대표조합 등 계약 상투어 오인 방지)
NAME_PAT = re.compile(
    r"([가-힣A-Za-z0-9&\-]{2,20}?"
    r"(?:투자조합|자산운용|인베스트먼트|인베스트|파트너스|캐피탈|"
    r"벤처스|증권|저축은행|홀딩스|컨소시엄))")
TAG_RE = re.compile(r"<[^>]+>")
# 일반명사·유형어 블랙리스트 (정규화 후 정확 일치 시 제외)
GENERIC = {
    "투자조합", "개인투자조합", "벤처투자조합", "신기술투자조합",
    "신기술사업투자조합", "사모투자조합", "일반사모투자조합",
    "경영참여형사모투자조합", "기관전용사모투자조합", "창업벤처전문사모투자조합",
    "업무집행조합", "대표조합", "공동투자조합",
}
# 주관·인수 계약 상투 문맥(대상자가 아닌 역할 언급) 근처 등장 제외용
ROLE_PAT = re.compile(r"(대표주관|주관회사|인수단|청약취급|모집주선)")


def log(m=""):
    print(m); REPORT.append(str(m))


def norm_name(s: str) -> str:
    s = re.sub(r"\s|\(주\)|주식회사", "", s)
    s = re.sub(r"제?\d+호$", "", s)          # OO투자조합 제3호 → OO투자조합
    return s


def extract_subscribers(rcept_no: str) -> list:
    cp = CACHE / f"doc_{rcept_no}.txt"
    if not cp.exists():
        return []
    text = cp.read_text(encoding="utf-8", errors="ignore")
    clean = re.sub(r"\s+", " ", TAG_RE.sub(" ", text))
    i = clean.find("대상자")
    seg = clean[i:i + 4000] if i != -1 else clean[:8000]
    out = set()
    for m in NAME_PAT.finditer(seg):
        nm = norm_name(m.group(1))
        if len(nm) < 4 or nm in GENERIC:
            continue
        # 주관·청약 역할 문맥(±40자)에 걸린 이름은 인수자가 아님
        ctx = seg[max(0, m.start() - 40):m.end() + 40]
        if ROLE_PAT.search(ctx):
            continue
        out.add(nm)
    return sorted(out)[:20]


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


def main():
    df = pd.read_parquet(OUT / "07_dataset.parquet")
    df = df[df["stock_code"].notna()].copy().reset_index(drop=True)

    # 1) 성과 라벨
    lab = pd.DataFrame([abs180(ev) for _, ev in df.iterrows()], index=df.index)
    df = pd.concat([df, lab], axis=1)

    # 2) 인수자 추출
    df["subs"] = [extract_subscribers(r) for r in df["rcept_no"]]
    n_any = int((df["subs"].str.len() > 0).sum())
    all_names = pd.Series([n for subs in df["subs"] for n in subs])
    log(f"인수자 추출: {n_any}/{len(df)}건에서 기관형 인수자 발견 / "
        f"고유 인수자 {all_names.nunique()}곳")
    log("최다 등장 인수자 상위 10:")
    log(all_names.value_counts().head(10).to_string())

    # 3) (인수자, 딜) 롱테이블 → 이력 점수
    long = df.explode("subs").dropna(subset=["subs"])
    long = long[long["subs"] != ""]
    long = long.sort_values("event_dt")

    scores, n_priors = [], []
    hist: dict = {}                       # name -> list of (close_dt, ret)
    df_sorted = df.sort_values("event_dt")
    for _, ev in df_sorted.iterrows():
        t0 = pd.Timestamp(ev["event_dt"])
        vals = []
        for nm in ev["subs"]:
            past = [r for (cd, r) in hist.get(nm, []) if cd <= t0]
            if len(past) >= MIN_PRIOR:
                vals.append(np.mean(past))
        scores.append(np.mean(vals) if vals else np.nan)
        n_priors.append(len(vals))
        # 현재 딜을 이력에 등록 (성과는 t0+240일 후부터 관측 가능)
        if not np.isnan(ev.get("abs_ret", np.nan)) and not ev.get("cap", False):
            close_dt = t0 + pd.Timedelta(days=240)
            for nm in ev["subs"]:
                hist.setdefault(nm, []).append((close_dt, ev["abs_ret"]))
    df_sorted["smart_score"] = scores
    df_sorted["n_scored_subs"] = n_priors
    df = df_sorted
    df.to_parquet(OUT / "subscriber_dataset.parquet", index=False)
    log(f"\n이력점수 산출 가능: {int(df['smart_score'].notna().sum())}건 "
        f"(인수자 과거딜 {MIN_PRIOR}건 이상 필요)")

    # 4) 층화
    df["score_bin"] = pd.cut(df["smart_score"],
                             [-np.inf, -0.15, 0.0, 0.15, np.inf],
                             labels=["부진(<-15%)", "약(-15~0)",
                                     "양(0~15%)", "우수(>15%)"])
    for split in (None, "test"):
        d = df if split is None else df[df["split"] == split]
        d = d[d["abs_ret"].notna() & ~d["cap"].fillna(False)]
        log("\n" + "=" * 70)
        log(f"[{'전구간' if split is None else 'test'}] 클린 n={len(d)}")
        log("=" * 70)
        g = (d.groupby("score_bin", observed=True, dropna=False)["abs_ret"]
               .agg(n="size", mean="mean", med="median",
                    win=lambda s: (s > 0).mean()))
        g = g[g["n"] >= 20].round(3)
        log("\n### 인수자 이력점수별 → 이번 딜 180일 절대수익")
        log(g.to_string())

    log("\n판정: '우수' 구간의 클린 test 평균>0 + 점수 단조성. "
        "이름 정규화 한계로 상위 인수자 몇 곳은 원문 수검수 권장.")
    (OUT / "subscriber_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log(f"저장: {OUT / 'subscriber_report.txt'}")


if __name__ == "__main__":
    main()
