"""50_absolute_check.py — '돈을 버는가' 기준 최종 재판정 (절대수익).

지수 비교 없이, 실제로 매수→180일 보유→매도했을 때 계좌에 남는
수익률로 지금까지의 전략 후보 전부를 재평가한다.

전략:
  S1  전체 CB 매수           (발행공시일 진입)  — "CB를 사면 돈이 되는가" 베이스라인
  S2  할증>10% 발행만         (동일 진입·보유)   — 1라운드 생존 신호
  S3  바닥 리픽싱(첫 도달)     (조정공시일 진입)
  S4  바닥 × 콜옵션
  S5  바닥 × 콜 최대주주귀속   — 2라운드 최종 후보

각 전략 출력:
  - 절대수익 평균 / 중앙값 / 승률(>0) / 연환산 근사
  - 클린 버전: 자본이벤트(관측창 내 일간 ±35% 초과 = 감자·병합 착시) 제외
  - test 구간(모델 미접촉 기간) 별도
  - 연도별 분해: 수익이 특정 불장(2020~21)에만 몰렸는지 확인

주의:
  - 상폐 종목은 마지막 거래일 종가로 청산 처리 → 실제로는 그보다 나쁠 수
    있어 결과는 '상한선'으로 해석
  - 거래비용·슬리피지 미반영. 소형주 특성상 평균에서 3~5%p 깎아 해석
  - 캐시 시세만 사용, API 호출 없음

사용법: cb_study 폴더에서  python 50_absolute_check.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/out")
CACHE = Path("data/cache")
HOLD_D = 180                      # 보유일
UP_LIM, DN_LIM = 1.35, 0.65       # 일간 ±35% 밖 = 자본이벤트
pd.set_option("display.width", 170)
REPORT: list[str] = []


def log(m=""):
    print(m); REPORT.append(str(m))


def _abs_return(ticker, t0: pd.Timestamp, bgn: str, end: str) -> dict:
    """캐시 시세에서 t0 첫 종가 진입 → +180일 내 마지막 종가 청산."""
    try:
        tk = str(ticker).split(".")[0].zfill(6)
        cp = CACHE / f"pxraw_{tk}_{bgn}_{end}.parquet"
        if not cp.exists():
            return {"cache_miss": 1}
        px = pd.read_parquet(cp)
        if px.empty or "종가" not in px.columns:
            return {"cache_miss": 1}
        close = px["종가"].astype(float)
        close = close[close > 0]
        w = close[(close.index >= t0)
                  & (close.index <= t0 + pd.Timedelta(days=HOLD_D))]
        if len(w) < 5:
            return {"cache_miss": 1}
        ratio = w / w.shift(1)
        return {
            "cache_miss": 0,
            "abs_ret": float(w.iloc[-1]) / float(w.iloc[0]) - 1,
            "capital_event": bool((ratio.max() > UP_LIM)
                                  or (ratio.min() < DN_LIM)),
            "early_end": int(len(w) < HOLD_D * 0.55),   # 관측일 급감=상폐 의심
            "year": int(t0.year),
        }
    except Exception:
        return {"cache_miss": 1}


def attach_returns(df: pd.DataFrame, bgn_off: int, end_off: int,
                   name: str) -> pd.DataFrame:
    rows = []
    for i, (_, ev) in enumerate(df.iterrows()):
        t0 = pd.Timestamp(ev["event_dt"])
        bgn = (t0 - pd.Timedelta(days=bgn_off)).strftime("%Y%m%d")
        end = (t0 + pd.Timedelta(days=end_off)).strftime("%Y%m%d")
        rows.append(_abs_return(ev["stock_code"], t0, bgn, end))
        if (i + 1) % 500 == 0:
            print(f"  ... {name} {i+1}/{len(df)}")
    lab = pd.DataFrame(rows, index=df.index)
    out = pd.concat([df, lab], axis=1)
    miss = int(out["cache_miss"].fillna(1).sum())
    out = out[out["abs_ret"].notna()]
    log(f"[{name}] 수익 계산 {len(out)}건 / 캐시미스 {miss}건 / "
        f"자본이벤트 {int(out['capital_event'].sum())}건 / "
        f"조기종료(상폐의심) {int(out['early_end'].sum())}건")
    return out


def stats(g: pd.DataFrame, name: str) -> None:
    x = g["abs_ret"].dropna()
    if len(x) < 10:
        log(f"  [{name}] 표본 부족 (n={len(x)})")
        return
    ann = (1 + x.mean()) ** (365 / HOLD_D) - 1
    log(f"  [{name:26s}] n={len(x):5d}  평균 {x.mean():+.3f}  "
        f"중앙 {x.median():+.3f}  승률 {(x > 0).mean():.2f}  "
        f"연환산≈{ann:+.1%}")


def evaluate(df: pd.DataFrame, title: str) -> None:
    log("\n" + "=" * 74)
    log(title)
    log("=" * 74)
    stats(df, "전체(착시 포함, 참고용)")
    clean = df[~df["capital_event"].fillna(False)]
    stats(clean, "클린 = 자본이벤트 제외")
    stats(clean[clean["split"] == "test"], "클린 | test")
    if len(clean) >= 40:
        yr = (clean.groupby("year")["abs_ret"]
                   .agg(["count", "mean", "median"]).round(3))
        yr = yr[yr["count"] >= 10]
        log("  연도별(클린):")
        for y, r in yr.iterrows():
            log(f"    {int(y)}: n={int(r['count']):4d}  "
                f"평균 {r['mean']:+.3f}  중앙 {r['median']:+.3f}  "
                f"{'▲' if r['mean'] > 0 else '▽'}")


def main() -> None:
    log("현금 보유 벤치마크 = 0.000 (절대수익 기준의 자연 비교선)")

    # ================= 1라운드: 발행공시일 진입 =================
    r1 = pd.read_parquet(OUT / "07_dataset.parquet")
    r1 = r1[r1["stock_code"].notna() & r1["event_dt"].notna()].copy()
    r1 = attach_returns(r1, bgn_off=40, end_off=750, name="1라운드")

    evaluate(r1, "S1. 전체 CB 매수 — 발행공시일 진입, 180일 보유")
    if "cv_premium" in r1.columns:
        evaluate(r1[r1["cv_premium"] > 0.10], "S2. 할증>10% 발행만")

    # ================= 2라운드: 바닥 리픽싱 =================
    p2 = OUT / "r2_04_dataset.parquet"
    if not p2.exists():
        log("\n(2라운드 데이터 없음 — S3~S5 생략)")
        _finish(); return
    r2 = pd.read_parquet(p2)
    r2 = r2[(r2["is_up_refix"] == 0) & (r2["at_floor"] == 1)
            & r2["stock_code"].notna()].copy()
    key = [c for c in ("corp_code", "bd_tm") if c in r2.columns]
    r2 = r2.sort_values("event_dt").groupby(key, dropna=False).head(1)
    r2 = attach_returns(r2, bgn_off=10, end_off=210, name="2라운드 첫바닥")

    evaluate(r2, "S3. 바닥 리픽싱(첫 도달) — 조정공시일 진입, 180일 보유")
    evaluate(r2[r2["has_call"] == True],                       # noqa: E712
             "S4. 바닥 × 콜옵션")
    evaluate(r2[r2["call_owner_side"] == True],                # noqa: E712
             "S5. 바닥 × 콜 최대주주귀속")
    _finish()


def _finish():
    log("\n" + "=" * 74)
    log("판정 기준 ('돈을 버는가'):")
    log("  1) 클린 test 평균 > 0  (착시 빼고, 미접촉 기간에서, 현금보다 나은가)")
    log("  2) 연도별로 특정 불장(2020~21)에만 의존하지 않는가")
    log("  3) 거래비용 버퍼 3~5%p를 깎아도 남는가")
    log("=" * 74)
    (OUT / "absolute_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log(f"저장: {OUT / 'absolute_report.txt'}")


if __name__ == "__main__":
    main()
