"""00_pykrx_probe.py — pykrx가 감자 종목에 대해 실제로 뭘 반환하는지 진단.

수정주가 왜곡의 정확한 원인을 확인하기 위한 일회성 진단.
오가닉티코스메틱(900300)은 2023년 대규모 감자 이력이 있는 종목.

사용법: python 00_pykrx_probe.py
"""
from __future__ import annotations

import pandas as pd
from pykrx import stock

pd.set_option("display.width", 160)

TICKER = "900300"          # 오가닉티코스메틱 (감자 이력)
BGN, END = "20210801", "20210930"    # 2021-09-01 CB 결의(cv_prc=1155) 전후

print("=" * 70)
print(f"종목 {TICKER}, 기간 {BGN}~{END}")
print("이 시기 실제 주가는 1,000원 안팎이어야 정상 (cv_prc=1155)")
print("=" * 70)

# --- (A) get_market_ohlcv (현재 파이프라인이 쓰는 함수, 파라미터 없음)
print("\n[A] get_market_ohlcv(BGN, END, TICKER) — 현재 파이프라인 방식")
try:
    a = stock.get_market_ohlcv(BGN, END, TICKER)
    print(a[["종가"]].head(3).to_string())
    print("...", a["종가"].iloc[-1], "(마지막)")
except Exception as e:
    print("에러:", e)

# --- (B) adjusted=False (원주가 명시)
print("\n[B] get_market_ohlcv_by_date(..., adjusted=False) — 원주가")
try:
    b = stock.get_market_ohlcv_by_date(BGN, END, TICKER, adjusted=False)
    print(b[["종가"]].head(3).to_string())
    print("...", b["종가"].iloc[-1], "(마지막)")
except Exception as e:
    print("에러:", e)

# --- (C) adjusted=True (수정주가 명시)
print("\n[C] get_market_ohlcv_by_date(..., adjusted=True) — 수정주가")
try:
    c = stock.get_market_ohlcv_by_date(BGN, END, TICKER, adjusted=True)
    print(c[["종가"]].head(3).to_string())
    print("...", c["종가"].iloc[-1], "(마지막)")
except Exception as e:
    print("에러:", e)

print("\n" + "=" * 70)
print("판정:")
print("  - [B]가 1,000원 안팎이면 → adjusted=False가 정답. labels.py를 이걸로 교체")
print("  - [A]와 [C]가 수십만원이면 → 기본값(수정주가)이 왜곡의 원인 확정")
print("  - [B]도 수십만원이면 → pykrx가 원주가를 못 주는 것. 다른 방법 필요")
print("=" * 70)
