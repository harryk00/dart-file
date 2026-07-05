"""CB 발행공시 후향 연구 파이프라인 - 전역 설정."""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 기본 경로
ROOT = Path(os.environ.get("CB_STUDY_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # API 원본 응답 캐시 (재실행 시 재사용)
OUT_DIR = DATA_DIR / "out"              # 단계별 산출물 (parquet)
for d in (CACHE_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- DART API
DART_API_KEY = os.environ.get("DART_API_KEY", "")   # export DART_API_KEY=...
DART_BASE = "https://opendart.fss.or.kr/api"
DART_SLEEP = 0.15        # 요청 간 대기(초). 일 20,000건 한도 고려
DART_MAX_RETRY = 3

# ---------------------------------------------------------------- 수집 범위
# 발행결의(rcept_dt) 기준. 라벨 최장 관측창(발행 후 24개월)이 현재 시점(2026-07)
# 이전에 닫히도록 2024-06-30에서 끊는다.
COLLECT_BGN = "20160101"
COLLECT_END = "20240630"
MARKETS = {"Y", "K"}     # corp_cls: Y=유가증권(코스피), K=코스닥

# ---------------------------------------------------------------- 시간 분할
# 2021-12 전환가액 상향조정(리픽싱) 의무화로 조건 분포의 레짐이 바뀌므로
# 학습 구간이 규제 전/후를 모두 포함하도록 설계하고, 레짐 더미를 피처로 둔다.
#   TRAIN : 2016-01-01 ~ 2021-12-31  (규제 전 구간 전체)
#   VALID : 2022-01-01 ~ 2023-06-30  (규제 후, 하이퍼파라미터/컷오프 튜닝)
#   TEST  : 2023-07-01 ~ 2024-06-30  (최종 홀드아웃, 단 1회만 조회)
SPLIT_TRAIN_END = "20211231"
SPLIT_VALID_END = "20230630"
REFIX_REGIME_DATE = "20211201"   # 상향 리픽싱 의무화 시행일(레짐 더미 기준)

# 같은 회사의 발행 결의가 이 일수 이내로 겹치면 하나의 클러스터로 묶어
# 클러스터 단위로 분할(leakage 방지)
CLUSTER_GAP_DAYS = 180

# ---------------------------------------------------------------- 라벨 파라미터
L1_TARGET_MULT = 1.30    # 최초 전환가 대비 130% 달성 여부
L1_MIN_DAYS = 5          # 130% 이상을 유지해야 하는 최소 영업일수
L1_HORIZON_D = 720       # L1 관측창: 발행 후 최대 720일(전환청구기간과 교집합)
L2_HORIZON_D = 180       # L2: 지수 대비 상대수익률 관측창
L3_HORIZON_D = 360       # L3: MFE/MAE 관측창
BENCH = {"Y": "1001", "K": "2001"}   # pykrx 지수코드: KOSPI / KOSDAQ

# ---------------------------------------------------------------- 재무 계정 매핑
# fnlttSinglAcntAll(전체 재무제표)에서 뽑을 계정. account_nm 부분일치로 탐색.
FIN_ACCOUNTS = {
    "cash":        ["현금및현금성자산"],
    "st_fin":      ["단기금융상품", "단기금융자산"],
    "equity":      ["자본총계"],
    "capital":     ["자본금"],
    "liab_cur":    ["유동부채"],
    "cfo":         ["영업활동현금흐름", "영업활동으로인한현금흐름"],
    "revenue":     ["매출액", "영업수익"],
}
REPRT_CODES = {"11013": "1Q", "11012": "반기", "11014": "3Q", "11011": "사업"}
