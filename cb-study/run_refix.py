"""2라운드 실행: 전환가액조정 트리거 데이터셋 구축.

사용법:
    export DART_API_KEY=... KRX_ID=... KRX_PW=...
    python run_refix.py

전제: 1라운드 산출물 data/out/07_dataset.parquet 존재 (CB 조건 연결용)
"""
from cb_study import refix
from cb_study import config as C

if __name__ == "__main__":
    if not C.DART_API_KEY:
        raise SystemExit("환경변수 DART_API_KEY 를 설정하세요.")
    refix.run()
    print("다음: python 30_refix_strata.py")
