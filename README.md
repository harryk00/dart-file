# CB 발행공시 후향 연구 — 수집 파이프라인

코스피+코스닥 전환사채권발행결정(주요사항보고서) 전수를 수집해, **발행 결의
시점에 알 수 있던 조건**과 **이후 실제 주가 상승 여부**를 연동하는 데이터셋을
만든다. 목적은 예측 이전에 "어떤 조건 조합이 상승으로 이어졌는가"의 근거 검증.

## 실행

```bash
pip install pandas pyarrow requests pykrx
export DART_API_KEY=발급키          # https://opendart.fss.or.kr
python run_pipeline.py              # 전체 실행 (캐시로 중단 후 재개 가능)
python run_pipeline.py --no-docs    # 원문 콜/풋 파싱 생략(빠른 1차 수집)
python run_pipeline.py --stage labels   # 특정 단계부터 재개
```

## 단계별 산출물 (`data/out/`)

| 파일 | 내용 |
|---|---|
| `01_filings.parquet` | list API 스캔 결과 (접수번호 목록) |
| `03_events.parquet` | 정정 반영·회차 단위로 정리된 발행 이벤트 |
| `05_features_fin.parquet` | 조건 피처 + 원문 콜/풋 + 결의일 이전 공시 재무 |
| `06_labeled.parquet` | L1/L2/L3 라벨 + 상폐 플래그 + 결의일 시세 피처 |
| `07_dataset.parquet` | 클러스터링 + train/valid/test 분할 완료본 |

## 시간 분할 (클러스터 단위)

| 구간 | 결의일 기준 | 근거 |
|---|---|---|
| TRAIN | 2016-01-01 ~ 2021-12-31 | 상향 리픽싱 의무화(2021-12) 이전 레짐 전체 포함 |
| VALID | 2022-01-01 ~ 2023-06-30 | 규제 후 레짐에서 튜닝·컷오프 결정 |
| TEST | 2023-07-01 ~ 2024-06-30 | 최종 홀드아웃, 1회만 평가 |

수집 종료를 2024-06-30로 끊은 이유: L1 관측창(발행 후 최대 720일)이
현재 시점 이전에 닫혀야 라벨이 미확정 상태로 남지 않음.
`post_refix_rule` 레짐 더미가 피처에 포함되며, 규제 전/후 서브샘플에서
동일 조건 조합의 효과가 유지되는지 비교하는 것 자체가 분석 포인트.

## 라벨 정의

- **L1 (주 라벨)**: 전환청구 가능 구간에서 종가 ≥ 최초 전환가 × 1.30이
  5영업일 이상 → CB 투자자 차익 실현이 실제로 가능했는가
- **L2**: 결의일 후 180일 시장지수 대비 초과수익 여부 (`L2_excess`)
- **L3**: 360일 내 MFE/MAE + 리픽싱 최저가 터치 여부 (`touched_floor`)
- **delisted_flag**: 관측창 대비 시세 조기 소멸(상폐/장기정지 추정).
  **분모에서 제거하지 말 것** — 생존편향 통제의 핵심. L1=0이면서
  delisted_flag=1인 표본이 "CB 다회차 → 상폐" 경로의 증거가 된다.

## 주요 피처

- 조건: `is_zero_zero`(표면0/만기0), `cv_premium`(전환가/결의일주가−1),
  `refix_floor_ratio`, `refix_below_70`, `days_to_cv_open`, `tenor_days`,
  자금목적 비중(`use_*_w`, `use_debt_heavy`), `has_call`/`call_owner_side`/
  `call_limit_pct`, `has_put`/`put_start_num` (원문 정규식 파싱)
- 희석: `dilution_pct_disclosed`(공시 필드), `dilution_now`,
  `dilution_max`(최저조정가 기준 최대 잠재 희석률), `cum_dilution_24m`
  (직전 24개월 동일회사 발행 누적)
- 재무(결의일 이전 접수분만): `cb_to_cash`, `impairment`(자본잠식률),
  `cfo_negative`, `fin_lag_days`
- 시세(결의 전): `ret_pre20`(결의 전 20영업일 수익률)

## 알려진 한계 / TODO

1. `cvbdIsDecsn` 필드명은 OpenDART DS005 개발가이드 기준으로 작성 —
   실행 전 가이드 페이지와 1회 대조 (없는 필드는 NaN 처리되어 죽지는 않음)
2. 콜/풋 원문 정규식은 1차 버전. 라벨링 후 has_call 표본 일부를 수검수해
   재현율 확인 후 패턴 보강 권장
3. 상폐 종목의 상폐 '이전' 시세는 pykrx로 대부분 수집되나, 일부 결측 시
   KRX 정보데이터시스템 수동 보완 필요 (`no_price_data=1` 표본 확인)
4. 발행철회(발행결정 후 철회 공시) 건 미제거 — 철회 공시 스캔 추가 예정
5. `shares_out_est`는 공시 희석률 필드에서 역산한 근사치. 정밀화하려면
   `stockTotqySttus` 연동으로 교체

## 다음 단계 (파이프라인 밖)

1. 단변량 층화표: (is_zero_zero × call_owner_side × dilution 구간)별
   L1 달성률 vs 베이스레이트 — 근거 제시의 본체
2. LightGBM 이진분류(L1) + SHAP, valid로 조기종료·컷오프, test 1회 평가
3. L2/L3로 강건성 확인, `touched_floor` 조건부 경로 분석
