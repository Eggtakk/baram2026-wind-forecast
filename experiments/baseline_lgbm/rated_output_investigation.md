# ⚠️ 중요 갱신: 기존 holdout 검증 방식의 결함 발견 (2026-07-24)

오늘 saturation+power curve+재튜닝 버전(v2)을 실제 제출했더니 리더보드 점수가
**0.6060 → 0.6036으로 오히려 하락**했다. 로컬 holdout에서는 개선(0.606→
~0.624 평균)으로 나왔던 것과 정반대 결과라 원인을 조사했다.

**원인**: 기존 `time_based_split(holdout_ratio=0.2)`는 train 기간
(2023-01-01~2025-01-01, 2년) 뒤쪽 20%를 그대로 자르는데, 이 20%가
**8~12월(약 5개월)에만 몰려 있고 1~7월이 전혀 없다**. 반면 실제 test
기간은 2025년 "전체"(1~12월, 모든 계절 포함). 계절성이 강한 데이터에서
검증 구간과 실제 평가 구간의 계절 구성이 다르면 holdout 점수가 실제
성능을 신뢰성 있게 예측하지 못한다 — 이 대회 데이터에서 정확히 그 일이
일어난 것.

**수정**: `scripts/validate_yearly_holdout.py` — train=이전 연도 전체,
holdout=그 다음 연도 전체로 나눠(`time_based_split_by_date`) test(2025년
전체)와 계절 구성이 같은 "연 단위" 검증으로 전환. 이 방식으로 오늘 시도한
세 버전(물리 feature only+기본파라미터 / 물리 feature only+튜닝된 파라미터
/ saturation+powercurve+튜닝된 파라미터)을 재검증한 결과:

| group | v1(기본) | v1b(튜닝만) | v2(saturation+튜닝, 오늘 제출본) |
|---|---|---|---|
| 1 | 0.6099 | 0.6036 | 0.6058 |
| 2 | 0.6453 | 0.6467 | 0.6460 |
| 3 | 0.5531 | 0.5555 | 0.5550 |
| 평균 | 0.6028 | 0.6019 | 0.6023 |

**세 버전이 사실상 통계적으로 구분 불가능(스프레드 0.001 수준)** — 즉 오늘
시도한 하이퍼파라미터 재튜닝과 saturation/power-curve feature는 기존
(계절 편중) holdout에서 보였던 ~0.02 개선폭이 실제로는 대부분 그 holdout
자체의 편향에 대한 과적합이었다는 뜻. 실제 리더보드 하락(0.606→0.6036)이
이를 뒷받침한다.

**결론 및 권장**: 오늘 제출한 v2 대신 기존에 선택되어 있던 v1(0.606, 원래
제출본)을 그대로 유지 권장. 하이퍼파라미터 튜닝(`tune_baseline.py`)과
feature 검증(`validate_baseline.py` 등)은 앞으로 반드시
`validate_yearly_holdout.py` 방식(연 단위 holdout)으로 다시 수행해야
신뢰할 수 있는 개선을 만들 수 있음. 오늘 진행한 앙상블/보정/objective/
sample_weight 검증들도 전부 기존 결함 있는 holdout으로 측정된 것이라
결론이 다시 뒤집힐 수 있음 — 향후 세션에서 연 단위 holdout으로 재검증 필요.

---

# 정격출력(90-100% capacity) 구간 오차 개선 시도 기록

리더보드 점수(0.606, 666등)가 낮았던 원인을 오차율 분포로 분석한 결과,
holdout 기준 평가대상 시간의 56.8%가 FICR 8% 임계치를 초과했고, 특히
설비용량 90-100% 구간에서 오차가 가장 컸다 (mean error rate ~18-30%,
8% 초과 비율 75-99%). 이 구간을 개선하기 위해 세 가지 방법을 시도했다.

## 1. 정격출력 구간 전용 feature 보강 (채택)

- `src/features.py::add_saturation_features` — 풍속 clip/sigmoid saturation/
  excess-over-threshold feature 추가
- `src/power_curve.py` — train 데이터로 적합한 isotonic 파워커브(풍속->발전량)
  feature 추가

결과: 전체 점수는 재튜닝 효과와 합쳐 소폭 상승(group1 +0.0013, group2 -0.0003,
group3 +0.0020)했지만, **목표했던 90-100% 구간 자체는 거의 개선되지 않음**
(mean_err 0.1779->0.1779, pct_over8 75.3%->75.1%, 변화 없음). 그래도 feature
자체는 정당하고 무해하므로 프로덕션에 유지.

## 2. 시드 앙상블(bagging) — 기각

`scripts/validate_ensemble.py`: 튜닝된 파라미터로 random_state만 바꾼
3~5개 모델을 평균. Holdout 결과:

| group | single | ensemble | delta |
|---|---|---|---|
| 1 | 0.6218 | 0.6215 | -0.0004 |
| 2 | 0.6407 | 0.6421 | +0.0014 |
| 3 | 0.6093 | 0.6032 | -0.0070 |

개선 효과가 없거나(group1) 오히려 악화(group3)되어 **프로덕션에 반영하지
않음**. 정격출력 구간(90-100%) 오차도 세 그룹 모두 개선되지 않거나 악화.

## 3. 예측값 기반 사후 보정(isotonic bias calibration) — 기각

`src/calibration.py`, `scripts/validate_calibration.py`: 앙상블 검증 중
발견한 사실 — 90-100% 구간에서 세 그룹 모두 예측이 실제보다 체계적으로
낮았다(under-prediction, capacity 대비 group1 -18%p, group2 -13%p,
group3 -30%p). 이를 겨냥해 calibration holdout(train 뒤쪽 20%)에서
(예측값 -> 실제값) 등단조회귀를 학습해 최종 예측을 보정하는 방식을 시도.

Holdout(outer, 즉 calibration holdout보다도 더 뒤쪽 시간대) 결과:

| group | raw | calibrated | delta |
|---|---|---|---|
| 1 | 0.6049 | 0.5940 | -0.0109 |
| 2 | 0.6429 | 0.5948 | -0.0481 |
| 3 | 0.5885 | 0.5901 | +0.0016 |

group3만 소폭 개선되고 group1/2는 큰 폭으로 악화. 원인으로 추정되는 것은
calibration holdout(train 뒤쪽 20%)의 기상 조건이 최종 holdout과 달라
isotonic 보정 함수가 그 구간의 특정 패턴에 과적합된 것으로 보임(계절성이
강한 시계열 데이터 특성상 단일 시간대 기반 calibration은 다른 시간대로
일반화가 잘 안 됨). **프로덕션에 반영하지 않음.**

## 결론

두 가지 사후 교정 시도(앙상블, calibration) 모두 일관된 개선을 만들지
못했다. 90-100% 구간의 근본 원인은 모델 구조나 사후 보정보다는 (a) 해당
구간에서 학습 샘플 자체가 희소하다는 점, (b) 극단적 고풍속 상황에서
LDAPS/GFS 예보 입력 자체의 정확도가 낮아질 가능성이 있다는 점(예보 오차가
전파됨) 쪽에 더 가까운 것으로 보인다. 현재는 saturation feature +
power curve + 튜닝된 단일 LightGBM 모델을 그대로 프로덕션으로 유지한다.

시도해볼 수 있는 후속 방향(미착수): K-fold 기반 다중 시간대 calibration(분산
감소), 하이퍼파라미터 탐색 확대(현재 그룹당 10회).

## 4. LDAPS/GFS 예보 풍속 자체의 정확도 검증 (원인 확인됨)

`scripts/analyze_forecast_accuracy.py`: SCADA `scada_mean_ws`(나셀 풍속계
실측)를 정답으로 삼아 예보 풍속(원시/허브높이 외삽)과 비교. 결과, **정격출력
(y/capacity) 구간이 높아질수록 예보 풍속의 under-forecast bias가 뚜렷하게,
거의 단조적으로 커진다** — 특히 GFS에서 두드러짐:

| group | band | ldaps_hub_speed bias | gfs_hub_speed bias |
|---|---|---|---|
| 1 | 0-10% | +0.80 | -1.60 |
| 1 | 90-100% | **-2.87** | **-6.58** |
| 2 | 0-10% | +0.57 | -1.66 |
| 2 | 90-100% | **-2.56** | **-7.20** |
| 3 | 0-10% | +1.40 | -0.78 |
| 3 | 90-100% | -0.31 | **-5.29** |

(단위: m/s, 음수=예보가 실제보다 풍속을 낮게 예측)

세 그룹 모두 고출력 구간에서 예보가 실제 풍속을 최대 5~7m/s까지 과소평가.
이는 수치예보모델(NWP)이 극단적 고풍속을 평활화(smoothing)하는 전형적
경향과 일치한다. **이 구간의 발전량 과소예측은 모델의 학습 문제가 아니라
입력 예보 데이터 자체의 체계적 한계에서 상당 부분 기인함이 확인됨.**

다음 단계로 고려할 수 있는 것: 예보값(입력) 자체에 대해 forecast->actual
등단조회귀 보정을 시도(이전에 실패한 "모델 출력값 보정"과 달리, 이건 NWP의
안정적인 체계적 편향을 보정하는 것이라 전체 train으로 fit하면 더 안정적일
가능성 있음 — 단, 검증 필요).

## 5. 예보 풍속 입력 자체 보정(wind bias correction) — 시도했으나 효과 미미

`src/wind_bias_correction.py`, `scripts/validate_wind_correction.py`: 위 4번
에서 확인한 예보풍속 under-forecast 편향을 직접 고치기 위해, `ldaps_hub_speed`/
`gfs_hub_speed` -> `scada_mean_ws`(실측) 등단조회귀 보정기를 **train_df
전체(80%, 여러 계절 포함)** 로 학습해 적용(이전 3번의 좁은 20% slice 문제를
피하려는 의도). Holdout 결과:

| group | baseline | corrected | delta |
|---|---|---|---|
| 1 | 0.6194 | 0.6189 | -0.0005 |
| 2 | 0.6367 | 0.6397 | +0.0030 |
| 3 | 0.6097 | 0.6008 | -0.0089 |

세 그룹 모두 변화폭이 작고 방향도 일관되지 않음 (90-100% 구간도 마찬가지로
소폭 개선/악화가 섞여 있음). **프로덕션에 반영하지 않음.**

**이유 추정(중요, 향후 유사 시도에 참고)**: LightGBM 같은 트리 기반 모델은
개별 feature의 단조(monotonic) 변환에 대해 사실상 불변이다 — 트리는 feature
값 자체가 아니라 최적 분할 임계값(threshold)을 학습하므로, `forecast_ws`를
그대로 쓰든 그것을 단조 함수로 보정한 값을 쓰든 모델이 찾아낼 수 있는
분할 능력은 이론적으로 거의 동일하다. 즉 "예보값을 물리적으로 더 정확하게
고쳐주는" 사후 보정은 이미 모델이 학습 과정에서 (근사적으로) 스스로 찾아낼
수 있는 정보라서 추가 이득이 거의 없다.

더 근본적인 신호는 편향(bias)이 아니라 **분산(noise)**이다:
`analyze_forecast_accuracy.py` 결과에서 고출력 구간은 편향뿐 아니라 MAE
자체도 크게 증가한다(예: group1 gfs_hub_speed MAE가 1.8m/s(0-10%
구간)에서 6.8~7.4m/s(90-100% 구간)로 증가). 등단조회귀 같은 평균 보정은
"평균적으로 얼마나 치우쳤는지"만 고칠 뿐 이 확산(산포)은 줄이지 못한다.
극단적 고풍속에서 예보 자체의 불확실성이 커지는 것은 같은 입력을 재가공하는
feature engineering만으로는 해소하기 어려운, 구조적인 한계로 보인다.

## 6. Objective function 변경 (L2 vs MAE vs Huber) — 기각

`scripts/validate_objective.py`: 대회 지표가 절대오차/임계값 기반이라 기본
L2(제곱오차) 대신 L1(MAE)이나 Huber가 더 잘 맞을 수 있다는 가설로 검증.
결과: 기본 L2가 세 그룹 모두에서 가장 좋았음 (group1 L2 0.6218 > MAE 0.6090
> Huber 0.4221; group3 L2 0.6093 > MAE 0.5955 > Huber 0.4335). Huber는
스케일(수천 kWh) 대비 delta 파라미터가 기본값이라 성능이 크게 떨어짐.
**기각, 현재 L2 유지.**

## 7. FICR 가중치 특성을 반영한 sample_weight 재학습 — 기각

`scripts/validate_sample_weight.py`: FICR = sum(a*unit_price)/sum(a*4) 로
실제 발전량(a)에 가중평균되는 지표라, 고출력 시간대 오차가 NMAE보다 FICR에
훨씬 크게 반영된다는 점에 착안해 `sample_weight = y/capacity`(또는 sqrt)로
학습 시 고출력 구간을 우선하도록 시도. 결과: NMAE는 소폭 개선되지만
(group3: 0.1332→0.1315) FICR이 오히려 악화되고(0.3518→0.3353) 90-100%
구간의 8% 초과 비율도 되레 증가(98.6%→100%). **기각.**

## 최종 결론 (갱신)

시도한 6가지(saturation/power-curve feature, 앙상블, 모델출력 보정, 예보입력
보정, objective 변경, FICR 가중 sample_weight) 중 saturation/power-curve만
전체 점수에 소폭 기여했고, 나머지 5가지는 효과가 없거나 불안정했다.
정격출력 구간의 오차는 예보 입력의 구조적 불확실성(분산 증가)에서 오는
것으로 보이며, 같은 입력 데이터와 같은 모델(LightGBM)을 재가공/재조정하는
접근으로는 한계에 도달한 것으로 판단. 이 구간을 더 파고들기보다는
하이퍼파라미터 탐색 확대, 서로 다른 모델 계열(XGBoost/CatBoost 등) 블렌딩,
혹은 다른 그룹3 업무로 리소스를 옮기는 것을 권장.
