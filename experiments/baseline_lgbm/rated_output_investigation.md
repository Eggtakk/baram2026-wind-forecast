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
감소), 하이퍼파라미터 탐색 확대(현재 그룹당 10회), LDAPS/GFS 예보 자체의
고풍속 구간 정확도 검증.
