"""
예측값 기반 사후 보정(post-hoc bias calibration).

앙상블(시드 평균) 검증 중 발견한 사실: 그룹1/2/3 모두 holdout의 정격출력
90-100% 구간에서 예측이 실제보다 체계적으로 낮았다(under-prediction,
설비용량 대비 group1 -18%p, group2 -13%p, group3 -30%p 수준). 이는 트리
기반 모델이 학습 데이터에서 드문 극단 구간(고출력)에서 평균으로 회귀하는
전형적인 경향이다.

이 모듈은 "모델이 이 정도로 예측했을 때 실제로는 평균적으로 얼마였는지"를
등단조회귀(isotonic regression)로 학습해 예측값을 사후 보정한다. 분류
문제의 isotonic calibration을 회귀에 적용한 것과 동일한 발상.

주의(누수 방지, 중요): 반드시 **모델 학습에 쓰지 않은 별도의 calibration
holdout**에서 나온 예측값-실제값 쌍으로 fit해야 한다. 모델이 학습에 쓴
데이터로 fit하면 in-sample residual이 0에 가까워 우리가 고치려는
under-prediction 패턴 자체가 보이지 않는다 (즉 보정 효과가 없다).
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_bias_calibrator(pred: np.ndarray, actual: np.ndarray, capacity: float) -> IsotonicRegression:
    """calibration holdout에서 얻은 (모델 원시 예측값 -> 실제 발전량) 쌍으로
    등단조회귀를 적합한다. pred/actual은 모델 학습에 쓰이지 않은 데이터에서
    나와야 한다."""
    model = IsotonicRegression(y_min=0, y_max=capacity, increasing=True, out_of_bounds="clip")
    model.fit(pred, actual)
    return model


def apply_bias_calibrator(pred: np.ndarray, calibrator: IsotonicRegression) -> np.ndarray:
    return calibrator.predict(pred)


def save_calibrator(calibrator: IsotonicRegression, path: Path):
    joblib.dump(calibrator, path)


def load_calibrator(path: Path) -> IsotonicRegression:
    return joblib.load(path)
