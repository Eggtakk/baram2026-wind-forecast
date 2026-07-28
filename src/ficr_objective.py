"""
FICR 보상구조를 닮은 LightGBM custom objective (실험용).

배경: FICR = sum(actual*unit_price) / sum(actual*4), unit_price는
오차율 e = |pred-actual|/capacity 에 대한 계단함수(e<=0.06 -> 4원,
e<=0.08 -> 3원, e>0.08 -> 0원). rated_output_investigation.md 24번
섹션에서 capacity 스케일 Huber로 이 구조를 근사하려 했으나 실패 —
Huber는 경계를 넘어도 오차가 커질수록 페널티가 계속 커지는 선형
구조라, "8%를 넘으면 오차 크기와 무관하게 전부 0원"이라는 진짜 계단형
성격을 반영하지 못했다.

이 모듈은 unit_price(e)를 두 개의 로지스틱 sigmoid로 매끄럽게 근사해
(각각 6%, 8% 지점에서 전환) grad/hess를 직접 유도한 진짜 "보상 모양"
custom objective다. 추가로:
  - FICR의 실제 발전량(actual) 가중치를 그대로 반영(weight_by_actual).
  - 순수 FICR-shaped 항만 쓰면 오차가 이미 8%를 크게 넘은 표본(사실상
    "포기한" 구간)의 gradient/hessian이 거의 0으로 죽어 모델이 그
    표본들을 마음대로 발산시킬 위험이 있다 — 이를 막기 위해 표준 L2
    항을 약하게 섞어(lambda_l2) 항상 "정답 쪽으로 당기는 힘"이 남아있게
    한다.

수학적 유도
-----------
P(e) = low_price - a1*sigmoid((e-low)/T) - a2*sigmoid((e-high)/T)
  (a1 = low_price-mid_price, a2 = mid_price; P(0)≈low_price, P(inf)≈0)
Loss_i = -ficr_weight * weight_i * P(e_i) + lambda_l2 * 0.5*(pred_i-actual_i)^2/capacity
  (부호 반전: LightGBM은 손실을 최소화하므로 보상 P를 최대화하려면 -P를 최소화)

grad, hess는 e = |pred-actual|/capacity 를 통해 연쇄법칙으로 직접 유도
(이 파일의 docstring이 아니라 코드 자체가 유도 과정의 기록).

Huber(24번 섹션)와 다른 점: Huber는 경계 밖에서 오차가 커질수록 계속
불이익이 커지는 선형 구조지만, 이 objective는 gradient/hessian이 6%/8%
근방에서만 크고 그 밖(아주 안전하거나 이미 포기한 구간)에서는 거의
0으로 죽는다 — "임계값을 넘나드는 경계선 근처 표본"에 학습 신호를
집중시키는 게 목적.
"""
import numpy as np


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class _FicrObjective:
    """LGBMRegressor(objective=...)에 바로 넘길 수 있는 picklable callable.

    클로저(지역 함수) 대신 모듈 레벨 클래스로 구현한 이유: joblib/pickle은
    클로저를 직렬화하지 못해(`_pickle.PicklingError`) 학습된 모델을
    `.pkl`로 저장할 수 없었다 — 프로덕션에서는 모델을 저장했다가 나중에
    불러와 추론해야 하므로 반드시 picklable해야 한다.

    호출 시그니처: __call__(y_true, y_pred) -> (grad, hess)
    (scikit-learn API의 custom objective 규약)
    """

    def __init__(
        self,
        capacity: float,
        ficr_weight: float = 0.01,
        lambda_l2: float = 1.0,
        T: float = 0.01,
        low: float = 0.06,
        high: float = 0.08,
        low_price: float = 4.0,
        mid_price: float = 3.0,
        weight_by_actual: bool = True,
    ):
        self.capacity = capacity
        self.ficr_weight = ficr_weight
        self.lambda_l2 = lambda_l2
        self.T = T
        self.low = low
        self.high = high
        self.low_price = low_price
        self.mid_price = mid_price
        self.weight_by_actual = weight_by_actual
        self.a1 = low_price - mid_price  # 6%~8% 구간에서의 낙폭(4->3)
        self.a2 = mid_price               # 8% 초과 시 추가 낙폭(3->0)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        capacity, T = self.capacity, self.T
        low, high = self.low, self.high
        a1, a2 = self.a1, self.a2
        ficr_weight, lambda_l2 = self.ficr_weight, self.lambda_l2

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        r = y_pred - y_true
        s = np.sign(r)
        s[s == 0.0] = 1.0
        e = np.abs(r) / capacity

        x1 = (e - low) / T
        x2 = (e - high) / T
        g1 = _stable_sigmoid(x1)
        g2 = _stable_sigmoid(x2)
        g1p = g1 * (1 - g1)          # d(sigmoid)/dx
        g2p = g2 * (1 - g2)
        g1pp = g1p * (1 - 2 * g1)    # d^2(sigmoid)/dx^2
        g2pp = g2p * (1 - 2 * g2)

        weight = y_true if self.weight_by_actual else np.ones_like(y_true)

        # dP/de = -(a1*g1p + a2*g2p)/T,  de/dpred = s/capacity
        # d(-ficr_weight*weight*P)/dpred = ficr_weight*weight*s/capacity * (a1*g1p+a2*g2p)/T
        grad_ficr = ficr_weight * weight * s / capacity * (a1 * g1p + a2 * g2p) / T
        hess_ficr = ficr_weight * weight / (capacity**2) * (a1 * g1pp + a2 * g2pp) / (T**2)

        grad_l2 = lambda_l2 * r / capacity
        hess_l2 = lambda_l2 / capacity * np.ones_like(r)

        grad = grad_l2 + grad_ficr
        hess = hess_l2 + hess_ficr
        # 비볼록 손실이라 hess가 음수가 될 수 있음 -> 안정성을 위해 최소값 보장
        hess = np.maximum(hess, lambda_l2 / capacity * 0.1)

        # --- 스케일 보정 (버그 수정) ---
        # LightGBM의 leaf value는 -sum(grad)/(sum(hess)+reg_lambda)로 계산된다.
        # 지금까지의 grad/hess는 1/capacity 배로 축소되어 있어(hess_l2 ~ 4.6e-5),
        # reg_lambda(0에 가깝지 않은 값, 예: group3의 optuna 튜닝값 0.12)가
        # sum(hess)를 완전히 압도해 leaf value가 거의 0으로 짓눌리는 문제가 있었다
        # (reg_alpha/reg_lambda는 원래 LightGBM 네이티브 L2의 hess~2 스케일을
        # 가정하고 튜닝된 값이기 때문). grad, hess를 동일한 상수(capacity)로
        # 곱하면 grad/hess 비율(=순수 leaf value 모양)은 완전히 그대로 유지되면서
        # reg_lambda와의 상대적 스케일만 네이티브 L2와 맞게 복원된다.
        grad = grad * capacity
        hess = hess * capacity

        return grad, hess


def make_ficr_objective(
    capacity: float,
    ficr_weight: float = 0.01,
    lambda_l2: float = 1.0,
    T: float = 0.01,
    low: float = 0.06,
    high: float = 0.08,
    low_price: float = 4.0,
    mid_price: float = 3.0,
    weight_by_actual: bool = True,
):
    """`_FicrObjective` 인스턴스(picklable callable)를 만든다.

    반환 객체 호출 시그니처: obj(y_true, y_pred) -> (grad, hess)
    (scikit-learn API의 custom objective 규약)
    """
    return _FicrObjective(
        capacity, ficr_weight=ficr_weight, lambda_l2=lambda_l2, T=T,
        low=low, high=high, low_price=low_price, mid_price=mid_price,
        weight_by_actual=weight_by_actual,
    )
