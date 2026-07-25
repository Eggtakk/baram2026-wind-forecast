"""
SCADA 기반 커틀먼트/고장 의심 구간 탐지 및 제거.

scripts/diagnose_curtailment.py 분석 결과: 실측 풍속(scada_mean_ws)이
8m/s 이상인데 실제 발전량(y)이 경험적 파워커브(실측 풍속 기준 isotonic
회귀) 대비 설비용량의 30%p 이상 낮은 시간대가 그룹당 0.6~1.9% 존재하며,
겨울철(12~2월)에 집중되고 최대 83시간까지 이어지는 연속 구간도 있다 —
착빙, 커틀먼트, 계획정지 등 실제 운영 이벤트로 추정된다(측정 노이즈라기엔
너무 크고 길게 이어짐).

이런 시간대를 학습 데이터에 그대로 두면 모델이 "바람은 좋은데 발전량이
낮은" 사례를 정상적인 날씨-발전량 관계로 착각해 학습하게 되어, 대부분의
정상적인 시간대에 대한 예측 정확도가 오히려 떨어진다. train에서만
제거하고 holdout/test는 그대로 둔다(실제로 그런 이벤트가 있으면 예측이
틀릴 수밖에 없고, 그건 모델의 한계가 아니라 데이터의 한계이므로 인위적으로
숨기지 않는다).
"""
import pandas as pd
from sklearn.isotonic import IsotonicRegression

RESIDUAL_THRESHOLD_RATIO = 0.30
HIGH_WIND_THRESHOLD = 8.0


def flag_curtailment(
    df: pd.DataFrame,
    capacity: float,
    residual_threshold_ratio: float = RESIDUAL_THRESHOLD_RATIO,
    high_wind_threshold: float = HIGH_WIND_THRESHOLD,
) -> pd.Series:
    """df(반드시 y, scada_mean_ws 컬럼 포함)에서 커틀먼트/고장 의심 행에 대해
    True인 boolean Series를 반환한다 (df와 같은 index)."""
    valid = df.dropna(subset=["scada_mean_ws", "y"])
    curve = IsotonicRegression(y_min=0, y_max=capacity, increasing=True, out_of_bounds="clip")
    curve.fit(valid["scada_mean_ws"], valid["y"])

    est = curve.predict(df["scada_mean_ws"].fillna(0))
    residual_ratio = (df["y"] - est) / capacity
    mask = (residual_ratio < -residual_threshold_ratio) & (df["scada_mean_ws"].fillna(0) >= high_wind_threshold)
    return mask.fillna(False)


def remove_curtailment(df: pd.DataFrame, capacity: float, **kwargs) -> tuple[pd.DataFrame, int]:
    """의심 행을 제거한 df와 제거된 행 수를 반환."""
    mask = flag_curtailment(df, capacity, **kwargs)
    cleaned = df[~mask].reset_index(drop=True)
    return cleaned, int(mask.sum())
