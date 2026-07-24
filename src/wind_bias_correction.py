"""
예보 풍속(허브높이 외삽값) 사후 보정.

scripts/analyze_forecast_accuracy.py 분석 결과: 정격출력(y/capacity) 구간이
높아질수록(=실제 풍속이 높을수록) LDAPS/GFS 예보가 SCADA 실측 풍속보다
체계적으로 낮게 예측한다(under-forecast). 수치예보모델(NWP)이 극단적
고풍속을 평활화(smoothing)하는 전형적 경향과 일치하며, 세 그룹 모두에서
일관되게 관찰됨 (자세한 수치는 experiments/baseline_lgbm/rated_output_investigation.md).

이 모듈은 "예보가 이 정도 풍속이라고 했을 때 실측은 평균적으로 얼마였는지"를
등단조회귀로 학습해 예보 자체(허브높이 외삽 풍속)를 보정한다. src.power_curve
및 이전에 시도했다가 기각한 src.calibration(모델 출력값 보정)과 발상은
비슷하지만 대상이 다르다:
  - src.calibration: "모델의 최종 예측 발전량"을 보정 -> 작은 시간대 slice로
    fit했더니 다른 시간대(holdout)로 일반화가 안 돼 기각됨.
  - 이 모듈: "예보 풍속 입력값" 자체를 보정 -> NWP 모델의 안정적인 체계적
    편향(계절과 무관하게 존재하는 물리적 특성)을 고치는 것이라 훨씬 큰
    표본(전체 train)으로 fit 가능하고 더 안정적일 것으로 기대. 검증은
    scripts/validate_wind_correction.py에서 수행.

주의(누수 방지): fit에는 반드시 SCADA가 있는 train(또는 train의 하위 split)만
사용한다. test 추론 시에는 이미 학습된 보정기를 "예보값 -> 보정값" 매핑으로만
적용하며 test의 실측치는 필요 없다(애초에 없음).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression

WIND_BIAS_COLS = ["ldaps_hub_speed", "gfs_hub_speed"]


def fit_wind_bias_correctors_from_raw(raw_df: pd.DataFrame, wind_cols: list[str] = WIND_BIAS_COLS) -> dict:
    """raw_df(=build_group_dataset(split='train')의 출력, scada_mean_ws 포함)로부터
    허브높이 예보풍속 -> 실측풍속 보정기를 학습한다.

    내부적으로 add_default_wind_features + add_physics_features를 한 번 더
    적용해 hub_speed를 계산한다(약간의 중복 계산이지만 벡터 연산이라 저렴하고,
    호출부(train.py 등)가 파이프라인 단계를 신경 쓰지 않아도 되게 해준다).
    """
    from src.features import add_default_wind_features, add_physics_features

    tmp = add_default_wind_features(raw_df)
    tmp = add_physics_features(tmp)

    if "scada_mean_ws" not in tmp.columns:
        raise ValueError("raw_df에 scada_mean_ws가 없습니다. build_group_dataset(..., include_scada=True)로 만든 train 데이터여야 합니다.")

    valid = tmp.dropna(subset=["scada_mean_ws"])
    correctors = {}
    for col in wind_cols:
        if col not in valid.columns:
            continue
        sub = valid.dropna(subset=[col])
        model = IsotonicRegression(y_min=0, increasing=True, out_of_bounds="clip")
        model.fit(sub[col], sub["scada_mean_ws"])
        correctors[col] = model
    return correctors


def apply_wind_bias_correction(df: pd.DataFrame, correctors: dict) -> pd.DataFrame:
    """correctors(fit_wind_bias_correctors_from_raw의 결과)를 df에 적용.

    add_physics_features()가 이미 실행되어 hub_speed/power_proxy/cubed
    컬럼이 존재하는 df에 대해 호출해야 한다 (build_baseline_features 내부에서
    add_physics_features 직후 호출됨). hub_speed를 보정값으로 덮어쓴 뒤,
    그 값에 의존하는 파생 feature(power_proxy, cubed)를 다시 계산한다.
    """
    df = df.copy()
    for col, corrector in correctors.items():
        if col in df.columns:
            df[col] = corrector.predict(df[col])

    if "ldaps_hub_speed" in correctors and {"ldaps_hub_speed", "ldaps_air_density"}.issubset(df.columns):
        df["ldaps_power_proxy"] = df["ldaps_air_density"] * df["ldaps_hub_speed"] ** 3

    for col in ["ldaps_hub_speed", "gfs_hub_speed"]:
        if col in correctors and f"{col}_cubed" in df.columns:
            df[f"{col}_cubed"] = df[col] ** 3

    return df


def save_wind_bias_correctors(correctors: dict, path: Path):
    joblib.dump(correctors, path)


def load_wind_bias_correctors(path: Path) -> dict:
    return joblib.load(path)
