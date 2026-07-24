"""
경험적(데이터 기반) 파워커브 feature.

터빈 파워커브는 특정 풍속(rated wind speed) 이상에서 출력이 설비용량에서
평평해지는 단조증가+포화 형태를 띤다. 정확한 rated wind speed를 그룹마다
직접 추정하는 대신, train 데이터(풍속 feature -> 실제 발전량 y)에 단조증가
회귀(isotonic regression)를 적합해 "이 풍속이면 대략 이 정도 발전량"이라는
경험적 커브를 만들고, 이를 하나의 강력한 feature로 추가한다.

주의(누수 방지): 반드시 **train(또는 train의 하위 split)만으로 fit**하고,
동일하게 적합된 모델을 train/holdout/test 모두에 apply해야 한다. holdout이나
test의 y를 fit에 사용하면 안 된다 — 이 모듈의 fit_* 함수들은 df에 있는 y를
그대로 쓰므로, 호출하는 쪽(train.py, validate_baseline.py 등)이 올바른
부분집합만 넘겨야 한다.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# 상관관계가 가장 높았던 풍속 feature들(EDA/베이스라인 feature importance 기준)
POWER_CURVE_WIND_COLS = ["ldaps_ws10_speed", "gfs_hub_speed"]


def fit_power_curve_models(df: pd.DataFrame, capacity: float, wind_cols: list[str] = POWER_CURVE_WIND_COLS) -> dict:
    """df(반드시 y 컬럼 포함, 학습에 쓸 부분만)로 풍속 -> 발전량 단조증가 커브를 적합.

    출력 범위를 [0, capacity]로 제한해 물리적으로 말이 안 되는 값이 나오지 않게 한다.
    """
    valid = df.dropna(subset=["y"])
    models = {}
    for col in wind_cols:
        if col not in df.columns:
            continue
        sub = valid.dropna(subset=[col])
        model = IsotonicRegression(y_min=0, y_max=capacity, increasing=True, out_of_bounds="clip")
        model.fit(sub[col], sub["y"])
        models[col] = model
    return models


def apply_power_curve_models(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """fit_power_curve_models()로 만든 커브를 df에 적용해 f"{col}_curve_est" feature를 추가."""
    df = df.copy()
    for col, model in models.items():
        if col in df.columns:
            df[f"{col}_curve_est"] = model.predict(df[col])
    return df


def save_power_curve_models(models: dict, path: Path):
    joblib.dump(models, path)


def load_power_curve_models(path: Path) -> dict:
    return joblib.load(path)
