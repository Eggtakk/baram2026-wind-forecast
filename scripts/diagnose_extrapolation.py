"""
36번 섹션 후속 — test(2025) 기간이 train(2022~2024) 관측 범위보다 풍속이
높다는 게 확인됐으니, 실제로 배포된 stage-1 모델이 "관측 범위 밖 고풍속"에서
어떻게 행동하는지 직접 점검한다: 제작사 물리 파워커브(포화 곡선)에 합리적으로
수렴하는지, 아니면 이상하게 발산/진동하는지.

방법: test 데이터 중 풍속이 가장 높은 행들을 anchor로 삼아, 풍속 관련 컬럼
전부(원시 _speed, 세제곱, saturation, lag/rolling, 경험적 파워커브
curve_est feature까지)를 배율(scale factor)만큼 일관되게 스케일업한 뒤
모델 예측을 뽑아 풍속-발전량 곡선을 그린다. train에서 학습한 경험적
파워커브(src/power_curve.py, IsotonicRegression out_of_bounds="clip")는
관측 최댓값을 넘으면 그 지점 값으로 그대로 고정(clip)되므로, 원시 풍속/
saturation feature는 계속 올라가는데 이 feature 하나만 평평해지는 "신호
충돌" 상황이 생긴다 — 최종 모델이 그 충돌을 어떻게 처리하는지가 관심사.

실행: python3 scripts/diagnose_extrapolation.py <group_id>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import json
import numpy as np
import pandas as pd

from src.features import (
    add_default_wind_features,
    add_lag_rolling_features,
    add_physics_features,
    add_saturation_features,
    add_time_features,
    build_baseline_features,
    get_feature_cols,
    SATURATION_WIND_COLS,
    SATURATION_CLIP_SPEEDS,
)
from src.manufacturer_power_curve import estimate_group_power_kwh
from src.metrics import CAPACITY_KWH
from src.power_curve import apply_power_curve_models, load_power_curve_models
from src.preprocess import build_group_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "experiments" / "baseline_lgbm"

RAW_SPEED_COLS = [
    "ldaps_ws10_speed", "gfs_ws10_speed", "gfs_ws80_speed", "gfs_ws100_speed",
    "gfs_ws_pbl_speed", "ldaps_hub_speed", "gfs_hub_speed",
]
SCALES = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]

DEPLOYED = {
    1: dict(suffix="quantile", recipe="physics", model_type="residual"),
    2: dict(suffix="ficr", recipe="full", model_type="residual"),
    3: dict(suffix=None, recipe="full", model_type="l2_only"),
}


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def rescale_row_set(df: pd.DataFrame, scale: float, curve_models: dict | None) -> pd.DataFrame:
    """df의 풍속 관련 컬럼을 scale배 하고 파생 feature를 일관되게 재계산."""
    df = df.copy()

    # 1) 원시/허브고도 풍속 스케일
    for col in RAW_SPEED_COLS:
        if col in df.columns:
            df[col] = df[col] * scale
    # lag/rolling 풍속 파생도 동일 배율로(최근 며칠도 똑같이 windy했다고 가정)
    for col in df.columns:
        base = col.split("_lag")[0].split("_roll")[0]
        if (col.endswith(tuple(f"_lag{l}" for l in [1, 2, 3])) or "_roll" in col) and base in RAW_SPEED_COLS:
            if "_std" not in col:  # std는 배율 제곱이 아니라 단순 배율로 근사(부호 보존 위해)
                df[col] = df[col] * scale
            else:
                df[col] = df[col] * scale

    # 2) 세제곱류 재계산
    for col in ["ldaps_ws10_speed", "gfs_ws100_speed", "gfs_hub_speed"]:
        cubed_col = f"{col}_cubed"
        if cubed_col in df.columns and col in df.columns:
            df[cubed_col] = df[col] ** 3
    if "ldaps_power_proxy" in df.columns and {"ldaps_hub_speed", "ldaps_air_density"}.issubset(df.columns):
        df["ldaps_power_proxy"] = df["ldaps_air_density"] * df["ldaps_hub_speed"] ** 3

    # 3) saturation feature 재계산 (full 레시피만 해당)
    for col in SATURATION_WIND_COLS:
        if col not in df.columns:
            continue
        for clip in SATURATION_CLIP_SPEEDS:
            cc = f"{col}_clip{clip}"
            if cc in df.columns:
                df[cc] = df[col].clip(upper=clip)
        sat_col = f"{col}_saturation"
        if sat_col in df.columns:
            df[sat_col] = 1 / (1 + np.exp(-(df[col] - 8)))
        exc_col = f"{col}_excess8"
        if exc_col in df.columns:
            df[exc_col] = (df[col] - 8).clip(lower=0)

    # 4) 경험적(학습된) 파워커브 curve_est 재계산 -- out_of_bounds="clip"이라
    #    관측 최댓값 넘으면 그 지점에서 고정된다(핵심 관찰 대상).
    if curve_models:
        for col, model in curve_models.items():
            cc = f"{col}_curve_est"
            if col in df.columns and cc in df.columns:
                df[cc] = model.predict(df[col])

    return df


def main(group_id: int):
    cfg = DEPLOYED[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    recipe = cfg["recipe"]

    df_test = build_group_dataset(group_id, split="test")
    if recipe == "physics":
        df_test = build_physics_features(df_test)
        curve_models = None
    else:
        df_test = build_baseline_features(df_test)
        curve_path = MODEL_DIR / (
            f"group{group_id}_final_{cfg['suffix']}_power_curve.pkl" if cfg["suffix"]
            else f"group{group_id}_final_power_curve.pkl"
        )
        curve_models = load_power_curve_models(curve_path)
        df_test = apply_power_curve_models(df_test, curve_models)

    # 관측 train 최댓값(비교 기준선)
    df_train_raw = build_group_dataset(group_id, split="train")
    train_max_hub = (df_train_raw.pipe(add_default_wind_features).pipe(add_physics_features)["ldaps_hub_speed"]).max()
    print(f"=== group{group_id}: train ldaps_hub_speed 관측 최댓값 = {train_max_hub:.2f} m/s ===")

    # anchor: test에서 풍속 상위 20행
    anchor = df_test.nlargest(20, "ldaps_hub_speed").copy()
    print(f"anchor 20행 ldaps_hub_speed 범위: {anchor['ldaps_hub_speed'].min():.2f} ~ {anchor['ldaps_hub_speed'].max():.2f} m/s")

    # stage-1 (+잔차보정) 모델 로드
    suffix = cfg["suffix"]
    if suffix:
        meta = json.load(open(MODEL_DIR / f"group{group_id}_final_{suffix}_meta.json"))
        stage1_model = joblib.load(MODEL_DIR / f"group{group_id}_final_{suffix}_model.pkl")
    else:
        meta = json.load(open(MODEL_DIR / f"group{group_id}_final_meta.json"))
        stage1_model = joblib.load(MODEL_DIR / f"group{group_id}_final_model.pkl")
    feature_cols = meta["feature_cols"]

    stage2_model = None
    threshold = None
    if cfg["model_type"] == "residual":
        rmeta = json.load(open(MODEL_DIR / f"group{group_id}_residual_stage_{suffix}_meta.json"))
        stage2_model = joblib.load(MODEL_DIR / f"group{group_id}_residual_stage_{suffix}_model.pkl")
        threshold = rmeta["threshold"]

    print(f"\n{'배율':>6} {'평균hub_speed':>14} {'모델예측(kWh)':>14} {'물리커브(kWh)':>14} {'capacity대비%':>12}")
    for scale in SCALES:
        scaled = rescale_row_set(anchor, scale, curve_models)
        missing = [c for c in feature_cols if c not in scaled.columns]
        if missing:
            raise ValueError(f"누락 feature: {missing[:5]}")

        stage1_pred = stage1_model.predict(scaled[feature_cols]).clip(min=0, max=capacity)
        final_pred = stage1_pred.copy()
        if stage2_model is not None:
            regime_mask = (stage1_pred / capacity) >= threshold
            if regime_mask.sum() > 0:
                X2 = scaled.loc[regime_mask, feature_cols].copy()
                X2["stage1_pred"] = stage1_pred[regime_mask]
                correction = stage2_model.predict(X2)
                final_pred[regime_mask] = stage1_pred[regime_mask] + correction
        final_pred = np.clip(final_pred, 0, capacity)

        physics_est = estimate_group_power_kwh(
            scaled["ldaps_hub_speed"], scaled["ldaps_air_density"], group_id, capacity
        )
        physics_est = np.clip(physics_est, 0, capacity)

        mean_hub = scaled["ldaps_hub_speed"].mean()
        mean_pred = final_pred.mean()
        mean_phys = np.mean(physics_est)
        pct_cap = mean_pred / capacity * 100
        flag = " <- train 관측 범위 밖" if mean_hub > train_max_hub else ""
        print(f"{scale:>6.2f} {mean_hub:>14.2f} {mean_pred:>14.1f} {mean_phys:>14.1f} {pct_cap:>11.1f}%{flag}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
