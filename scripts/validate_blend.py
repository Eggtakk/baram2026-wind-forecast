"""
physics-only / saturation+power-curve(full) 두 레시피 예측을 블렌딩(가중평균)했을 때
연 단위 holdout 점수가 단일 레시피보다 나아지는지 검증.

동일 모델(LightGBM)이라도 feature 구성이 다르면(physics-only vs full) 서로
다른 국소 패턴을 학습하므로, 예측 오차의 상관관계가 완전히 1은 아니다 —
같은 모델·같은 feature에 seed만 바꾼 예전 앙상블 시도(기각됨)와 달리 진짜
다양성을 기대할 수 있다.

각 그룹에 대해 physics/full 각각 (이미 확인된) 최적 파라미터로 학습한 뒤,
가중치 w in {0, 0.1, ..., 1.0} 로 블렌딩해 최고 조합을 찾는다 (w=0이면
full만, w=1이면 physics만 쓰는 것과 동일 — 그래서 기존 단일 레시피 결과도
자동으로 포함됨).

실행: (레포 루트에서) python3 scripts/validate_blend.py [group_id ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np

from src.features import (
    add_default_wind_features,
    add_lag_rolling_features,
    add_physics_features,
    add_time_features,
    build_baseline_features,
    get_feature_cols,
)
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_by_date

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
SPLIT_DATE = "2024-01-01"


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_best_params(group_id: int, recipe: str) -> dict:
    """physics/full 각각의 이제까지 발견된 최선 파라미터를 불러온다
    (optuna 결과가 있으면 그걸, 없으면 yearly 그리드 결과를 사용)."""
    optuna_path = OUT_DIR / f"group{group_id}_optuna_best_params_{recipe}.json"
    yearly_path = OUT_DIR / f"group{group_id}_yearly_best_params_{recipe}.json"

    candidates = []
    for p in [optuna_path, yearly_path]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                candidates.append((p, json.load(f)))
    return candidates


def fit_predict(train_df, holdout_df, feature_cols, params, group_id):
    params = dict(params)
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(train_df[feature_cols], train_df["y"])
    return model.predict(holdout_df[feature_cols]).clip(min=0)


def run_group(group_id: int):
    df = build_group_dataset(group_id, split="train")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    train_raw, holdout_raw = time_based_split_by_date(df, split_date=SPLIT_DATE)
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    # physics
    train_p = build_physics_features(train_raw)
    holdout_p = build_physics_features(holdout_raw)
    feat_p = get_feature_cols(train_p)
    cands_p = load_best_params(group_id, "physics")
    best_p_score, best_p_pred, best_p_src = -1, None, None
    for src, params in cands_p:
        pred = fit_predict(train_p, holdout_p, feat_p, params, group_id)
        r = validate_single_group(holdout_p["y"].to_numpy(), pred, group_id=group_id)
        score = 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]
        if score > best_p_score:
            best_p_score, best_p_pred, best_p_src = score, pred, src.name

    # full
    train_f = build_baseline_features(train_raw)
    holdout_f = build_baseline_features(holdout_raw)
    curve_models = fit_power_curve_models(train_f, capacity=capacity)
    train_f = apply_power_curve_models(train_f, curve_models)
    holdout_f = apply_power_curve_models(holdout_f, curve_models)
    feat_f = get_feature_cols(train_f)
    cands_f = load_best_params(group_id, "full")
    best_f_score, best_f_pred, best_f_src = -1, None, None
    for src, params in cands_f:
        pred = fit_predict(train_f, holdout_f, feat_f, params, group_id)
        r = validate_single_group(holdout_f["y"].to_numpy(), pred, group_id=group_id)
        score = 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]
        if score > best_f_score:
            best_f_score, best_f_pred, best_f_src = score, pred, src.name

    y_true = holdout_p["y"].to_numpy()  # holdout_p/holdout_f 같은 시간축이라 y 동일
    print(f"\n=== group{group_id} ===")
    print(f"  physics best: score={best_p_score:.4f} (src={best_p_src})")
    print(f"  full best:    score={best_f_score:.4f} (src={best_f_src})")

    best_w, best_blend_score = None, -1
    for w in np.arange(0.0, 1.01, 0.1):
        blend_pred = w * best_p_pred + (1 - w) * best_f_pred
        r = validate_single_group(y_true, blend_pred, group_id=group_id)
        score = 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]
        marker = " <-- best" if score > best_blend_score else ""
        if score > best_blend_score:
            best_blend_score, best_w = score, w
        print(f"  w(physics)={w:.1f}  blend score={score:.4f}{marker}")

    print(f"  => group{group_id} 최종: best_w={best_w:.1f}, blend_score={best_blend_score:.4f} "
          f"(단일 최고 {max(best_p_score, best_f_score):.4f} 대비 {best_blend_score - max(best_p_score, best_f_score):+.4f})")


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for gid in groups:
        run_group(gid)


if __name__ == "__main__":
    main()
