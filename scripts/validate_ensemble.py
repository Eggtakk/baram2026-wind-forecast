"""
LightGBM 시드 앙상블(bagging) 검증 스크립트.

scripts/tune_baseline.py가 찾은 그룹별 최적 파라미터를 그대로 쓰되,
random_state만 바꾼 N개 모델을 학습해 예측을 평균하는 방식의 앙상블이
단일 모델 대비 holdout 성능(특히 정격출력 90-100% 구간)을 개선하는지 확인한다.

튜닝된 파라미터는 이미 bagging_fraction<1.0 또는 feature_fraction<1.0을
포함하므로(각 그룹 best_params.json 참고), random_state를 바꾸면 각 모델이
서로 다른 데이터/피처 부분집합으로 학습되어 실제로 다양성이 생긴다
(순수 배깅 앙상블).

실행: (레포 루트에서) python3 scripts/validate_ensemble.py [group_id ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import build_baseline_features, get_feature_cols
from src.metrics import CAPACITY_KWH, analyze_error_bands, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split

ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "experiments" / "baseline_lgbm"

HOLDOUT_RATIO = 0.2
N_SEEDS = 3
SEED_BASE = 100

DEFAULT_LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31)


def load_tuned_params(group_id: int) -> dict:
    params = dict(DEFAULT_LGBM_PARAMS)
    best_path = PARAM_DIR / f"group{group_id}_best_params.json"
    if best_path.exists():
        with open(best_path, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    return params


def run_group(group_id: int) -> dict:
    df = build_group_dataset(group_id, split="train")
    df = build_baseline_features(df)
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    train_df, holdout_df = time_based_split(df, holdout_ratio=HOLDOUT_RATIO)

    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    curve_models = fit_power_curve_models(train_df, capacity=capacity)
    train_df = apply_power_curve_models(train_df, curve_models)
    holdout_df = apply_power_curve_models(holdout_df, curve_models)

    feature_cols = get_feature_cols(train_df)
    base_params = load_tuned_params(group_id)
    bagging_freq = 1 if base_params.get("bagging_fraction", 1.0) < 1.0 else 0

    # --- 단일 모델(기존 방식, seed=42) ---
    single = lgb.LGBMRegressor(**base_params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    single.fit(train_df[feature_cols], train_df["y"])
    single_pred = single.predict(holdout_df[feature_cols]).clip(min=0)
    single_result = validate_single_group(holdout_df["y"].to_numpy(), single_pred, group_id=group_id)
    single_score = 0.5 * single_result["one_minus_nmae"] + 0.5 * single_result["ficr"]

    # --- N-시드 앙상블 ---
    preds = []
    for i in range(N_SEEDS):
        seed = SEED_BASE + i
        model = lgb.LGBMRegressor(**base_params, random_state=seed, bagging_freq=bagging_freq, verbosity=-1)
        model.fit(train_df[feature_cols], train_df["y"])
        preds.append(model.predict(holdout_df[feature_cols]).clip(min=0))
    ens_pred = np.mean(preds, axis=0)
    ens_result = validate_single_group(holdout_df["y"].to_numpy(), ens_pred, group_id=group_id)
    ens_score = 0.5 * ens_result["one_minus_nmae"] + 0.5 * ens_result["ficr"]

    single_bands = analyze_error_bands(holdout_df["y"].to_numpy(), single_pred, capacity)
    ens_bands = analyze_error_bands(holdout_df["y"].to_numpy(), ens_pred, capacity)

    return {
        "group_id": group_id,
        "single_score": single_score,
        "ens_score": ens_score,
        "single_result": single_result,
        "ens_result": ens_result,
        "single_bands": single_bands,
        "ens_bands": ens_bands,
    }


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for gid in groups:
        r = run_group(gid)
        print(f"\n=== group{gid} (N_SEEDS={N_SEEDS}) ===")
        print(f"  single: score={r['single_score']:.4f}  nmae={r['single_result']['nmae']:.4f}  ficr={r['single_result']['ficr']:.4f}")
        print(f"  ensemble: score={r['ens_score']:.4f}  nmae={r['ens_result']['nmae']:.4f}  ficr={r['ens_result']['ficr']:.4f}")
        print(f"  delta score: {r['ens_score'] - r['single_score']:+.4f}")

        sb = r["single_bands"]["overall"]
        eb = r["ens_bands"]["overall"]
        print(f"  [전체 구간] single >8%: {sb['pct_over8']:.3f} | ensemble >8%: {eb['pct_over8']:.3f}")

        band90 = "90-100%"
        s90 = r["single_bands"]["by_capacity_band"].loc[band90]
        e90 = r["ens_bands"]["by_capacity_band"].loc[band90]
        print(
            f"  [{band90} 구간, n={int(s90['n'])}] single mean_err={s90['mean_error_rate']:.4f} "
            f"pct_over8={s90['pct_over8']:.3f} bias={s90['mean_bias']:+.4f}"
        )
        print(
            f"  [{band90} 구간, n={int(e90['n'])}] ensemble mean_err={e90['mean_error_rate']:.4f} "
            f"pct_over8={e90['pct_over8']:.3f} bias={e90['mean_bias']:+.4f}"
        )


if __name__ == "__main__":
    main()
