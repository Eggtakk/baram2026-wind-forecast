"""
39번 섹션 후속 — 지금 실제로 배포돼 있는 파이프라인(group1: quantile
objective(alpha=0.60) stage-1 + 잔차보정, group2: FICR-shaped
objective(T=0.01, lambda_l2=1.0, ficr_weight=0.008) stage-1 + 잔차보정,
group3: 배포본은 잔차보정 없는 L2 단독 = validate_loyo_quarterly.py의
baseline과 동일하므로 재검증 불필요)를 39번의 분기 LOQO(진짜 재학습,
group1/2 12-fold)로 재검증한다.

목적: "새 std(분기 LOQO 기준)의 1.5~2배는 넘어야 신뢰할 만하다"는 39번
결론을, 지금 실 서비스 중인 파이프라인 자체에도 적용해서 — 지금 배포된
게 그 기준을 넉넉히 넘는지, 아니면 사실 배포본조차 애매한 margin이었는지
확인한다.

로직은 scripts/validate_ficr_blend_full_pipeline.py의 run_fold(연 단위)를
분기 단위로 바꾼 것 — stage-1을 quantile/FICR objective로 학습하고,
3-fold OOF로 잔차보정(stage-2)용 residual target을 만들어 정격출력
구간(threshold 이상)에서만 잔차보정을 적용한 최종 예측을 평가한다.

45초 제약으로 한 번에 다 못 돌 수 있어 분기를 나눠 여러 번 호출 가능:
  python3 scripts/validate_loyo_quarterly_deployed.py 1 2022-1 2022-2
  ...
결과: experiments/baseline_lgbm/loyo_quarterly_deployed_results.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb

from src.data_cleaning import remove_curtailment
from src.features import (
    add_default_wind_features,
    add_lag_rolling_features,
    add_physics_features,
    add_time_features,
    build_baseline_features,
    get_feature_cols,
)
from src.ficr_objective import make_ficr_objective
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_quarter_out

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
RESULTS_PATH = OUT_DIR / "loyo_quarterly_deployed_results.json"

RECIPE_CHOICE = {1: "physics", 2: "full"}
PARAMS_SOURCE = {1: "yearly", 2: "yearly"}
VALID_YEARS = {1: [2022, 2023, 2024], 2: [2022, 2023, 2024]}

# 실제 배포 설정 그대로.
STAGE1_STYLE = {1: "quantile", 2: "ficr"}
QUANTILE_ALPHA = {1: 0.60}
FICR_WEIGHT = {2: 0.008}
CONFIG_SUFFIX = {1: "quantile", 2: "ficr"}
MIN_REGIME_SAMPLES = 50
OOF_SPLITS = 3


def all_quarters(group_id: int) -> list[tuple[int, int]]:
    return [(y, q) for y in VALID_YEARS[group_id] for q in [1, 2, 3, 4]]


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_params(group_id: int, recipe: str) -> dict:
    source = PARAMS_SOURCE[group_id]
    path = OUT_DIR / f"group{group_id}_{source}_best_params_{recipe}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_stage1_params(group_id: int, recipe: str) -> dict:
    params = dict(load_params(group_id, recipe))
    style = STAGE1_STYLE[group_id]
    if style == "quantile":
        params["objective"] = "quantile"
        params["alpha"] = QUANTILE_ALPHA[group_id]
    elif style == "ficr":
        capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
        params["objective"] = make_ficr_objective(capacity, ficr_weight=FICR_WEIGHT[group_id], lambda_l2=1.0, T=0.01)
    return params


def _fit_lgbm(X, y, params, seed=42):
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=seed, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(X, y)
    return model


def get_oof_stage1_preds(train_feat, feature_cols, params, n_splits=OOF_SPLITS):
    n = len(train_feat)
    idx = np.arange(n)
    blocks = np.array_split(idx, n_splits)
    oof = np.zeros(n)
    for i in range(n_splits):
        test_idx = blocks[i]
        train_idx = np.concatenate([blocks[j] for j in range(n_splits) if j != i])
        model = _fit_lgbm(train_feat.iloc[train_idx][feature_cols], train_feat.iloc[train_idx]["y"], params)
        oof[test_idx] = model.predict(train_feat.iloc[test_idx][feature_cols]).clip(min=0)
    return oof


def run_fold(group_id: int, df, holdout_quarter: tuple[int, int], threshold: float, stage2_params: dict) -> dict:
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    train_raw, holdout_raw = time_based_split_leave_quarter_out(
        df, holdout_quarter=holdout_quarter, valid_quarters=all_quarters(group_id)
    )
    cleaned_train, n_removed = remove_curtailment(train_raw, capacity=capacity)

    if recipe == "physics":
        train_feat = build_physics_features(cleaned_train)
        holdout_feat = build_physics_features(holdout_raw)
    else:
        train_feat = build_baseline_features(cleaned_train)
        holdout_feat = build_baseline_features(holdout_raw)
        curve_models = fit_power_curve_models(train_feat, capacity=capacity)
        train_feat = apply_power_curve_models(train_feat, curve_models)
        holdout_feat = apply_power_curve_models(holdout_feat, curve_models)

    feature_cols = get_feature_cols(train_feat)
    stage1_params = get_stage1_params(group_id, recipe)

    stage1_model = _fit_lgbm(train_feat[feature_cols], train_feat["y"], stage1_params)
    stage1_holdout_pred = stage1_model.predict(holdout_feat[feature_cols]).clip(min=0, max=capacity)
    oof_pred = get_oof_stage1_preds(train_feat, feature_cols, stage1_params)

    y_train = train_feat["y"].to_numpy()
    train_frac = y_train / capacity
    regime_mask_train = train_frac >= threshold
    final_pred = stage1_holdout_pred.copy()
    if regime_mask_train.sum() >= MIN_REGIME_SAMPLES:
        residual_train = y_train - oof_pred
        X_stage2 = train_feat.loc[regime_mask_train, feature_cols].copy()
        X_stage2["stage1_pred"] = oof_pred[regime_mask_train]
        y_stage2 = residual_train[regime_mask_train]
        stage2_model = _fit_lgbm(X_stage2, y_stage2, stage2_params)

        holdout_frac = stage1_holdout_pred / capacity
        regime_mask_holdout = holdout_frac >= threshold
        if regime_mask_holdout.sum() > 0:
            X_holdout_stage2 = holdout_feat.loc[regime_mask_holdout, feature_cols].copy()
            X_holdout_stage2["stage1_pred"] = stage1_holdout_pred[regime_mask_holdout]
            correction = stage2_model.predict(X_holdout_stage2)
            final_pred[regime_mask_holdout] = stage1_holdout_pred[regime_mask_holdout] + correction
    final_pred = np.clip(final_pred, 0, capacity)

    y_holdout = holdout_feat["y"].to_numpy()
    result = validate_single_group(y_holdout, final_pred, group_id=group_id)
    score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]

    return {
        "group_id": group_id,
        "holdout_year": holdout_quarter[0],
        "holdout_quarter": holdout_quarter[1],
        "score": score,
        "nmae": result["nmae"],
        "one_minus_nmae": result["one_minus_nmae"],
        "ficr": result["ficr"],
        "n_train": len(train_feat),
        "n_holdout": len(holdout_feat),
        "n_eval": result["n_eval"],
        "n_curtailment_removed": n_removed,
    }


def parse_quarter_tokens(tokens: list[str]) -> list[tuple[int, int]]:
    out = []
    for t in tokens:
        y, q = t.split("-")
        out.append((int(y), int(q)))
    return out


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit("사용법: validate_loyo_quarterly_deployed.py <group_id> [YYYY-Q ...]")
    group_id = int(args[0])
    quarter_tokens = args[1:]
    quarters = parse_quarter_tokens(quarter_tokens) if quarter_tokens else all_quarters(group_id)

    suffix = CONFIG_SUFFIX[group_id]
    config_path = OUT_DIR / f"group{group_id}_residual_stage_{suffix}_best_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold, stage2_params = config["threshold"], config["stage2_params"]

    all_results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    done = {(r["group_id"], r["holdout_year"], r["holdout_quarter"]) for r in all_results}
    todo = [q for q in quarters if (group_id, q[0], q[1]) not in done]

    if not todo:
        print(f"group{group_id}: 요청된 분기 전부 이미 처리됨 (스킵)")
    else:
        df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)
        print(f"=== group{group_id} 배포 설정(stage1={STAGE1_STYLE[group_id]}, threshold={threshold}) "
              f"분기 LOQO ({len(todo)}개 분기 처리) ===")
        for hq in todo:
            t0 = time.time()
            r = run_fold(group_id, df, hq, threshold, stage2_params)
            all_results.append(r)
            print(f"  holdout={hq[0]}-Q{hq[1]}: score={r['score']:.4f} "
                  f"(1-NMAE={r['one_minus_nmae']:.4f}, FICR={r['ficr']:.4f}) [{time.time()-t0:.1f}s]")
            # 폴드마다 즉시 저장 -- 45초 제약으로 배치 도중 타임아웃돼도 진행 상황을 잃지 않기 위함.
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"저장: {RESULTS_PATH}")

    print("\n=== 누적 요약 (배포 파이프라인, 분기 LOQO) ===")
    for gid in sorted({r["group_id"] for r in all_results}):
        scores = np.array([r["score"] for r in all_results if r["group_id"] == gid])
        n = len(scores)
        expected = len(all_quarters(gid))
        std = scores.std(ddof=1) if n > 1 else 0.0
        print(f"  group{gid}: n={n}/{expected} mean={scores.mean():.4f} std={std:.4f}")


if __name__ == "__main__":
    main()
