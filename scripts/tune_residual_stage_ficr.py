"""
FICR-shaped custom objective(25번 섹션)로 stage-1을 바꾼 뒤, 그 위에 기존
잔차보정(19~23번 섹션, threshold+stage2_params는 이미 튜닝된 값을 그대로
재사용)을 쌓았을 때도 여전히 도움이 되는지 LOYO로 검증한다.

배경: 25번 섹션에서 stage-1의 objective를 FICR-shaped로 바꾸는 것 자체는
group1(+0.0042)/group2(+0.0093) 모두 양성이었지만 아직 노이즈 수준이었고,
이미 실제 배포된 잔차보정(22·23번 섹션)은 이 새 stage-1 위에서 재도출된
적이 없었다 — "OOF 잔차 타깃"이 stage-1의 objective가 바뀌면 잔차의 통계적
성질 자체가 달라질 수 있어, 기존 stage-2(threshold+파라미터)가 여전히
유효한지는 별도 확인이 필요했다.

`scripts/tune_residual_stage.py`(OOF3, threshold/stage2 재탐색용)를 그대로
가져와 stage-1 fit에 FICR objective를 주입하도록만 바꿨다 — threshold와
stage2_params는 재탐색하지 않고 기존 `group{gid}_residual_stage_best_config_oof3.json`
값을 그대로 재사용한다(잔차의 "모양"이 stage-1 objective 교체로 근본적으로
달라지진 않을 것이라는 가정 하에, 우선 "여전히 도움되는가"만 저비용으로
확인 — 필요하면 나중에 재탐색 가능).

실행 (레포 루트에서):
  python3 scripts/tune_residual_stage_ficr.py prepare <group_id> [year ...]
  python3 scripts/tune_residual_stage_ficr.py validate <group_id>
캐시: /tmp/residual_stage_ficr_cache_group{gid}.pkl
결과: experiments/baseline_lgbm/loyo_residual_stage_ficr_combined_results.json
"""
import json
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import lightgbm as lgb

from src.data_cleaning import remove_curtailment
from src.features import build_baseline_features, get_feature_cols
from src.ficr_objective import make_ficr_objective
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out
from validate_loyo import OUT_DIR, RECIPE_CHOICE, VALID_YEARS, build_physics_features, load_params
from validate_loyo_candidates import get_baseline_stats, summarize
from tune_residual_stage import eval_fold, MIN_REGIME_SAMPLES  # noqa: F401 (재사용)

CACHE_DIR = Path(tempfile.gettempdir())
SEED = 42
OOF_SPLITS = 3

# 25번 섹션 LOYO 그리드서치에서 확인된 최적점(둘 다 아직 노이즈 수준이지만
# 방향이 가장 뚜렷했던 지점).
FICR_WEIGHT = {1: 0.003, 2: 0.008}
LAMBDA_L2 = 1.0
T = 0.01

RESULTS_PATH = OUT_DIR / "loyo_residual_stage_ficr_combined_results.json"
BEST_CONFIG_PATH_TMPL = str(OUT_DIR / "group{gid}_residual_stage_best_config_oof3.json")


def cache_path(group_id: int) -> Path:
    return CACHE_DIR / f"residual_stage_ficr_cache_group{group_id}.pkl"


def _fit_lgbm(X, y, params, seed=SEED):
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=seed, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(X, y)
    return model


def _build_features(cleaned_train, holdout_raw, capacity, recipe):
    if recipe == "physics":
        train_feat = build_physics_features(cleaned_train)
        holdout_feat = build_physics_features(holdout_raw)
    else:
        train_feat = build_baseline_features(cleaned_train)
        holdout_feat = build_baseline_features(holdout_raw)
        curve_models = fit_power_curve_models(train_feat, capacity=capacity)
        train_feat = apply_power_curve_models(train_feat, curve_models)
        holdout_feat = apply_power_curve_models(holdout_feat, curve_models)
    return train_feat, holdout_feat


def get_oof_stage1_preds(train_feat, feature_cols, params, n_splits: int = OOF_SPLITS):
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


def prepare(group_id: int, holdout_years=None):
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    base_params = load_params(group_id, recipe)
    params = dict(base_params)
    params["objective"] = make_ficr_objective(
        capacity, ficr_weight=FICR_WEIGHT[group_id], lambda_l2=LAMBDA_L2, T=T
    )
    df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)

    years = holdout_years if holdout_years else VALID_YEARS[group_id]

    fold_cache = {}
    path = cache_path(group_id)
    if path.exists():
        with open(path, "rb") as f:
            fold_cache = pickle.load(f)["folds"]

    for holdout_year in years:
        train_raw, holdout_raw = time_based_split_leave_year_out(
            df, holdout_year=holdout_year, valid_years=VALID_YEARS[group_id]
        )
        cleaned_train, _ = remove_curtailment(train_raw, capacity=capacity)
        train_feat, holdout_feat = _build_features(cleaned_train, holdout_raw, capacity, recipe)
        feature_cols = get_feature_cols(train_feat)

        stage1_model = _fit_lgbm(train_feat[feature_cols], train_feat["y"], params)
        stage1_holdout_pred = stage1_model.predict(holdout_feat[feature_cols]).clip(min=0)
        baseline_result = validate_single_group(holdout_feat["y"].to_numpy(), stage1_holdout_pred, group_id=group_id)
        baseline_score = 0.5 * baseline_result["one_minus_nmae"] + 0.5 * baseline_result["ficr"]

        oof_pred = get_oof_stage1_preds(train_feat, feature_cols, params, n_splits=OOF_SPLITS)

        fold_cache[holdout_year] = {
            "capacity": capacity,
            "feature_cols": feature_cols,
            "X_train": train_feat[feature_cols].reset_index(drop=True),
            "y_train": train_feat["y"].to_numpy(),
            "oof_pred": oof_pred,
            "X_holdout": holdout_feat[feature_cols].reset_index(drop=True),
            "y_holdout": holdout_feat["y"].to_numpy(),
            "stage1_holdout_pred": stage1_holdout_pred,
            "baseline_score": baseline_score,  # FICR objective stage-1 단독 점수(잔차보정 전)
        }
        print(f"  fold holdout={holdout_year}: ficr-stage1-only score={baseline_score:.4f} "
              f"n_train={len(train_feat)} n_holdout={len(holdout_feat)}")

    with open(cache_path(group_id), "wb") as f:
        pickle.dump({"group_id": group_id, "recipe": recipe, "folds": fold_cache}, f)
    print(f"캐시 저장(누적, {len(fold_cache)}개 폴드): {cache_path(group_id)}")


def load_cache(group_id: int):
    path = cache_path(group_id)
    if not path.exists():
        raise SystemExit(f"캐시가 없습니다. 먼저 `prepare {group_id}` 실행하세요: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def validate(group_id: int):
    config_path = Path(BEST_CONFIG_PATH_TMPL.format(gid=group_id))
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold = config["threshold"]
    stage2_params = config["stage2_params"]
    print(f"[group{group_id}] 기존 잔차보정 설정 재사용: threshold={threshold} stage2_params={stage2_params}")

    cache = load_cache(group_id)
    folds = cache["folds"]
    baseline_stats = get_baseline_stats()
    l2_base_mean = baseline_stats[group_id][0]

    fold_results = []
    for year, fold in folds.items():
        fold_with_gid = dict(fold, group_id=group_id)
        score, n_regime = eval_fold(fold_with_gid, threshold, stage2_params)
        fold_results.append({
            "group_id": group_id, "holdout_year": year, "score": score,
            "ficr_stage1_only_score": fold["baseline_score"],
            "l2_baseline_score_approx": l2_base_mean,
            "threshold": threshold, "n_regime_train": n_regime,
        })
        print(f"  holdout={year}: combined_score={score:.4f} "
              f"ficr_stage1_only={fold['baseline_score']:.4f} "
              f"delta_vs_L2_baseline_mean={score-l2_base_mean:+.4f}")

    existing = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing = [r for r in existing if r["group_id"] != group_id]
    existing.extend(fold_results)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    scores = np.array([r["score"] for r in fold_results])
    ficr_only = np.array([r["ficr_stage1_only_score"] for r in fold_results])
    l2_std = baseline_stats[group_id][1]
    print(f"\n=== group{group_id} 최종 요약 ===")
    print(f"  L2 baseline(스코어 산식 기존): mean={l2_base_mean:.4f} std={l2_std:.4f}")
    print(f"  FICR objective stage-1 단독:   mean={ficr_only.mean():.4f} delta={ficr_only.mean()-l2_base_mean:+.4f}")
    print(f"  FICR stage-1 + 잔차보정(결합): mean={scores.mean():.4f} delta={scores.mean()-l2_base_mean:+.4f}")
    print(f"결과 저장(누적): {RESULTS_PATH}")


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    cmd = args[0]
    if cmd == "prepare":
        rest = args[1:]
        gid = int(rest[0])
        years = [int(a) for a in rest[1:]] if len(rest) > 1 else None
        prepare(gid, years)
    elif cmd == "validate":
        validate(int(args[1]))
    else:
        raise SystemExit(f"알 수 없는 명령: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
