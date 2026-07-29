"""
34번 섹션 후속 — FICR objective 하이퍼파라미터를 T=0.025/lambda_l2=0.5로
바꾼(기존 배포값은 T=0.01/lambda_l2=1.0) 새 stage-1 위에 정격출력 2단계
잔차보정의 threshold/stage2_params를 처음부터 재탐색한다.
`scripts/tune_residual_stage_quantile.py`(quantile stage-1 버전)와 병렬 구조.

배경: stage-1의 T/lambda_l2를 바꾸면 잔차(residual = y - oof_pred)의
분포가 기존 FICR objective(T=0.01/lambda_l2=1.0) 때와는 미묘하게 달라질
수 있어 재사용 없이 threshold 그리드 + stage2 Optuna를 처음부터 다시 돌린다.

실행 (레포 루트에서):
  python3 scripts/tune_residual_stage_ficr_v2.py prepare <group_id> [year ...]
  python3 scripts/tune_residual_stage_ficr_v2.py grid <group_id> [threshold ...]
  python3 scripts/tune_residual_stage_ficr_v2.py search <group_id> <threshold> [budget_sec]
  python3 scripts/tune_residual_stage_ficr_v2.py best <group_id> <threshold>
  python3 scripts/tune_residual_stage_ficr_v2.py validate <group_id>
캐시: /tmp/residual_stage_ficr_v2_cache_group{gid}.pkl
결과: experiments/baseline_lgbm/loyo_residual_stage_ficr_v2_results.json
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
import optuna
from optuna.samplers import TPESampler

from src.data_cleaning import remove_curtailment
from src.features import build_baseline_features, get_feature_cols
from src.ficr_objective import make_ficr_objective
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out
from validate_loyo import OUT_DIR, RECIPE_CHOICE, VALID_YEARS, build_physics_features, load_params
from validate_loyo_candidates import get_baseline_stats
from tune_residual_stage import eval_fold, MIN_REGIME_SAMPLES, suggest_stage2_params

CACHE_DIR = Path(tempfile.gettempdir())
LOCAL_DB_DIR = Path(tempfile.gettempdir()) / "baram_residual_optuna"
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = f"sqlite:///{LOCAL_DB_DIR / 'residual_optuna_ficr_v2.db'}"
SEED = 42
OOF_SPLITS = 3

# 34번 섹션 T/lambda_l2 그리드서치에서 확인된 최적점.
FICR_WEIGHT = {2: 0.008}
T = 0.025
LAMBDA_L2 = 0.5
THRESHOLD_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

RESULTS_PATH = OUT_DIR / "loyo_residual_stage_ficr_v2_results.json"
BEST_CONFIG_PATH_TMPL = str(OUT_DIR / "group{gid}_residual_stage_ficr_v2_best_config.json")


def cache_path(group_id: int) -> Path:
    return CACHE_DIR / f"residual_stage_ficr_v2_cache_group{group_id}.pkl"


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
    params["objective"] = make_ficr_objective(capacity, ficr_weight=FICR_WEIGHT[group_id], lambda_l2=LAMBDA_L2, T=T)
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
            "baseline_score": baseline_score,
        }
        print(f"  fold holdout={holdout_year}: ficr-v2-stage1-only score={baseline_score:.4f} "
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


def grid_search(group_id: int, thresholds=None):
    thresholds = thresholds or THRESHOLD_GRID
    cache = load_cache(group_id)
    folds = cache["folds"]
    baseline_scores = np.array([f["baseline_score"] for f in folds.values()])
    print(f"\n=== group{group_id} threshold 그리드서치 (FICR-v2 stage-1 단독 mean={baseline_scores.mean():.4f}) ===")
    default_stage2 = dict(
        n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=10,
        feature_fraction=0.8, bagging_fraction=0.8,
    )
    results = {}
    for thr in thresholds:
        scores, n_regimes = [], []
        for year, fold in folds.items():
            fold_with_gid = dict(fold, group_id=group_id)
            s, n = eval_fold(fold_with_gid, thr, default_stage2)
            scores.append(s)
            n_regimes.append(n)
        scores = np.array(scores)
        delta = scores.mean() - baseline_scores.mean()
        print(f"  threshold={thr:.2f}: mean={scores.mean():.4f} delta={delta:+.4f} n_regime={n_regimes}")
        results[thr] = {"mean": float(scores.mean()), "delta": float(delta)}
    best_thr = max(results, key=lambda t: results[t]["mean"])
    print(f"  -> 최고 threshold={best_thr:.2f}")
    return results, best_thr


def make_objective(group_id: int, cache: dict, threshold: float):
    folds = cache["folds"]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_stage2_params(trial)
        scores = []
        for year, fold in folds.items():
            fold_with_gid = dict(fold, group_id=group_id)
            s, _ = eval_fold(fold_with_gid, threshold, params)
            scores.append(s)
        scores = np.array(scores)
        trial.set_user_attr("fold_scores", scores.tolist())
        std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        trial.set_user_attr("fold_std", std)
        return float(scores.mean()) - 0.1 * std

    return objective


def search(group_id: int, threshold: float, budget_seconds: int = 30):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cache = load_cache(group_id)
    study_name = f"group{group_id}_residual_thr{threshold:.2f}_ficr_v2"
    study = optuna.create_study(
        study_name=study_name, storage=STORAGE, direction="maximize",
        sampler=TPESampler(seed=SEED), load_if_exists=True,
    )
    objective = make_objective(group_id, cache, threshold)
    study.optimize(objective, timeout=budget_seconds, show_progress_bar=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"[{study_name}] 누적 완료 시도: {len(completed)}")
    if completed:
        best = study.best_trial
        print(f"  BEST objective={best.value:.4f} fold_scores={best.user_attrs.get('fold_scores')}")
        print(f"  params={best.params}")


def best(group_id: int, threshold: float):
    study_name = f"group{group_id}_residual_thr{threshold:.2f}_ficr_v2"
    study = optuna.load_study(study_name=study_name, storage=STORAGE)
    b = study.best_trial
    print(f"[{study_name}] n_trials={len(study.trials)}  BEST objective={b.value:.4f}")
    print(f"  fold_scores={b.user_attrs.get('fold_scores')}")
    print(f"  params={b.params}")
    config = {"threshold": threshold, "stage2_params": b.params, "ficr_weight": FICR_WEIGHT[group_id], "T": T, "lambda_l2": LAMBDA_L2}
    path = Path(BEST_CONFIG_PATH_TMPL.format(gid=group_id))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"저장: {path}")
    return config


def validate(group_id: int):
    path = Path(BEST_CONFIG_PATH_TMPL.format(gid=group_id))
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold, stage2_params = config["threshold"], config["stage2_params"]

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
            "ficr_v2_stage1_only_score": fold["baseline_score"],
            "threshold": threshold, "n_regime_train": n_regime,
        })
        print(f"  holdout={year}: combined={score:.4f} ficr_v2_stage1_only={fold['baseline_score']:.4f} "
              f"delta_vs_L2_baseline={score-l2_base_mean:+.4f}")

    scores = np.array([r["score"] for r in fold_results])
    stage1_only = np.array([r["ficr_v2_stage1_only_score"] for r in fold_results])
    print(f"\n=== group{group_id} 최종 요약 ===")
    print(f"  L2 baseline: mean={l2_base_mean:.4f}")
    print(f"  FICR-v2 stage-1 단독: mean={stage1_only.mean():.4f} delta={stage1_only.mean()-l2_base_mean:+.4f}")
    print(f"  FICR-v2 stage-1 + 잔차보정: mean={scores.mean():.4f} delta={scores.mean()-l2_base_mean:+.4f} "
          f"fold_scores={scores.round(4).tolist()}")

    existing = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing = [r for r in existing if r["group_id"] != group_id]
    existing.extend(fold_results)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {RESULTS_PATH}")


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
    elif cmd == "grid":
        gid = int(args[1])
        thresholds = [float(a) for a in args[2:]] if len(args) > 2 else None
        grid_search(gid, thresholds)
    elif cmd == "search":
        gid, thr = int(args[1]), float(args[2])
        budget = int(args[3]) if len(args) > 3 else 30
        search(gid, thr, budget)
    elif cmd == "best":
        gid, thr = int(args[1]), float(args[2])
        best(gid, thr)
    elif cmd == "validate":
        validate(int(args[1]))
    else:
        raise SystemExit(f"알 수 없는 명령: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
