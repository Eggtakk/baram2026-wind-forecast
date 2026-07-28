"""
정격출력 구간 2단계 잔차 모델(19번 섹션) 개선 — REGIME_THRESHOLD 그리드서치 +
stage-2 하이퍼파라미터 Optuna 튜닝.

배경: `scripts/validate_loyo_residual_stage.py`(REGIME_THRESHOLD=0.80 고정,
stage-2 파라미터 임의 고정)가 세 그룹 모두 baseline std를 넘지는 못했지만
가장 일관된 방향성(양수)을 보였다. threshold나 stage-2 파라미터가 최적이
아니었을 수 있어 재탐색한다.

효율화: stage-1(프로덕션과 동일) 모델과 OOF 예측은 REGIME_THRESHOLD나
stage-2 파라미터를 바꿔도 전혀 달라지지 않는다 — threshold를 바꾸면 stage-2
학습에 쓰는 "행의 부분집합"만 달라질 뿐이다. 그래서 그룹별로 LOYO 폴드마다
stage-1 fit(최종 1회 + OOF 2회)과 feature 생성을 **한 번만** 수행해
캐싱하고, threshold/stage-2 파라미터를 바꿔가며 재사용한다 — 매번 전체를
다시 계산하는 것보다 훨씬 빠르다.

실행 (레포 루트에서):
  python3 scripts/tune_residual_stage.py prepare <group_id>
  python3 scripts/tune_residual_stage.py grid <group_id>                # threshold 그리드서치(기본 stage-2 파라미터)
  python3 scripts/tune_residual_stage.py search <group_id> <threshold> [budget_sec]  # 그 threshold 고정, stage-2 파라미터 Optuna 탐색(반복 호출 가능)
  python3 scripts/tune_residual_stage.py best <group_id>                # 현재까지 최고 조합 출력/저장
  python3 scripts/tune_residual_stage.py validate <group_id>            # 최고 조합으로 최종 LOYO 재검증, baseline과 비교
캐시: /tmp/residual_stage_cache_group{gid}.pkl
study DB: /tmp/baram_residual_optuna/residual_optuna.db
결과: experiments/baseline_lgbm/loyo_residual_stage_tuned_results.json
"""
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

from src.data_cleaning import remove_curtailment
from src.features import build_baseline_features, get_feature_cols
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out
from validate_loyo import OUT_DIR, RECIPE_CHOICE, VALID_YEARS, build_physics_features, load_params
from validate_loyo_candidates import get_baseline_stats, summarize

CACHE_DIR = Path(tempfile.gettempdir())
LOCAL_DB_DIR = Path(tempfile.gettempdir()) / "baram_residual_optuna"
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = f"sqlite:///{LOCAL_DB_DIR / 'residual_optuna.db'}"
SEED = 42

MIN_REGIME_SAMPLES = 50
DEFAULT_STAGE2_PARAMS = dict(
    n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=10,
    feature_fraction=0.8, bagging_fraction=0.8,
)
THRESHOLD_GRID = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

TUNED_RESULTS_PATH = OUT_DIR / "loyo_residual_stage_tuned_oof3_results.json"
BEST_CONFIG_PATH_TMPL = str(OUT_DIR / "group{gid}_residual_stage_best_config_oof3.json")


OOF_SPLITS = 3  # 2-fold -> 3-fold로 확대(잔차 추정 안정성 개선, 2026-07-28)


def cache_path(group_id: int) -> Path:
    return CACHE_DIR / f"residual_stage_cache_group{group_id}_oof{OOF_SPLITS}.pkl"


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
    """train을 시간순으로 n_splits개 블록으로 나눠, 각 블록을 나머지 블록들로
    학습한 모델로 예측(leave-one-block-out) -> train 전체에 대한 out-of-fold
    stage-1 예측을 만든다. n_splits=2였던 이전 버전보다 각 fit에 쓰이는 학습
    데이터가 더 많아지고(각 블록 제외 후 나머지 (n_splits-1)/n_splits 비율)
    잔차 추정에 쓰이는 "안 본 예측"의 폴드 수도 늘어 잔차 추정이 더 안정적일
    것으로 기대."""
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
    """holdout_years를 생략하면 그룹의 모든 LOYO 폴드를 처리. 지정하면 그
    연도들만 (재)계산해 기존 캐시에 병합 — 그룹당 폴드 수가 많아 45초 예산
    안에 한 번에 다 못 돌 때 연도 단위로 나눠 호출하기 위함."""
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    params = load_params(group_id, recipe)
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
        print(f"  fold holdout={holdout_year}: baseline_score={baseline_score:.4f} "
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


def eval_fold(fold, threshold: float, stage2_params: dict):
    capacity = fold["capacity"]
    y_train, oof_pred = fold["y_train"], fold["oof_pred"]
    train_frac = y_train / capacity
    regime_mask_train = train_frac >= threshold
    n_regime_train = int(regime_mask_train.sum())

    stage1_holdout_pred = fold["stage1_holdout_pred"]
    if n_regime_train < MIN_REGIME_SAMPLES:
        final_pred = stage1_holdout_pred
    else:
        residual_train = y_train - oof_pred
        X_stage2 = fold["X_train"].loc[regime_mask_train].copy()
        X_stage2["stage1_pred"] = oof_pred[regime_mask_train]
        y_stage2 = residual_train[regime_mask_train]
        stage2_model = _fit_lgbm(X_stage2, y_stage2, stage2_params)

        holdout_frac = stage1_holdout_pred / capacity
        regime_mask_holdout = holdout_frac >= threshold
        final_pred = stage1_holdout_pred.copy()
        if regime_mask_holdout.sum() > 0:
            X_holdout_stage2 = fold["X_holdout"].loc[regime_mask_holdout].copy()
            X_holdout_stage2["stage1_pred"] = stage1_holdout_pred[regime_mask_holdout]
            correction = stage2_model.predict(X_holdout_stage2)
            final_pred[regime_mask_holdout] = stage1_holdout_pred[regime_mask_holdout] + correction
        final_pred = np.clip(final_pred, 0, capacity)

    result = validate_single_group(fold["y_holdout"], final_pred, group_id=fold.get("group_id", 0))
    score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]
    return score, n_regime_train


def grid_search(group_id: int, thresholds=None):
    thresholds = thresholds or THRESHOLD_GRID
    cache = load_cache(group_id)
    folds = cache["folds"]
    baseline_scores = np.array([f["baseline_score"] for f in folds.values()])
    print(f"\n=== group{group_id} threshold 그리드서치 (baseline mean={baseline_scores.mean():.4f}) ===")
    results = {}
    for thr in thresholds:
        scores = []
        n_regimes = []
        for year, fold in folds.items():
            fold_with_gid = dict(fold, group_id=group_id)
            s, n = eval_fold(fold_with_gid, thr, DEFAULT_STAGE2_PARAMS)
            scores.append(s)
            n_regimes.append(n)
        scores = np.array(scores)
        delta = scores.mean() - baseline_scores.mean()
        print(f"  threshold={thr:.2f}: mean={scores.mean():.4f} delta={delta:+.4f} "
              f"n_regime_train(폴드별)={n_regimes}")
        results[thr] = {"mean": float(scores.mean()), "delta": float(delta), "scores": scores.tolist()}
    best_thr = max(results, key=lambda t: results[t]["mean"])
    print(f"  -> 최고 threshold={best_thr:.2f} (delta={results[best_thr]['delta']:+.4f})")
    return results, best_thr


def suggest_stage2_params(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 30, 400, step=10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 3, 63),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
        feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


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


def search(group_id: int, threshold: float, budget_seconds: int = 25):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cache = load_cache(group_id)
    study_name = f"group{group_id}_residual_thr{threshold:.2f}_oof{OOF_SPLITS}"
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
        print(f"  BEST objective={best.value:.4f} fold_scores={best.user_attrs['fold_scores']}")
        print(f"  params={best.params}")


def best(group_id: int, threshold: float):
    study_name = f"group{group_id}_residual_thr{threshold:.2f}_oof{OOF_SPLITS}"
    study = optuna.load_study(study_name=study_name, storage=STORAGE)
    b = study.best_trial
    print(f"[{study_name}] n_trials={len(study.trials)}  BEST objective={b.value:.4f}")
    print(f"  fold_scores={b.user_attrs['fold_scores']}")
    print(f"  params={b.params}")
    config = {"threshold": threshold, "stage2_params": b.params}
    path = Path(BEST_CONFIG_PATH_TMPL.format(gid=group_id))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"저장: {path}")
    return config


def validate(group_id: int):
    path = Path(BEST_CONFIG_PATH_TMPL.format(gid=group_id))
    if not path.exists():
        raise SystemExit(f"먼저 `best {group_id} <threshold>` 실행하세요: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold = config["threshold"]
    stage2_params = config["stage2_params"]

    cache = load_cache(group_id)
    folds = cache["folds"]
    fold_results = []
    for year, fold in folds.items():
        fold_with_gid = dict(fold, group_id=group_id)
        score, n_regime = eval_fold(fold_with_gid, threshold, stage2_params)
        fold_results.append({
            "group_id": group_id, "holdout_year": year, "score": score,
            "baseline_score": fold["baseline_score"], "threshold": threshold,
            "n_regime_train": n_regime,
        })
        print(f"  holdout={year}: score={score:.4f} baseline={fold['baseline_score']:.4f} "
              f"delta={score-fold['baseline_score']:+.4f}")

    existing = []
    if TUNED_RESULTS_PATH.exists():
        with open(TUNED_RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing = [r for r in existing if r["group_id"] != group_id]
    existing.extend(fold_results)
    with open(TUNED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    baseline_stats = get_baseline_stats()
    summary = summarize("후보 9 (정격출력 2단계 잔차 모델, threshold+stage2 재탐색)", baseline_stats, existing)
    print(f"\n결과 저장(누적): {TUNED_RESULTS_PATH}")
    return summary


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    cmd = args[0]
    if cmd == "prepare":
        years = [int(a) for a in args[2:]] if len(args) > 2 else None
        prepare(int(args[1]), years)
    elif cmd == "grid":
        grid_search(int(args[1]))
    elif cmd == "search":
        gid, thr = int(args[1]), float(args[2])
        budget = int(args[3]) if len(args) > 3 else 25
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
