"""
43번 섹션 후속 — group1 stage-1 앙상블(L2+quantile+FICR 균등)에 얹은
잔차보정을 "quantile 단독용으로 튜닝된 설정 재사용"이 아니라, **이
앙상블의 OOF residual 분포에 맞춰 threshold/stage2 하이퍼파라미터를
처음부터 재탐색**한다. 검증은 분기 LOQO(12-fold, 진짜 재학습) 그대로.

`scripts/tune_residual_stage_quantile.py`(연 단위, quantile 단독)와
`scripts/tune_residual_stage.py`(eval_fold/suggest_stage2_params 등
공용 유틸)를 재사용하되, prepare 단계만 분기 분할 + 3-objective 앙상블
stage-1/OOF로 바꾼다 — stage-1이 앙상블 3개 모델(각 3-fold OOF)이라
폴드당 12회 LightGBM 학습이 필요해(quantile 단독의 12번 섹션 캐싱 방식
대비 훨씬 비쌈) prepare는 반드시 폴드 1~2개씩 나눠 호출.

실행 (레포 루트에서):
  python3 scripts/tune_residual_stage_ensemble_quarterly.py prepare [YYYY-Q ...]
  python3 scripts/tune_residual_stage_ensemble_quarterly.py grid [threshold ...]
  python3 scripts/tune_residual_stage_ensemble_quarterly.py search <threshold> [budget_sec]
  python3 scripts/tune_residual_stage_ensemble_quarterly.py best <threshold>
  python3 scripts/tune_residual_stage_ensemble_quarterly.py validate
캐시: /tmp/residual_stage_ensemble_quarterly_cache_group1.pkl
결과: experiments/baseline_lgbm/loyo_quarterly_residual_stage_ensemble_results.json
      experiments/baseline_lgbm/group1_residual_stage_ensemble_best_config.json
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
from src.features import (
    add_default_wind_features,
    add_lag_rolling_features,
    add_physics_features,
    add_time_features,
    get_feature_cols,
)
from src.ficr_objective import make_ficr_objective
from src.metrics import CAPACITY_KWH, validate_single_group
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_quarter_out
from tune_residual_stage import eval_fold, suggest_stage2_params, MIN_REGIME_SAMPLES

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
CACHE_DIR = Path(tempfile.gettempdir())
LOCAL_DB_DIR = Path(tempfile.gettempdir()) / "baram_residual_optuna"
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = f"sqlite:///{LOCAL_DB_DIR / 'residual_optuna_ensemble_quarterly.db'}"
SEED = 42
OOF_SPLITS = 3

GROUP_ID = 1
RECIPE = "physics"
PARAMS_SOURCE = "yearly"
VALID_YEARS = [2022, 2023, 2024]
QUANTILE_ALPHA = 0.60
FICR_WEIGHT, LAMBDA_L2, T = 0.003, 1.0, 0.01
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
THRESHOLD_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

CACHE_PATH = CACHE_DIR / "residual_stage_ensemble_quarterly_cache_group1.pkl"
RESULTS_PATH = OUT_DIR / "loyo_quarterly_residual_stage_ensemble_results.json"
BEST_CONFIG_PATH = OUT_DIR / "group1_residual_stage_ensemble_best_config.json"


def all_quarters():
    return [(y, q) for y in VALID_YEARS for q in [1, 2, 3, 4]]


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_base_params():
    path = OUT_DIR / f"group{GROUP_ID}_{PARAMS_SOURCE}_best_params_{RECIPE}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fit_lgbm(X, y, params, seed=SEED):
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=seed, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(X, y)
    return model


def stage1_param_variants(base_params, capacity):
    l2_params = dict(base_params)
    q_params = dict(base_params)
    q_params["objective"] = "quantile"
    q_params["alpha"] = QUANTILE_ALPHA
    f_params = dict(base_params)
    f_params["objective"] = make_ficr_objective(capacity, ficr_weight=FICR_WEIGHT, lambda_l2=LAMBDA_L2, T=T)
    return [l2_params, q_params, f_params]


def get_oof_ensemble_preds(train_feat, feature_cols, param_list, weights, n_splits=OOF_SPLITS):
    n = len(train_feat)
    idx = np.arange(n)
    blocks = np.array_split(idx, n_splits)
    oof_per_objective = [np.zeros(n) for _ in param_list]
    for i in range(n_splits):
        test_idx = blocks[i]
        train_idx = np.concatenate([blocks[j] for j in range(n_splits) if j != i])
        for oi, params in enumerate(param_list):
            model = _fit_lgbm(train_feat.iloc[train_idx][feature_cols], train_feat.iloc[train_idx]["y"], params)
            oof_per_objective[oi][test_idx] = model.predict(train_feat.iloc[test_idx][feature_cols]).clip(min=0)
    return sum(w * o for w, o in zip(weights, oof_per_objective))


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)


def prepare(quarters=None):
    quarters = quarters or all_quarters()
    capacity = CAPACITY_KWH[f"kpx_group_{GROUP_ID}"]
    base_params = load_base_params()
    df = build_group_dataset(GROUP_ID, split="train").dropna(subset=["y"]).reset_index(drop=True)

    fold_cache = load_cache()
    for hq in quarters:
        train_raw, holdout_raw = time_based_split_leave_quarter_out(
            df, holdout_quarter=hq, valid_quarters=all_quarters()
        )
        cleaned_train, _ = remove_curtailment(train_raw, capacity=capacity)
        train_feat = build_physics_features(cleaned_train)
        holdout_feat = build_physics_features(holdout_raw)
        feature_cols = get_feature_cols(train_feat)
        param_list = stage1_param_variants(base_params, capacity)

        holdout_preds = []
        for params in param_list:
            model = _fit_lgbm(train_feat[feature_cols], train_feat["y"], params)
            holdout_preds.append(model.predict(holdout_feat[feature_cols]).clip(min=0, max=capacity))
        ensemble_holdout_pred = sum(w * p for w, p in zip(WEIGHTS, holdout_preds))
        ensemble_oof_pred = get_oof_ensemble_preds(train_feat, feature_cols, param_list, WEIGHTS)

        baseline_result = validate_single_group(holdout_feat["y"].to_numpy(), ensemble_holdout_pred, group_id=GROUP_ID)
        baseline_score = 0.5 * baseline_result["one_minus_nmae"] + 0.5 * baseline_result["ficr"]

        fold_cache[hq] = {
            "capacity": capacity,
            "feature_cols": feature_cols,
            "X_train": train_feat[feature_cols].reset_index(drop=True),
            "y_train": train_feat["y"].to_numpy(),
            "oof_pred": ensemble_oof_pred,
            "X_holdout": holdout_feat[feature_cols].reset_index(drop=True),
            "y_holdout": holdout_feat["y"].to_numpy(),
            "stage1_holdout_pred": ensemble_holdout_pred,
            "baseline_score": baseline_score,
        }
        print(f"  fold {hq[0]}-Q{hq[1]}: 앙상블-stage1-only score={baseline_score:.4f} "
              f"n_train={len(train_feat)} n_holdout={len(holdout_feat)}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(fold_cache, f)
    print(f"캐시 저장(누적, {len(fold_cache)}/12 폴드): {CACHE_PATH}")


def grid_search(thresholds=None):
    thresholds = thresholds or THRESHOLD_GRID
    folds = load_cache()
    if len(folds) < 12:
        print(f"경고: 캐시에 {len(folds)}/12 폴드만 있음 — prepare로 마저 채우세요.")
    baseline_scores = np.array([f["baseline_score"] for f in folds.values()])
    print(f"\n=== group1 앙상블 threshold 그리드서치 (stage1-only mean={baseline_scores.mean():.4f}, n={len(folds)}) ===")
    default_stage2 = dict(
        n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=10,
        feature_fraction=0.8, bagging_fraction=0.8,
    )
    results = {}
    for thr in thresholds:
        scores, n_regimes = [], []
        for hq, fold in folds.items():
            fold_with_gid = dict(fold, group_id=GROUP_ID)
            s, n = eval_fold(fold_with_gid, thr, default_stage2)
            scores.append(s)
            n_regimes.append(n)
        scores = np.array(scores)
        delta = scores.mean() - baseline_scores.mean()
        print(f"  threshold={thr:.2f}: mean={scores.mean():.4f} std={scores.std(ddof=1):.4f} "
              f"delta={delta:+.4f} n_regime_range=[{min(n_regimes)},{max(n_regimes)}]")
        results[thr] = {"mean": float(scores.mean()), "delta": float(delta)}
    best_thr = max(results, key=lambda t: results[t]["mean"])
    print(f"  -> 최고 threshold={best_thr:.2f}")
    return results, best_thr


def make_objective(threshold: float):
    folds = load_cache()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_stage2_params(trial)
        scores = []
        for hq, fold in folds.items():
            fold_with_gid = dict(fold, group_id=GROUP_ID)
            s, _ = eval_fold(fold_with_gid, threshold, params)
            scores.append(s)
        scores = np.array(scores)
        trial.set_user_attr("fold_scores", scores.tolist())
        std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        trial.set_user_attr("fold_std", std)
        return float(scores.mean()) - 0.1 * std

    return objective


def search(threshold: float, budget_seconds: int = 30):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = f"group1_residual_ensemble_thr{threshold:.2f}"
    study = optuna.create_study(
        study_name=study_name, storage=STORAGE, direction="maximize",
        sampler=TPESampler(seed=SEED), load_if_exists=True,
    )
    objective = make_objective(threshold)
    study.optimize(objective, timeout=budget_seconds, show_progress_bar=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"[{study_name}] 누적 완료 시도: {len(completed)}")
    if completed:
        best_trial = study.best_trial
        print(f"  BEST objective={best_trial.value:.4f} fold_scores={best_trial.user_attrs.get('fold_scores')}")
        print(f"  params={best_trial.params}")


def best(threshold: float):
    study_name = f"group1_residual_ensemble_thr{threshold:.2f}"
    study = optuna.load_study(study_name=study_name, storage=STORAGE)
    b = study.best_trial
    print(f"[{study_name}] n_trials={len(study.trials)}  BEST objective={b.value:.4f}")
    print(f"  fold_scores={b.user_attrs.get('fold_scores')}")
    print(f"  params={b.params}")
    config = {"threshold": threshold, "stage2_params": b.params, "ensemble_weights": {"l2": WEIGHTS[0], "quantile": WEIGHTS[1], "ficr": WEIGHTS[2]}}
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"저장: {BEST_CONFIG_PATH}")
    return config


def validate():
    with open(BEST_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold, stage2_params = config["threshold"], config["stage2_params"]

    folds = load_cache()
    fold_results = []
    for hq, fold in folds.items():
        fold_with_gid = dict(fold, group_id=GROUP_ID)
        score, n_regime = eval_fold(fold_with_gid, threshold, stage2_params)
        fold_results.append({
            "group_id": GROUP_ID, "holdout_year": hq[0], "holdout_quarter": hq[1],
            "score": score, "ensemble_stage1_only_score": fold["baseline_score"],
            "threshold": threshold, "n_regime_train": n_regime,
        })
        print(f"  holdout={hq[0]}-Q{hq[1]}: combined={score:.4f} "
              f"stage1_only={fold['baseline_score']:.4f}")

    scores = np.array([r["score"] for r in fold_results])
    stage1_only = np.array([r["ensemble_stage1_only_score"] for r in fold_results])
    print(f"\n=== group1 앙상블+잔차보정(재탐색) 최종 요약 (n={len(fold_results)}) ===")
    print(f"  stage-1 단독: mean={stage1_only.mean():.4f} std={stage1_only.std(ddof=1):.4f}")
    print(f"  stage-1+잔차보정(재탐색): mean={scores.mean():.4f} std={scores.std(ddof=1):.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {RESULTS_PATH}")


def parse_quarter_tokens(tokens):
    return [(int(t.split("-")[0]), int(t.split("-")[1])) for t in tokens]


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    cmd = args[0]
    if cmd == "prepare":
        quarters = parse_quarter_tokens(args[1:]) if len(args) > 1 else None
        prepare(quarters)
    elif cmd == "grid":
        thresholds = [float(a) for a in args[1:]] if len(args) > 1 else None
        grid_search(thresholds)
    elif cmd == "search":
        thr = float(args[1])
        budget = int(args[2]) if len(args) > 2 else 30
        search(thr, budget)
    elif cmd == "best":
        thr = float(args[1])
        best(thr)
    elif cmd == "validate":
        validate()
    else:
        raise SystemExit(f"알 수 없는 명령: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
