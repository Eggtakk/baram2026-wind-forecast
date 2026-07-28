"""
LOYO(leave-one-year-out) 폴드 평균을 목적함수로 쓰는 하이퍼파라미터 재탐색.

배경: 현재 프로덕션 하이퍼파라미터(experiments/baseline_lgbm/
group{n}_{yearly|optuna}_best_params_{recipe}.json)는 전부 "2024년 단일
holdout" 기준으로 tune_yearly.py / tune_optuna.py가 찾은 값이다(scripts/
tune_optuna.py 참고 — SPLIT_DATE="2024-01-01" 고정). 이번 세션에서 확립한
LOYO 프레임(rated_output_investigation.md 10번 섹션)으로 재검증한 baseline
결과를 보면, 그룹별 연도 간 표준편차가 꽤 크다(group1 std=0.0148, group2
std=0.0243, group3 std=0.0098) — 즉 "2024년 기준 최적 파라미터"가 다른 해
에서는 최적이 아닐 수 있고, 지금 프로덕션 파라미터 자체가 2024년 특성에
과적합됐을 위험이 있다.

지금까지 이번 세션에서 검증한 7개 후보(커틀먼트 재탐색/모델교체/feature
추가 3종/pooled 2종/외부데이터)는 전부 "레시피·모델·데이터에 뭔가를
더하거나 바꾸는" 접근이었다. 이 스크립트는 그 대신 "탐색 방식 자체를 더
견고하게" 만든다 — 레시피/모델군(LightGBM, 프로덕션과 동일)은 그대로 두고,
Optuna 목적함수를 "단일 2024 holdout 점수"에서 "그룹의 전체 LOYO 폴드
평균 점수"로 바꿔서 하이퍼파라미터만 재탐색한다.

실행 (레포 루트에서, 그룹 단위로 나눠서):
  python3 scripts/tune_loyo_optuna.py prepare <group_id>              # 폴드별 feature 캐싱(1회)
  python3 scripts/tune_loyo_optuna.py search <group_id> [budget_sec]  # Optuna 탐색(반복 호출 가능, 누적)
  python3 scripts/tune_loyo_optuna.py best <group_id>                 # 현재까지 최고 파라미터 출력/저장
  python3 scripts/tune_loyo_optuna.py validate <group_id>             # 최고 파라미터를 validate_loyo.run_group으로
                                                                        # 재검증(공식 LOYO) 후 baseline과 비교
캐시: /tmp/loyo_optuna_cache_group{gid}.pkl (레포 폴더 밖 — 파일락 문제 회피)
study DB: /tmp/baram_loyo_optuna/loyo_optuna.db (SQLite, load_if_exists=True로 누적)
최종 결과: experiments/baseline_lgbm/loyo_tuned_params_results.json
          experiments/baseline_lgbm/group{n}_loyo_optuna_best_params_{recipe}.json
"""
import json
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # validate_loyo 임포트용

import lightgbm as lgb
import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.data_cleaning import remove_curtailment
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out
from validate_loyo import (
    RECIPE_CHOICE,
    VALID_YEARS,
    build_physics_features,
    run_group,
)
from validate_loyo_candidates import get_baseline_stats, summarize

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
TUNED_RESULTS_PATH = OUT_DIR / "loyo_tuned_params_results.json"

CACHE_DIR = Path(tempfile.gettempdir())
LOCAL_DB_DIR = Path(tempfile.gettempdir()) / "baram_loyo_optuna"
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = f"sqlite:///{LOCAL_DB_DIR / 'loyo_optuna.db'}"
SEED = 42


def cache_path(group_id: int) -> Path:
    return CACHE_DIR / f"loyo_optuna_cache_group{group_id}.pkl"


def build_baseline_features_local(df):
    """validate_loyo.py에는 build_baseline_features가 없어서(physics만 재구현되어
    있음) src.features의 것을 그대로 씀 — full 레시피(group2/3)용."""
    from src.features import build_baseline_features

    return build_baseline_features(df)


def prepare(group_id: int):
    """그룹의 모든 LOYO 폴드에 대해 feature를 미리 만들어 pickle로 캐싱한다.
    (매 trial마다 feature를 다시 만들면 낭비 — 파라미터만 바뀌므로 1회만 계산)
    """
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)

    folds = []
    for holdout_year in VALID_YEARS[group_id]:
        train_raw, holdout_raw = time_based_split_leave_year_out(
            df, holdout_year=holdout_year, valid_years=VALID_YEARS[group_id]
        )
        cleaned_train, n_removed = remove_curtailment(train_raw, capacity=capacity)

        if recipe == "physics":
            train_feat = build_physics_features(cleaned_train)
            holdout_feat = build_physics_features(holdout_raw)
        else:
            train_feat = build_baseline_features_local(cleaned_train)
            holdout_feat = build_baseline_features_local(holdout_raw)
            curve_models = fit_power_curve_models(train_feat, capacity=capacity)
            train_feat = apply_power_curve_models(train_feat, curve_models)
            holdout_feat = apply_power_curve_models(holdout_feat, curve_models)

        from src.features import get_feature_cols

        feature_cols = get_feature_cols(train_feat)
        folds.append({
            "holdout_year": holdout_year,
            "X_train": train_feat[feature_cols],
            "y_train": train_feat["y"],
            "X_holdout": holdout_feat[feature_cols],
            "y_holdout": holdout_feat["y"].to_numpy(),
            "feature_cols": feature_cols,
            "n_removed": n_removed,
        })
        print(f"  fold holdout={holdout_year}: n_train={len(train_feat)} "
              f"n_holdout={len(holdout_feat)} n_features={len(feature_cols)} "
              f"curtailment_removed={n_removed}")

    with open(cache_path(group_id), "wb") as f:
        pickle.dump({"group_id": group_id, "recipe": recipe, "folds": folds}, f)
    print(f"캐시 저장: {cache_path(group_id)}")


def load_cache(group_id: int):
    path = cache_path(group_id)
    if not path.exists():
        raise SystemExit(f"캐시가 없습니다. 먼저 `prepare {group_id}` 실행하세요: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def suggest_params(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def make_objective(group_id: int, cache: dict):
    folds = cache["folds"]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        bagging_freq = 1 if params["bagging_fraction"] < 1.0 else 0
        scores = []
        for fold in folds:
            model = lgb.LGBMRegressor(**params, random_state=SEED, verbosity=-1, bagging_freq=bagging_freq)
            model.fit(fold["X_train"], fold["y_train"])
            pred = model.predict(fold["X_holdout"]).clip(min=0)
            r = validate_single_group(fold["y_holdout"], pred, group_id=group_id)
            scores.append(0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"])
        scores = np.array(scores)
        trial.set_user_attr("fold_scores", scores.tolist())
        trial.set_user_attr("fold_std", float(scores.std(ddof=1)) if len(scores) > 1 else 0.0)
        # LOYO 평균을 최대화 — 폴드 간 표준편차가 큰(운 좋은 한 해에만 잘 맞는)
        # 파라미터보다, 여러 해에 걸쳐 고르게 잘 맞는 파라미터를 우대하기 위해
        # std에 작은 페널티를 준다(단일 폴드 과적합 방지).
        return float(scores.mean()) - 0.1 * float(scores.std(ddof=1) if len(scores) > 1 else 0.0)

    return objective


def search(group_id: int, budget_seconds: int = 25):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cache = load_cache(group_id)
    study_name = f"group{group_id}_loyo_{cache['recipe']}"
    study = optuna.create_study(
        study_name=study_name, storage=STORAGE, direction="maximize",
        sampler=TPESampler(seed=SEED), load_if_exists=True,
    )
    objective = make_objective(group_id, cache)
    study.optimize(objective, timeout=budget_seconds, show_progress_bar=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"[{study_name}] 누적 완료 시도: {len(completed)}")
    if completed:
        best = study.best_trial
        print(f"  BEST objective={best.value:.4f} fold_scores={best.user_attrs['fold_scores']} "
              f"fold_std={best.user_attrs['fold_std']:.4f}")
        print(f"  params={best.params}")


def best(group_id: int):
    cache = load_cache(group_id)
    study_name = f"group{group_id}_loyo_{cache['recipe']}"
    study = optuna.load_study(study_name=study_name, storage=STORAGE)
    b = study.best_trial
    print(f"[{study_name}] n_trials={len(study.trials)}  BEST objective={b.value:.4f}")
    print(f"  fold_scores={b.user_attrs['fold_scores']}  fold_std={b.user_attrs['fold_std']:.4f}")
    print(f"  params={b.params}")
    out_path = OUT_DIR / f"group{group_id}_loyo_optuna_best_params_{cache['recipe']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(b.params, f, indent=2)
    print(f"저장: {out_path}")
    return b.params


def validate(group_id: int):
    """탐색한 최고 파라미터를 validate_loyo.run_group으로 다시 돌려(공식 LOYO
    파이프라인 재사용) baseline과 비교한다."""
    cache = load_cache(group_id)
    params_path = OUT_DIR / f"group{group_id}_loyo_optuna_best_params_{cache['recipe']}.json"
    if not params_path.exists():
        raise SystemExit(f"먼저 `best {group_id}` 실행하세요: {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    fold_results = run_group(group_id, model_params=params)

    existing = {}
    if TUNED_RESULTS_PATH.exists():
        with open(TUNED_RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    key = "loyo_tuned_params_candidate_folds"
    kept = [r for r in existing.get(key, []) if r["group_id"] != group_id]
    existing[key] = kept + fold_results

    baseline_stats = get_baseline_stats()
    summary = summarize("후보 6 (LOYO 폴드 평균 기준 하이퍼파라미터 재탐색)", baseline_stats, existing[key])
    existing["loyo_tuned_params_candidate_summary"] = summary

    with open(TUNED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장(누적): {TUNED_RESULTS_PATH}")


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    cmd = args[0]
    if cmd == "prepare":
        prepare(int(args[1]))
    elif cmd == "search":
        budget = int(args[2]) if len(args) > 2 else 25
        search(int(args[1]), budget)
    elif cmd == "best":
        best(int(args[1]))
    elif cmd == "validate":
        validate(int(args[1]))
    else:
        raise SystemExit(f"알 수 없는 명령: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
