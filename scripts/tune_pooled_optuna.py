"""
Pooled(3그룹 통합) LightGBM 모델을 "full" 레시피(saturation+그룹별 파워커브)로
짓고 Optuna로 하이퍼파라미터를 탐색한다.

배경: experiments/baseline_lgbm/rated_output_investigation.md 14번 섹션 —
1차 pooled 시도(physics-only, 튜닝 없는 파라미터)는 LOYO 기준으로 baseline
(그룹별 개별 모델)보다 나은 delta를 못 만들었다. 두 가지 한계(레시피
단순화, 튜닝 안 된 파라미터) 때문에 결론을 내리기엔 일렀다는 지적에 따라,
이번엔 (a) group2/3 프로덕션과 동일한 "full" 레시피(saturation+그룹별
파워커브)를 쓰고, (b) Optuna로 pooled 전용 하이퍼파라미터를 제대로 탐색해
더 공정하고 리소스를 더 들인 재시도를 한다.

탐색 전략(sandbox 45초 제약 대응, 기존 tune_optuna.py/tune_family_optuna.py와
동일한 패턴):
  1) `prepare`: holdout=2024 폴드 기준으로 pooled_train/holdout(full 레시피,
     커틀먼트 제거, 파워커브 fit 포함)을 한 번만 만들어 /tmp에 pickle로 캐시.
     (2024 폴드를 탐색용 대리 지표로 쓰는 이유: 매 trial마다 3개 폴드를 다
     돌리면 trial당 시간이 너무 커진다 — team이 그룹별 파라미터를 탐색할
     때도 항상 단일 holdout으로 빠르게 찾고 최종 후보만 전체 LOYO로
     재검증하는 방식을 써왔다.)
  2) `search [seconds]`: 캐시를 로드해(빠름) Optuna trial 반복 — trial마다
     데이터를 다시 읽지 않고 이미 준비된 pooled_train/holdout에서 LightGBM
     학습만 반복하므로 trial당 수 초. SQLite(/tmp — 마운트된 레포 폴더에
     두면 파일 락 문제 있음, 기존 tune_optuna.py 참고)에 저장해 여러 번의
     sandbox 호출에 걸쳐 trial을 누적한다. 목적함수: holdout=2024에서
     3그룹 평균 score(그룹당 가중치 동일 — 대회 total_score 산식과는 다르지만
     방향성 참고용, 기존 문서들과 동일한 관례).
  3) `best`: 지금까지의 study에서 최고 trial의 파라미터를 출력하고
     experiments/baseline_lgbm/pooled_full_optuna_best_params.json 으로 저장.
  4) `validate`: best 파라미터 + full 레시피로 2022/2023/2024 전체 LOYO를
     돌려(scripts/validate_loyo_pooled.py::run_pooled_fold 재사용)
     baseline(그룹별 개별 모델)과 최종 비교. 결과는
     experiments/baseline_lgbm/loyo_pooled_full_tuned_results.json.

실행 예:
  python3 scripts/tune_pooled_optuna.py prepare
  python3 scripts/tune_pooled_optuna.py search 35     # 반복 호출해서 trial 누적
  python3 scripts/tune_pooled_optuna.py best
  python3 scripts/tune_pooled_optuna.py validate
"""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # validate_loyo_pooled 임포트용

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.data_cleaning import remove_curtailment
from src.metrics import CAPACITY_KWH, validate_single_group
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out
from validate_loyo_pooled import (
    ALL_HOLDOUT_YEARS,
    BASELINE_PATH,
    GROUP_IDS,
    VALID_YEARS,
    _build_group_features,
    get_pooled_feature_cols,
    run_pooled_fold,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
BEST_PARAMS_PATH = OUT_DIR / "pooled_full_optuna_best_params.json"
TUNED_RESULTS_PATH = OUT_DIR / "loyo_pooled_full_tuned_results.json"

CACHE_PATH = Path("/tmp/pooled_optuna_cache_full_2024.pkl")
STUDY_DB = "sqlite:////tmp/pooled_optuna.db"
STUDY_NAME = "pooled_full_2024"
SEARCH_HOLDOUT_YEAR = 2024
RECIPE = "full"


def prepare():
    """holdout=2024, full 레시피로 pooled_train/holdout을 만들어 pickle 캐시."""
    t0 = time.time()
    train_frames = []
    holdout_by_group = {}

    for gid in GROUP_IDS:
        capacity = CAPACITY_KWH[f"kpx_group_{gid}"]
        df = build_group_dataset(gid, split="train").dropna(subset=["y"]).reset_index(drop=True)
        train_raw, holdout_raw = time_based_split_leave_year_out(
            df, holdout_year=SEARCH_HOLDOUT_YEAR, valid_years=VALID_YEARS[gid]
        )
        cleaned_train, n_removed = remove_curtailment(train_raw, capacity=capacity)
        train_feat, holdout_feat = _build_group_features(cleaned_train, holdout_raw, capacity, RECIPE)
        train_feat["group_id"] = gid
        train_feat["y_frac"] = train_feat["y"] / capacity
        train_frames.append(train_feat)
        holdout_feat["group_id"] = gid
        holdout_by_group[gid] = (holdout_feat, capacity)

    pooled_train = pd.concat(train_frames, ignore_index=True)
    pooled_train["group_id"] = pd.Categorical(pooled_train["group_id"], categories=GROUP_IDS)
    feature_cols = get_pooled_feature_cols(pooled_train)

    for gid, (holdout_feat, capacity) in holdout_by_group.items():
        holdout_feat["group_id"] = pd.Categorical(holdout_feat["group_id"], categories=GROUP_IDS)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(
            {
                "pooled_train": pooled_train,
                "holdout_by_group": holdout_by_group,
                "feature_cols": feature_cols,
            },
            f,
        )
    print(f"캐시 저장: {CACHE_PATH} (n_train={len(pooled_train)}, {len(feature_cols)} features) "
          f"[{time.time()-t0:.1f}s]")


def load_cache():
    if not CACHE_PATH.exists():
        raise SystemExit(f"캐시가 없습니다. 먼저 `python3 {Path(__file__).name} prepare`를 실행하세요.")
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)


def suggest_params(trial: optuna.Trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 200),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def make_objective(cache: dict):
    pooled_train = cache["pooled_train"]
    holdout_by_group = cache["holdout_by_group"]
    feature_cols = cache["feature_cols"]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        bagging_freq = 1 if params["subsample"] < 1.0 else 0
        model = lgb.LGBMRegressor(**params, bagging_freq=bagging_freq, random_state=42, verbosity=-1)
        model.fit(pooled_train[feature_cols], pooled_train["y_frac"], categorical_feature=["group_id"])

        scores = []
        for gid, (holdout_feat, capacity) in holdout_by_group.items():
            pred_frac = model.predict(holdout_feat[feature_cols]).clip(min=0)
            pred_y = pred_frac * capacity
            result = validate_single_group(holdout_feat["y"].to_numpy(), pred_y, group_id=gid)
            scores.append(0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"])
        mean_score = float(np.mean(scores))
        trial.set_user_attr("group_scores", scores)
        return mean_score

    return objective


def search(budget_seconds: float):
    cache = load_cache()
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB, direction="maximize", load_if_exists=True
    )
    n_before = len(study.trials)
    objective = make_objective(cache)
    study.optimize(objective, timeout=budget_seconds, show_progress_bar=False)
    n_after = len(study.trials)
    print(f"trial {n_before} -> {n_after} (이번 호출에서 {n_after - n_before}개 추가)")
    if study.best_trial is not None:
        print(f"현재까지 best: score={study.best_value:.4f} params={study.best_params}")


def best():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB)
    print(f"총 trial 수: {len(study.trials)}")
    print(f"best score (holdout=2024, 3그룹 평균): {study.best_value:.4f}")
    print(f"best group_scores: {study.best_trial.user_attrs.get('group_scores')}")
    print(f"best params: {json.dumps(study.best_params, indent=2, ensure_ascii=False)}")
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)
    print(f"저장: {BEST_PARAMS_PATH}")


def validate(years=None):
    """best 파라미터 + full 레시피로 LOYO 재검증 (연도별로 나눠 호출 가능 —
    sandbox 45초 제약 대응, 결과는 매 호출마다 누적/병합 저장됨)."""
    if not BEST_PARAMS_PATH.exists():
        raise SystemExit("best 파라미터가 없습니다. 먼저 `best` 서브커맨드를 실행하세요.")
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)
    if params.get("subsample", 1.0) < 1.0:
        params["bagging_freq"] = 1

    years = years or ALL_HOLDOUT_YEARS

    existing = []
    if TUNED_RESULTS_PATH.exists():
        with open(TUNED_RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing = [r for r in existing if r["holdout_year"] not in years]

    all_results = existing
    for year in years:
        print(f"\n=== pooled(full, tuned) fold holdout_year={year} ===")
        all_results.extend(run_pooled_fold(year, recipe=RECIPE, params=params))

    with open(TUNED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장(누적): {TUNED_RESULTS_PATH}")

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    print("\n=== pooled(full,tuned) vs baseline(그룹별 개별 모델) ===")
    for gid in GROUP_IDS:
        b = np.array([r["score"] for r in baseline if r["group_id"] == gid])
        p = np.array([r["score"] for r in all_results if r["group_id"] == gid])
        if len(b) == 0 or len(p) == 0:
            continue
        bmean, bstd = float(b.mean()), float(b.std(ddof=1))
        pmean = float(p.mean())
        pstd = float(p.std(ddof=1)) if len(p) > 1 else 0.0
        delta = pmean - bmean
        verdict = "신뢰 가능(노이즈 초과)" if abs(delta) > bstd else "노이즈 수준(불확실)"
        print(f"  group{gid}: baseline={bmean:.4f}(std={bstd:.4f}) -> "
              f"pooled_full_tuned={pmean:.4f}(std={pstd:.4f}, n={len(p)})  "
              f"delta={delta:+.4f}  [{verdict}]")


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit("사용법: prepare | search [seconds] | best | validate")
    cmd = args[0]
    if cmd == "prepare":
        prepare()
    elif cmd == "search":
        budget = float(args[1]) if len(args) > 1 else 35.0
        search(budget)
    elif cmd == "best":
        best()
    elif cmd == "validate":
        years = [int(a) for a in args[1:]] or None
        validate(years)
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
