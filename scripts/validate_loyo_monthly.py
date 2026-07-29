"""
37번 섹션 후속 — LOYO fold 수(group1/2=3, group3=2)가 너무 적어 std 추정
자체가 불안정하다는 문제에 대응. 학습(train/holdout year split, 모델
자체)은 기존 `scripts/validate_loyo.py`와 완전히 동일하게 유지하되(즉
"완전히 못 본 연도"라는 진짜 외삽 성질은 그대로 보존), 평가만 holdout
연도 "전체 1개 점수" 대신 **월별로 쪼개서** 계산한다.

이렇게 하면 group1/2는 3년×12개월=36개, group3는 2년×12개월=24개의 (거의)
독립적인 평가 지점을 얻어 mean/std/SEM(표준오차)을 훨씬 안정적으로 추정할
수 있다 — 재학습 비용은 기존과 동일(연도당 모델 1개), 평가만 더 세밀하게
나눈 것이므로 "저렴한" 확장이다.

주의(한계): 월별 점수들은 완전히 독립(i.i.d)은 아니다(같은 모델의 예측,
인접 월은 날씨 자기상관 있음) — SEM을 과신하면 안 되고 std 자체를
참고 삼아 기존 연 단위 std와 비교하는 용도로 쓴다.

실행: (레포 루트에서)
  python3 scripts/validate_loyo_monthly.py            # 그룹 1/2/3 전부
  python3 scripts/validate_loyo_monthly.py 1           # 그룹1만
결과는 experiments/baseline_lgbm/loyo_monthly_results.json 에 저장.
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
from src.metrics import CAPACITY_KWH, group_nmae_ficr
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
RESULTS_PATH = OUT_DIR / "loyo_monthly_results.json"

RECIPE_CHOICE = {1: "physics", 2: "full", 3: "full"}
PARAMS_SOURCE = {1: "yearly", 2: "yearly", 3: "optuna"}
VALID_YEARS = {1: [2022, 2023, 2024], 2: [2022, 2023, 2024], 3: [2023, 2024]}


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


def run_fold_monthly(
    group_id: int,
    df,
    holdout_year: int,
    *,
    curtailment_ratio: float | None = None,
    curtailment_wind_thresh: float | None = None,
    model_type: str = "lightgbm",
    model_params: dict | None = None,
    extra_feature_fn=None,
    seed: int = 42,
    min_actual_ratio: float = 0.10,
) -> list[dict]:
    """validate_loyo.run_fold와 동일한 train/holdout 분할·학습을 쓰되,
    holdout 연도 전체가 아니라 월별로 점수를 나눠서 반환한다."""
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    train_raw, holdout_raw = time_based_split_leave_year_out(
        df, holdout_year=holdout_year, valid_years=VALID_YEARS[group_id]
    )

    curtail_kwargs = {}
    if curtailment_ratio is not None:
        curtail_kwargs["residual_threshold_ratio"] = curtailment_ratio
    if curtailment_wind_thresh is not None:
        curtail_kwargs["high_wind_threshold"] = curtailment_wind_thresh
    cleaned_train, n_removed = remove_curtailment(train_raw, capacity=capacity, **curtail_kwargs)

    if recipe == "physics":
        train_feat = build_physics_features(cleaned_train)
        holdout_feat = build_physics_features(holdout_raw)
    else:
        train_feat = build_baseline_features(cleaned_train)
        holdout_feat = build_baseline_features(holdout_raw)
        curve_models = fit_power_curve_models(train_feat, capacity=capacity)
        train_feat = apply_power_curve_models(train_feat, curve_models)
        holdout_feat = apply_power_curve_models(holdout_feat, curve_models)

    if extra_feature_fn is not None:
        train_feat = extra_feature_fn(train_feat)
        holdout_feat = extra_feature_fn(holdout_feat)

    feature_cols = get_feature_cols(train_feat)
    params = model_params if model_params is not None else load_params(group_id, recipe)

    if model_type == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(**params, random_state=seed, tree_method="hist", verbosity=0)
    elif model_type == "catboost":
        import catboost as cb

        model = cb.CatBoostRegressor(**params, random_seed=seed, verbose=False)
    else:
        bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
        model = lgb.LGBMRegressor(**params, random_state=seed, bagging_freq=bagging_freq, verbosity=-1)

    model.fit(train_feat[feature_cols], train_feat["y"])
    pred = model.predict(holdout_feat[feature_cols]).clip(min=0)
    y = holdout_feat["y"].to_numpy()
    months = holdout_feat["month"].to_numpy()

    month_results = []
    for m in range(1, 13):
        sel = months == m
        if sel.sum() == 0:
            continue
        nmae, ficr, n_eval = group_nmae_ficr(y[sel], pred[sel], capacity, min_actual_ratio=min_actual_ratio)
        if n_eval == 0 or np.isnan(nmae):
            continue
        one_minus_nmae = 1 - nmae
        score = 0.5 * one_minus_nmae + 0.5 * ficr
        month_results.append({
            "group_id": group_id,
            "holdout_year": holdout_year,
            "month": int(m),
            "score": score,
            "one_minus_nmae": one_minus_nmae,
            "ficr": ficr,
            "n_eval": n_eval,
            "n_curtailment_removed": n_removed,
        })
    return month_results


def run_group_monthly(group_id: int, **fold_overrides) -> list[dict]:
    t0 = time.time()
    df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)
    print(f"\n=== group{group_id} monthly-block LOYO (recipe={RECIPE_CHOICE[group_id]}, "
          f"folds={VALID_YEARS[group_id]}, overrides={fold_overrides or '(none)'}) ===")

    all_month_results = []
    for holdout_year in VALID_YEARS[group_id]:
        mr = run_fold_monthly(group_id, df, holdout_year, **fold_overrides)
        all_month_results.extend(mr)
        yr_scores = np.array([r["score"] for r in mr])
        print(f"  holdout={holdout_year}: {len(mr)}개월, "
              f"연도내 월별 mean={yr_scores.mean():.4f} std={yr_scores.std(ddof=1):.4f}")

    scores = np.array([r["score"] for r in all_month_results])
    n = len(scores)
    mean, std = scores.mean(), scores.std(ddof=1)
    sem = std / np.sqrt(n)
    print(f"  -> group{group_id} 월별-블록 전체: n={n} mean={mean:.4f} std={std:.4f} "
          f"SEM={sem:.4f} (min={scores.min():.4f}, max={scores.max():.4f}) [{time.time()-t0:.1f}s]")
    return all_month_results


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]

    all_results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        all_results = [r for r in all_results if r["group_id"] not in groups]

    for gid in groups:
        all_results.extend(run_group_monthly(gid))

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {RESULTS_PATH}")

    print("\n=== 요약 (월별-블록 LOYO, 그룹별) ===")
    for gid in sorted({r["group_id"] for r in all_results}):
        scores = np.array([r["score"] for r in all_results if r["group_id"] == gid])
        n = len(scores)
        std = scores.std(ddof=1)
        sem = std / np.sqrt(n)
        print(f"  group{gid}: n={n} mean={scores.mean():.4f} std={std:.4f} SEM={sem:.4f}")


if __name__ == "__main__":
    main()
