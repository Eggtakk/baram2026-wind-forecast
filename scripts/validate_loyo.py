"""
Leave-one-year-out(LOYO) 다중 폴드 검증 — 단일 연도 holdout의 신뢰도 문제 대응.

배경 (experiments/baseline_lgbm/rated_output_investigation.md 참고):
기존 scripts/validate_yearly_holdout.py는 train=2023년, holdout=2024년
"단 한 번"의 연 단위 분할만 쓴다. 이 방식은 (기존 20% 시간순 holdout의
계절 편중 문제는 고쳤지만) 여전히 "2024년이라는 특정 한 해"에 결과가
좌우되는 한계가 있다. 실제로 2026-07-26 하루에만 holdout에서는 그룹당
+0.004~0.009 개선을 예측한 시도 세 건(커틀먼트 임계값 재탐색, XGBoost
기본형 3그룹 블렌딩, XGBoost 튜닝 후 group2 교체)이 전부 실제 리더보드
제출에서는 하락으로 뒤집혔다. 2024년이 착빙/커틀먼트 이벤트가 유독 많았던
해라 2025년 test와 계절 구성은 같아도 이상치 패턴까지는 대표하지 못했을
가능성이 크다 — 게다가 DACON 리더보드 자체도 전체 평가 데이터의 40%만
반영하는 Public Score라(대회 규칙 페이지 확인, 1차 평가는 Private Score
60% 기준) 노이즈가 이중으로 낀 상태였다.

수정: group1/2는 라벨이 2022/2023/2024 세 해 전부 있으므로 각 연도를
돌아가며 holdout으로 쓰는 3-fold leave-one-year-out을 수행한다(group3는
라벨이 2023/2024 두 해뿐이라 2-fold). 폴드별 점수의 평균뿐 아니라
표준편차까지 같이 봐서, "이 정도 개선이면 실제로 신뢰할 수 있다"는
기준(노이즈 밴드)을 세운다 — 표준편차보다 작은 delta는 애초에 제출
후보에서 제외하는 식으로 활용할 것.

현재 프로덕션 설정(scripts/train_final.py와 동일한 레시피/파라미터/커틀먼트
기준)을 그대로 이 방식으로 재검증하는 것이 1차 목적이며, 이후 새 아이디어나
과거에 기각된 아이디어(커틀먼트 임계값 재탐색, 모델 계열 블렌딩 등)를
재검증할 때도 이 스크립트의 run_group()을 재사용할 수 있다.

실행: (레포 루트에서)
  python3 scripts/validate_loyo.py            # 그룹 1/2/3 전부
  python3 scripts/validate_loyo.py 1           # 그룹1만
결과는 experiments/baseline_lgbm/loyo_validation_results.json 에도 저장된다.
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
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
RESULTS_PATH = OUT_DIR / "loyo_validation_results.json"

# scripts/train_final.py(현재 프로덕션 제출, 리더보드 0.61034)와 동일한 레시피/파라미터 소스.
RECIPE_CHOICE = {1: "physics", 2: "full", 3: "full"}
PARAMS_SOURCE = {1: "yearly", 2: "yearly", 3: "optuna"}

# group1/2: 라벨 2022~2024 전부 있음 -> 3-fold. group3: 2023~2024만 있음 -> 2-fold.
VALID_YEARS = {1: [2022, 2023, 2024], 2: [2022, 2023, 2024], 3: [2023, 2024]}


def build_physics_features(df):
    """scripts/train_final.py의 build_physics_features와 동일 (physics-only 레시피)."""
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


def run_fold(
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
) -> dict:
    """주어진 그룹 전체 df에서 holdout_year를 떼어내 1개 폴드를 학습/평가한다.

    커틀먼트 제거는 항상 train 쪽에만 적용(holdout/실제 test는 그대로 둔다는
    프로덕션 원칙 유지, src/data_cleaning.py 참고).

    curtailment_ratio / curtailment_wind_thresh / model_type / model_params를
    지정하면 프로덕션 기본값(0.30/8.0, LightGBM+group{n}_{source}_best_params_*)
    대신 다른 후보 설정으로 검증할 수 있다 — 과거에 단일 2024-holdout으로만
    검증됐던 후보(커틀먼트 임계값 재탐색, XGBoost/CatBoost 모델 교체 등)를
    LOYO로 재검증할 때 씀(scripts/validate_loyo_candidates.py 참고).

    extra_feature_fn: `df -> df` 형태의 함수를 넘기면, 레시피별 기본 feature를
    다 만든 뒤(train/holdout 각각) 추가로 적용한다 — 아직 프로덕션 레시피에
    편입되지 않은 새 feature 후보(예: src.features.add_forecast_disagreement_features)를
    LOYO로 검증할 때 씀.
    """
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
    result = validate_single_group(holdout_feat["y"].to_numpy(), pred, group_id=group_id)
    score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]

    return {
        "group_id": group_id,
        "holdout_year": holdout_year,
        "recipe": recipe,
        "model_type": model_type,
        "params_source": PARAMS_SOURCE[group_id] if model_params is None else "override",
        "score": score,
        "nmae": result["nmae"],
        "one_minus_nmae": result["one_minus_nmae"],
        "ficr": result["ficr"],
        "n_train": len(train_feat),
        "n_holdout": len(holdout_feat),
        "n_eval": result["n_eval"],
        "n_curtailment_removed": n_removed,
    }


def run_group(group_id: int, **fold_overrides) -> list[dict]:
    t0 = time.time()
    df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)
    print(f"\n=== group{group_id} (recipe={RECIPE_CHOICE[group_id]}, "
          f"params_source={PARAMS_SOURCE[group_id]}, folds={VALID_YEARS[group_id]}, "
          f"overrides={fold_overrides or '(none, 프로덕션 기본값)'}) ===")

    fold_results = []
    for holdout_year in VALID_YEARS[group_id]:
        r = run_fold(group_id, df, holdout_year, **fold_overrides)
        fold_results.append(r)
        print(f"  holdout={holdout_year}: score={r['score']:.4f} "
              f"(1-NMAE={r['one_minus_nmae']:.4f}, FICR={r['ficr']:.4f}) "
              f"n_train={r['n_train']} n_holdout={r['n_holdout']} "
              f"curtailment_removed={r['n_curtailment_removed']}")

    scores = np.array([r["score"] for r in fold_results])
    mean, std = scores.mean(), (scores.std(ddof=1) if len(scores) > 1 else 0.0)
    print(f"  -> group{group_id} mean={mean:.4f} std={std:.4f} "
          f"(min={scores.min():.4f}, max={scores.max():.4f}, n_folds={len(scores)}) "
          f"[{time.time()-t0:.1f}s]")
    return fold_results


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]

    all_results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        # 이번 실행에 포함된 그룹의 기존 결과는 덮어쓴다(재실행 시 중복 방지).
        all_results = [r for r in all_results if r["group_id"] not in groups]

    for gid in groups:
        all_results.extend(run_group(gid))

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {RESULTS_PATH}")

    # 요약: 그룹별 mean/std + 그룹 평균으로 대략적인 total_score 추정치
    print("\n=== 요약 (전체 그룹 결과 기준) ===")
    group_means = {}
    for gid in sorted({r["group_id"] for r in all_results}):
        scores = np.array([r["score"] for r in all_results if r["group_id"] == gid])
        std = scores.std(ddof=1) if len(scores) > 1 else 0.0
        group_means[gid] = float(scores.mean())
        print(f"  group{gid}: mean={scores.mean():.4f} std={std:.4f} (n_folds={len(scores)})")
    if len(group_means) == 3:
        overall = np.mean(list(group_means.values()))
        print(f"  전체 평균(그룹별 단순평균): {overall:.4f}")


if __name__ == "__main__":
    main()
