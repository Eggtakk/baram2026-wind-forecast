"""
정격출력(고출력) 구간 전용 2단계 잔차(residual) 모델 — LOYO 기준 검증.

배경: 1~4번 섹션(rated_output_investigation.md)에서 확인된 핵심 문제 —
90-100% 구간에서 세 그룹 모두 예측이 실제보다 체계적으로 낮다(group1
-18%p, group2 -13%p, group3 -30%p, capacity 대비). 4번 섹션에서 원인도
확인됨: 고풍속 구간일수록 LDAPS/GFS 예보 풍속 자체의 편향과 분산이
커진다(과소예측). 2/3번 섹션에서 시드 앙상블/isotonic 사후보정을 시도했지만
전부 기각됐다 — 특히 3번(사후보정)은 "예측값 -> 실제값" 등단조회귀를
좁은 20% holdout으로 학습해 다른 시간대로 일반화가 안 됐다는 게 원인으로
추정됐었다.

이번 시도는 3번과 두 가지가 다르다:
1. 사후에 "예측값 하나"만 보고 보정하는 게 아니라, **원본 feature 전체**를
   입력받는 별도 LightGBM(stage-2)을 고출력 구간에만 학습시켜 "이 구간에서
   1단계 모델이 놓친 것"을 새로 배우게 한다 — 단조 함수 하나보다 표현력이
   훨씬 크다.
2. LOYO(연도별 다중 폴드)로 검증해 특정 연도에만 과적합된 결과인지
   바로 확인할 수 있다 — 3번 시도 당시엔 이 프레임 자체가 없었다.

방법:
  1단계(stage-1): 프로덕션과 동일한 레시피/파라미터로 LightGBM을 각
    LOYO train에 학습.
  Out-of-fold(OOF) 잔차 생성: train을 시간순으로 반씩 나눠(2-fold),
    서로 교차 예측해 train 전체에 대한 "미리 보지 않은" stage-1 예측을
    얻는다(그냥 train 전체로 학습한 모델로 train을 예측하면 거의 완벽하게
    맞아버려 잔차가 다 0에 가까워지므로 학습 신호가 안 나옴 — 반드시
    out-of-fold로 만들어야 함).
  잔차 타깃 정의: 고출력 구간(실제 y/capacity >= REGIME_THRESHOLD)에서만
    residual = y - oof_pred 를 계산해 그 구간 행만으로 stage-2 모델을
    학습(입력 feature는 stage-1과 동일 + stage-1 OOF 예측값 자체도 추가
    feature로 줌 — "지금 얼마로 예측했는지"가 보정 크기의 단서가 될 수 있음).
  holdout 적용: stage-1 모델(전체 train으로 학습)의 holdout 예측이
    REGIME_THRESHOLD 이상인 행에 한해서만 stage-2로 예측한 잔차를 더한다
    (실제 y는 holdout에서 모르므로, "1단계 모델 자신의 예측"이 구간
    판정 기준 — 실전 추론과 동일한 정보만 사용).

실행 (레포 루트에서):
  python3 scripts/validate_loyo_residual_stage.py <group_id> [holdout_year]
  (holdout_year 생략 시 해당 그룹의 모든 LOYO 폴드 실행)
결과 누적: experiments/baseline_lgbm/loyo_residual_stage_results.json
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import lightgbm as lgb

from src.data_cleaning import remove_curtailment
from src.features import build_baseline_features, get_feature_cols
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out
from validate_loyo import OUT_DIR, RECIPE_CHOICE, VALID_YEARS, build_physics_features, load_params
from validate_loyo_candidates import get_baseline_stats, summarize

RESULTS_PATH = OUT_DIR / "loyo_residual_stage_results.json"
REGIME_THRESHOLD = 0.80  # capacity 대비 이 비율 이상인 행만 stage-2 대상(1~4번 섹션의 90-100% 분석보다 약간 넓게 잡아 경계 근처도 포함)
MIN_REGIME_SAMPLES = 50  # 이보다 적으면 stage-2를 학습하지 않고 stage-1 그대로 사용
STAGE2_PARAMS = dict(
    n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=10,
    feature_fraction=0.8, bagging_fraction=0.8,
)


def _fit_lgbm(X, y, params, seed=42):
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


def get_oof_stage1_preds(train_feat, feature_cols, params, capacity):
    """train을 시간순으로 반씩 나눠 교차 예측 -> train 전체에 대한 out-of-fold
    stage-1 예측을 만든다(누수 없이). train_feat는 이미 시간순 정렬돼 있다고
    가정(remove_curtailment가 순서를 보존하는 필터링이라 time_based_split_leave_year_out
    의 정렬을 그대로 물려받음)."""
    n = len(train_feat)
    mid = n // 2
    oof = np.zeros(n)

    idx_a = np.arange(0, mid)
    idx_b = np.arange(mid, n)

    model_b = _fit_lgbm(train_feat.iloc[idx_b][feature_cols], train_feat.iloc[idx_b]["y"], params)
    oof[idx_a] = model_b.predict(train_feat.iloc[idx_a][feature_cols]).clip(min=0)

    model_a = _fit_lgbm(train_feat.iloc[idx_a][feature_cols], train_feat.iloc[idx_a]["y"], params)
    oof[idx_b] = model_a.predict(train_feat.iloc[idx_b][feature_cols]).clip(min=0)

    return oof


def run_fold(group_id: int, df, holdout_year: int) -> dict:
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    train_raw, holdout_raw = time_based_split_leave_year_out(
        df, holdout_year=holdout_year, valid_years=VALID_YEARS[group_id]
    )
    cleaned_train, n_removed = remove_curtailment(train_raw, capacity=capacity)
    train_feat, holdout_feat = _build_features(cleaned_train, holdout_raw, capacity, recipe)
    feature_cols = get_feature_cols(train_feat)
    params = load_params(group_id, recipe)

    # --- stage 1: 프로덕션과 동일 ---
    stage1_model = _fit_lgbm(train_feat[feature_cols], train_feat["y"], params)
    stage1_holdout_pred = stage1_model.predict(holdout_feat[feature_cols]).clip(min=0)
    baseline_result = validate_single_group(holdout_feat["y"].to_numpy(), stage1_holdout_pred, group_id=group_id)
    baseline_score = 0.5 * baseline_result["one_minus_nmae"] + 0.5 * baseline_result["ficr"]

    # --- OOF 잔차로 stage-2 학습 데이터 준비 ---
    oof_pred = get_oof_stage1_preds(train_feat, feature_cols, params, capacity)
    train_frac = train_feat["y"].to_numpy() / capacity
    regime_mask_train = train_frac >= REGIME_THRESHOLD
    n_regime_train = int(regime_mask_train.sum())

    if n_regime_train < MIN_REGIME_SAMPLES:
        final_pred = stage1_holdout_pred
        n_regime_holdout = 0
        stage2_used = False
    else:
        residual_train = train_feat["y"].to_numpy() - oof_pred
        X_stage2 = train_feat.loc[regime_mask_train, feature_cols].copy()
        X_stage2["stage1_pred"] = oof_pred[regime_mask_train]
        y_stage2 = residual_train[regime_mask_train]
        stage2_model = _fit_lgbm(X_stage2, y_stage2, STAGE2_PARAMS)

        holdout_frac = stage1_holdout_pred / capacity
        regime_mask_holdout = holdout_frac >= REGIME_THRESHOLD
        n_regime_holdout = int(regime_mask_holdout.sum())

        final_pred = stage1_holdout_pred.copy()
        if n_regime_holdout > 0:
            X_holdout_stage2 = holdout_feat.loc[regime_mask_holdout, feature_cols].copy()
            X_holdout_stage2["stage1_pred"] = stage1_holdout_pred[regime_mask_holdout]
            correction = stage2_model.predict(X_holdout_stage2)
            final_pred[regime_mask_holdout] = stage1_holdout_pred[regime_mask_holdout] + correction
        final_pred = np.clip(final_pred, 0, capacity)
        stage2_used = True

    result = validate_single_group(holdout_feat["y"].to_numpy(), final_pred, group_id=group_id)
    score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]

    return {
        "group_id": group_id,
        "holdout_year": holdout_year,
        "recipe": recipe,
        "score": score,
        "baseline_score": baseline_score,
        "stage2_used": stage2_used,
        "n_regime_train": n_regime_train,
        "n_regime_holdout": n_regime_holdout if stage2_used else 0,
        "n_train": len(train_feat),
        "n_holdout": len(holdout_feat),
    }


def run_group(group_id: int, holdout_years=None) -> list[dict]:
    t0 = time.time()
    df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)
    years = holdout_years if holdout_years else VALID_YEARS[group_id]
    print(f"\n=== group{group_id} (recipe={RECIPE_CHOICE[group_id]}, folds={years}, "
          f"REGIME_THRESHOLD={REGIME_THRESHOLD}) ===")

    fold_results = []
    for holdout_year in years:
        r = run_fold(group_id, df, holdout_year)
        fold_results.append(r)
        print(f"  holdout={holdout_year}: score={r['score']:.4f} (baseline={r['baseline_score']:.4f}, "
              f"delta={r['score']-r['baseline_score']:+.4f}) "
              f"stage2_used={r['stage2_used']} n_regime_train={r['n_regime_train']} "
              f"n_regime_holdout={r['n_regime_holdout']}")

    scores = np.array([r["score"] for r in fold_results])
    print(f"  -> group{group_id} mean={scores.mean():.4f} "
          f"(n_folds={len(scores)}) [{time.time()-t0:.1f}s]")
    return fold_results


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    group_id = int(args[0])
    holdout_years = [int(a) for a in args[1:]] if len(args) > 1 else None

    import json
    existing = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)

    new_results = run_group(group_id, holdout_years)
    covered_years = {r["holdout_year"] for r in new_results}
    existing = [r for r in existing if not (r["group_id"] == group_id and r["holdout_year"] in covered_years)]
    existing.extend(new_results)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장(누적): {RESULTS_PATH}")

    baseline_stats = get_baseline_stats()
    summary = summarize("후보 8 (정격출력 구간 2단계 잔차 모델)", baseline_stats, existing)
    summary_path = OUT_DIR / "loyo_residual_stage_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
