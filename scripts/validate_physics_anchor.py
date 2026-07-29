"""
28번 섹션 끝 제안 — stage-1의 고출력 외삽 능력을 "물리 기반 파워커브를 더
강하게 반영"해 개선할 수 있는지 검증.

배경: 16번 섹션에서 제작사 공식 파워커브(src/manufacturer_power_curve.py)를
그냥 feature 하나로 추가했을 때는 효과가 거의 없었다(delta -0.0009~+0.0015,
전부 노이즈 수준) — 이미 있는 풍속/풍속³/saturation/isotonic 파워커브
feature들로 트리 모델이 비슷한 모양을 스스로 근사할 수 있어서, "정확한
물리 곡선을 추가로 주는 것"의 한계효용이 낮았다는 결론이었다.

이번엔 질적으로 다른 방식으로 반영한다 — feature로 주는 게 아니라 예측
"타깃 자체"를 바꾼다: 원래 목표 y를 직접 맞추는 대신, y - 물리곡선추정치
(=잔차)를 맞추도록 stage-1을 재구성한다. 최종 예측 = 물리곡선추정치 +
model.predict(X). 이렇게 하면:
  - 모델이 굳이 재학습으로 재현하지 않아도 되는 "정격출력 근처 포화 모양"을
    물리 공식이 이미 정확하게 깔아주고, 모델은 그 위의 (상대적으로 작고
    안정적인) 편차만 학습하면 된다.
  - 학습 데이터가 희박한 고풍속 구간에서도, 잔차가 저/중풍속 구간의 잔차와
    비슷한 분포를 가질 가능성이 높아(터빈의 물리적 거동 자체는 연속적이므로)
    "타깃값 자체를 외삽"해야 하는 것보다 일반화가 쉬울 수 있다는 가설.

이 스크립트는 이 가설만 순수하게 격리해서 본다 — 프로덕션과 동일한
레시피/하이퍼파라미터/L2 objective를 그대로 쓰고, 타깃 재구성 여부만
다르게 한다(FICR objective나 잔차보정 stage-2는 아직 얹지 않음 — 이
1차 검증이 통과해야 다음 단계로 의미가 있음).

실행: python3 scripts/validate_physics_anchor.py <group_id>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import lightgbm as lgb

from src.data_cleaning import remove_curtailment
from src.features import build_baseline_features, get_feature_cols
from src.manufacturer_power_curve import estimate_group_power_kwh
from src.metrics import CAPACITY_KWH, analyze_error_bands, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out
from validate_loyo import RECIPE_CHOICE, VALID_YEARS, build_physics_features, load_params
from validate_loyo_candidates import get_baseline_stats


def run_fold(group_id: int, df, holdout_year: int) -> dict:
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    train_raw, holdout_raw = time_based_split_leave_year_out(
        df, holdout_year=holdout_year, valid_years=VALID_YEARS[group_id]
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
    params = load_params(group_id, recipe)

    # --- 물리곡선 추정치(그룹 스케일) ---
    train_phys = estimate_group_power_kwh(
        train_feat["ldaps_hub_speed"], train_feat["ldaps_air_density"], group_id, capacity
    )
    holdout_phys = estimate_group_power_kwh(
        holdout_feat["ldaps_hub_speed"], holdout_feat["ldaps_air_density"], group_id, capacity
    )

    # --- baseline: 원래 타깃(y) 직접 학습 ---
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    base_model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    base_model.fit(train_feat[feature_cols], train_feat["y"])
    base_pred = base_model.predict(holdout_feat[feature_cols]).clip(min=0, max=capacity)

    # --- physics-anchored: y - 물리곡선추정치를 학습, 추론 시 다시 더함 ---
    anchor_target = train_feat["y"].to_numpy() - np.asarray(train_phys)
    anchor_model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    anchor_model.fit(train_feat[feature_cols], anchor_target)
    anchor_pred = (np.asarray(holdout_phys) + anchor_model.predict(holdout_feat[feature_cols])).clip(0, capacity)

    y_holdout = holdout_feat["y"].to_numpy()
    base_result = validate_single_group(y_holdout, base_pred, group_id=group_id)
    anchor_result = validate_single_group(y_holdout, anchor_pred, group_id=group_id)
    base_score = 0.5 * base_result["one_minus_nmae"] + 0.5 * base_result["ficr"]
    anchor_score = 0.5 * anchor_result["one_minus_nmae"] + 0.5 * anchor_result["ficr"]

    return {
        "holdout_year": holdout_year,
        "base_score": base_score, "base_nmae": base_result["nmae"], "base_ficr": base_result["ficr"],
        "anchor_score": anchor_score, "anchor_nmae": anchor_result["nmae"], "anchor_ficr": anchor_result["ficr"],
        "y_holdout": y_holdout, "base_pred": base_pred, "anchor_pred": anchor_pred, "capacity": capacity,
    }


def main(group_id: int):
    baseline_stats = get_baseline_stats()
    l2_base_mean, l2_base_std = baseline_stats[group_id][0], baseline_stats[group_id][1]
    print(f"=== group{group_id} L2 baseline(공식 loyo_validation_results.json) mean={l2_base_mean:.4f} std={l2_base_std:.4f} ===")

    df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)
    fold_results = []
    for year in VALID_YEARS[group_id]:
        r = run_fold(group_id, df, year)
        fold_results.append(r)
        print(f"  holdout={year}: base={r['base_score']:.4f}(NMAE={r['base_nmae']:.4f} FICR={r['base_ficr']:.4f})  "
              f"anchor={r['anchor_score']:.4f}(NMAE={r['anchor_nmae']:.4f} FICR={r['anchor_ficr']:.4f})  "
              f"delta={r['anchor_score']-r['base_score']:+.4f}")

    base_scores = np.array([r["base_score"] for r in fold_results])
    anchor_scores = np.array([r["anchor_score"] for r in fold_results])
    print(f"\n  base(직접 재현): mean={base_scores.mean():.4f} delta_vs_official={base_scores.mean()-l2_base_mean:+.4f}")
    print(f"  physics-anchor:  mean={anchor_scores.mean():.4f} delta_vs_official={anchor_scores.mean()-l2_base_mean:+.4f} "
          f"delta_vs_base={anchor_scores.mean()-base_scores.mean():+.4f}")

    # 고출력 구간(capacity_ratio) 오차 밴드 비교 (전체 폴드 합산)
    all_y = np.concatenate([r["y_holdout"] for r in fold_results])
    all_base = np.concatenate([r["base_pred"] for r in fold_results])
    all_anchor = np.concatenate([r["anchor_pred"] for r in fold_results])
    capacity = fold_results[0]["capacity"]
    band_base = analyze_error_bands(all_y, all_base, capacity)
    band_anchor = analyze_error_bands(all_y, all_anchor, capacity)
    print("\n  고출력(60~100%) 구간 mean_bias 비교 (base -> physics-anchor):")
    for b in ["60-70%", "70-80%", "80-90%", "90-100%"]:
        if b in band_base["by_capacity_band"].index:
            bb = band_base["by_capacity_band"].loc[b, "mean_bias"]
            ba = band_anchor["by_capacity_band"].loc[b, "mean_bias"]
            print(f"    {b}: {bb:+.4f} -> {ba:+.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
