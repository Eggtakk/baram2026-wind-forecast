"""
LOYO(leave-one-year-out) 프레임으로 과거에 단일 2024-holdout에서만 검증되고
실제 제출에서 뒤집혔던 후보들, 그리고 아직 시도하지 않은 새 feature 후보를
검증한다.

배경: experiments/baseline_lgbm/rated_output_investigation.md 8번/9번/10번
섹션. 커틀먼트/모델교체 두 후보 모두 2024년 단일 holdout에서는 그룹당
+0.004~0.009 개선을 예측했지만 실제 리더보드 제출에서는 하락했다
(①0.61034->0.60885, ③0.61034->0.60911). 10번 섹션에서 LOYO로 그룹별 연도 간
표준편차(노이즈 밴드)를 구했다: group1 std=0.0148, group2 std=0.0243, group3
std=0.0098. 이 스크립트는 그 노이즈 밴드를 기준으로 각 후보의 LOYO delta가
"신뢰할 만한 개선"인지(=표준편차보다 큰지) 판단한다.

후보 1 — 커틀먼트 임계값 재탐색 (scripts/tune_curtailment_threshold.py 결과,
experiments/baseline_lgbm/group{n}_curtailment_threshold_best.json):
  group1: ratio=0.20, wind=6.0 (기존 0.30/8.0)
  group2: ratio=0.35, wind=6.0
  group3: ratio=0.20, wind=6.0
  (11번 섹션에서 재검증 완료 — 전부 노이즈 수준, 기각 유지)

후보 2 — XGBoost/CatBoost 튜닝 모델 교체 (scripts/tune_family_optuna.py 결과):
  group1: XGBoost (레시피는 physics 그대로, 모델만 교체)
  group2: XGBoost
  group3: CatBoost (섹션 9에서 CatBoost가 XGBoost보다 우세했던 유일한 그룹)
  (11번 섹션에서 재검증 완료 — 전부 노이즈 수준, 기각 유지)

후보 3 — 예보 소스 간 불일치(disagreement) feature 추가
(src.features.add_forecast_disagreement_features, 신규 미채택 feature):
  group1/2/3 전부 동일 — 기존 레시피(physics/full)에 LDAPS-GFS 허브풍속/10m풍속/
  풍향 불일치 feature 3개를 추가만 한다(커틀먼트/모델은 프로덕션 그대로).
  4번 섹션(예보 정확도 분석)에서 확인된 "고출력 구간일수록 예보 MAE가 커진다"는
  사실에 근거 — 5번 섹션에서 기각된 "예보값 자체 보정"과 달리 트리 모델이
  스스로 만들 수 없는 새 interaction 정보라 다른 결과를 기대해볼 만함.

베이스라인(프로덕션, LightGBM + 커틀먼트 0.30/8.0, 기존 feature만)은
scripts/validate_loyo.py로 이미 계산해 저장해둔
experiments/baseline_lgbm/loyo_validation_results.json을 그대로 재사용한다
(재학습하지 않음).

실행 (레포 루트에서, 그룹/시간 제약 때문에 후보·그룹 단위로 나눠서 실행 가능):
  python3 scripts/validate_loyo_candidates.py curtailment      # 후보1, 전체 그룹
  python3 scripts/validate_loyo_candidates.py model_swap 2     # 후보2, group2만
  python3 scripts/validate_loyo_candidates.py disagreement 3   # 후보3, group3만
  python3 scripts/validate_loyo_candidates.py all              # 전부, 전체 그룹(시간 오래 걸릴 수 있음)
결과는 매 실행마다 누적/병합되어 저장된다:
  experiments/baseline_lgbm/loyo_candidates_results.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 같은 scripts/ 디렉토리의 validate_loyo 임포트용

import numpy as np

from src.features import (
    add_forecast_disagreement_features,
    add_manufacturer_power_curve_feature,
    add_wind_direction_cyclical_features,
)
from src.preprocess import compute_ldaps_grid_speed_spread
from src.terrain_features import add_terrain_upwind_feature
from validate_loyo import OUT_DIR, RESULTS_PATH, run_group

CANDIDATES_RESULTS_PATH = OUT_DIR / "loyo_candidates_results.json"

CURTAILMENT_CANDIDATE = {
    1: {"curtailment_ratio": 0.20, "curtailment_wind_thresh": 6.0},
    2: {"curtailment_ratio": 0.35, "curtailment_wind_thresh": 6.0},
    3: {"curtailment_ratio": 0.20, "curtailment_wind_thresh": 6.0},
}

MODEL_SWAP_CANDIDATE = {1: "xgboost", 2: "xgboost", 3: "catboost"}

CANDIDATE_NAMES = (
    "curtailment", "model_swap", "disagreement", "combined_features",
    "manufacturer_curve", "terrain_exposure",
)
RESULT_KEY = {
    "curtailment": "curtailment_candidate_folds",
    "model_swap": "model_swap_candidate_folds",
    "disagreement": "disagreement_candidate_folds",
    "combined_features": "combined_features_candidate_folds",
    "manufacturer_curve": "manufacturer_curve_candidate_folds",
    "terrain_exposure": "terrain_exposure_candidate_folds",
}
SUMMARY_LABEL = {
    "curtailment": "후보 1 (커틀먼트 임계값 재탐색)",
    "model_swap": "후보 2 (XGBoost/CatBoost 모델 교체)",
    "disagreement": "후보 3 (예보 소스 간 불일치 feature 추가)",
    "combined_features": "후보 4 (disagreement + 풍향 sin/cos + LDAPS 격자 spread 묶음)",
    "manufacturer_curve": "후보 5 (터빈 제작사 공식 파워커브, 외부 데이터)",
    "terrain_exposure": "후보 7 (SRTM DEM 방향별 풍상측 지형고도차, 외부 데이터)",
}


def make_combined_extra_feature_fn(group_id: int):
    """후보 3(disagreement) + 풍향 sin/cos 인코딩 + LDAPS 격자 간 풍속 spread를
    하나로 묶은 extra_feature_fn을 만든다.

    격자 spread는 그룹당 한 번만 계산해 클로저에 담아두고(폴드마다 다시
    계산하면 낭비), train_feat/holdout_feat에 forecast_kst_dtm 기준으로 merge한다.
    """
    spread_df = compute_ldaps_grid_speed_spread(group_id, split="train")

    def _fn(df):
        df = add_forecast_disagreement_features(df)
        df = add_wind_direction_cyclical_features(df)
        df = df.merge(spread_df, on="forecast_kst_dtm", how="left")
        return df

    return _fn


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_stats(fold_results, group_id):
    scores = np.array([r["score"] for r in fold_results if r["group_id"] == group_id])
    if len(scores) == 0:
        return None
    mean = float(scores.mean())
    std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
    return mean, std, len(scores)


def get_baseline_stats():
    """scripts/validate_loyo.py가 이미 만들어둔 프로덕션(기본 커틀먼트+LightGBM) LOYO 결과."""
    baseline = load_json(RESULTS_PATH)
    stats = {}
    for gid in [1, 2, 3]:
        s = group_stats(baseline, gid)
        if s:
            stats[gid] = s
    return stats


def summarize(name, baseline_stats, candidate_folds):
    if not candidate_folds:
        return {}
    print(f"\n=== {name}: baseline 대비 delta ===")
    summary = {}
    for gid in sorted({r["group_id"] for r in candidate_folds}):
        cand = group_stats(candidate_folds, gid)
        if cand is None or gid not in baseline_stats:
            continue
        cand_mean, cand_std, n = cand
        base_mean, base_std, _ = baseline_stats[gid]
        delta = cand_mean - base_mean
        noise_band = base_std  # 프로덕션 LOYO std를 노이즈 기준으로 사용
        verdict = "신뢰 가능(노이즈 초과)" if abs(delta) > noise_band else "노이즈 수준(불확실)"
        print(f"  group{gid}: baseline={base_mean:.4f}(std={base_std:.4f}) -> "
              f"candidate={cand_mean:.4f}(std={cand_std:.4f}, n_folds={n})  "
              f"delta={delta:+.4f}  [{verdict}]")
        summary[str(gid)] = {
            "baseline_mean": base_mean, "baseline_std": base_std,
            "candidate_mean": cand_mean, "candidate_std": cand_std,
            "delta": delta, "verdict": verdict,
        }
    return summary


def run_curtailment_groups(groups):
    results = []
    for gid in groups:
        overrides = CURTAILMENT_CANDIDATE[gid]
        results.extend(run_group(gid, **overrides))
    return results


def run_model_swap_groups(groups):
    results = []
    for gid in groups:
        model_type = MODEL_SWAP_CANDIDATE[gid]
        params = load_json(OUT_DIR / f"group{gid}_{model_type}_optuna_best_params.json")
        results.extend(run_group(gid, model_type=model_type, model_params=params))
    return results


def run_disagreement_groups(groups):
    results = []
    for gid in groups:
        results.extend(run_group(gid, extra_feature_fn=add_forecast_disagreement_features))
    return results


def run_combined_features_groups(groups):
    results = []
    for gid in groups:
        fn = make_combined_extra_feature_fn(gid)
        results.extend(run_group(gid, extra_feature_fn=fn))
    return results


def run_manufacturer_curve_groups(groups):
    results = []
    for gid in groups:
        fn = lambda df, gid=gid: add_manufacturer_power_curve_feature(df, group_id=gid)
        results.extend(run_group(gid, extra_feature_fn=fn))
    return results


def run_terrain_exposure_groups(groups):
    results = []
    for gid in groups:
        fn = lambda df, gid=gid: add_terrain_upwind_feature(df, group_id=gid)
        results.extend(run_group(gid, extra_feature_fn=fn))
    return results


RUNNER = {
    "curtailment": run_curtailment_groups,
    "model_swap": run_model_swap_groups,
    "disagreement": run_disagreement_groups,
    "combined_features": run_combined_features_groups,
    "manufacturer_curve": run_manufacturer_curve_groups,
    "terrain_exposure": run_terrain_exposure_groups,
}


def main():
    args = sys.argv[1:]
    candidate = args[0] if args else "all"
    groups = [int(a) for a in args[1:]] or [1, 2, 3]
    assert candidate in CANDIDATE_NAMES + ("all",), candidate
    run_names = CANDIDATE_NAMES if candidate == "all" else (candidate,)

    existing = {RESULT_KEY[name]: [] for name in CANDIDATE_NAMES}
    if CANDIDATES_RESULTS_PATH.exists():
        existing.update(load_json(CANDIDATES_RESULTS_PATH))
        for name in CANDIDATE_NAMES:
            existing.setdefault(RESULT_KEY[name], [])

    for name in run_names:
        print(f"\n########## {SUMMARY_LABEL[name]} ##########")
        key = RESULT_KEY[name]
        kept = [r for r in existing[key] if r["group_id"] not in groups]
        existing[key] = kept + RUNNER[name](groups)

    with open(CANDIDATES_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장(누적): {CANDIDATES_RESULTS_PATH}")

    baseline_stats = get_baseline_stats()
    for name in CANDIDATE_NAMES:
        summary = summarize(SUMMARY_LABEL[name], baseline_stats, existing[RESULT_KEY[name]])
        existing[f"{name}_candidate_summary"] = summary

    with open(CANDIDATES_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
