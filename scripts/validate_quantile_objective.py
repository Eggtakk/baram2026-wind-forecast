"""
28/29번 섹션 후속 — 물리 파워커브 anchoring(29번, 거의 무효과)이 실패한 뒤
재해석: 4/5번 섹션에서 이미 확인된 근본 원인은 "곡선 모양"이 아니라 예보
풍속 자체가 고풍속에서 체계적으로 과소평가되고(bias) 동시에 불확실성도
커진다(MAE 1.8m/s -> 6.8~7.4m/s)는 것이었다 — 이는 트리 모델이 feature
재가공만으로는 못 고치는 입력 데이터의 구조적 한계라고 결론지어졌었다.

이 스크립트는 "평균(conditional mean)이 아니라 상위 분위수(quantile)를
예측하면 이 비대칭적 불확실성을 보정할 수 있는가"를 시험한다 — 예보풍속이
고풍속을 과소평가하는 경향이 있다면, 같은 입력 조건에서도 실제 발전량의
분포는 L2 평균 예측보다 위쪽으로 치우쳐 있을 가능성이 있다(예보가 낮게
나온 시간대 중에서도 실제로는 더 강한 바람이었을 경우가 통계적으로 더
많을 것이므로). LightGBM 네이티브 quantile objective(alpha>0.5)로 이를
직접 시험한다 — 단, alpha를 올리면 저출력 구간(이미 양의 편향/과대예측)의
편향이 더 악화될 위험이 있어 NMAE/FICR을 함께 본다.

실행: python3 scripts/validate_quantile_objective.py <group_id> [alpha ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from validate_loyo import RECIPE_CHOICE, load_params, run_group
from validate_loyo_candidates import get_baseline_stats

DEFAULT_ALPHAS = [0.5, 0.55, 0.6, 0.65, 0.7]


def run_alpha(group_id: int, alpha: float):
    recipe = RECIPE_CHOICE[group_id]
    params = dict(load_params(group_id, recipe))
    params["objective"] = "quantile"
    params["alpha"] = alpha
    return run_group(group_id, model_params=params)


def main():
    args = sys.argv[1:]
    group_id = int(args[0])
    alphas = [float(a) for a in args[1:]] if len(args) > 1 else DEFAULT_ALPHAS

    baseline_stats = get_baseline_stats()
    base_mean, base_std = baseline_stats[group_id][0], baseline_stats[group_id][1]
    print(f"\n=== group{group_id} L2 baseline mean={base_mean:.4f} std={base_std:.4f} ===")

    for alpha in alphas:
        fold_results = run_alpha(group_id, alpha)
        scores = np.array([r["score"] for r in fold_results])
        nmaes = np.array([r["nmae"] for r in fold_results])
        ficrs = np.array([r["ficr"] for r in fold_results])
        mean = scores.mean()
        delta = mean - base_mean
        verdict = "신뢰 가능(노이즈 초과)" if abs(delta) > base_std else "노이즈 수준"
        print(f"  -> alpha={alpha}: score={mean:.4f}(delta={delta:+.4f}) "
              f"NMAE={nmaes.mean():.4f} FICR={ficrs.mean():.4f} [{verdict}]")


if __name__ == "__main__":
    main()
