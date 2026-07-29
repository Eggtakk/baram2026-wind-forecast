"""
FICR-shaped custom objective(src/ficr_objective.py)의 T(sigmoid 폭)와
lambda_l2(L2 안전망 강도)를 튜닝한다 — 지금까지는 ficr_weight만
그리드서치했고 T=0.01, lambda_l2=1.0은 고정값이었다(25번 섹션).

배경: T가 작을수록 6%/8% 경계 근처에서만 gradient/hessian이 날카롭게
집중되고(경계에서 먼 표본은 거의 무시), T가 클수록 더 넓은 구간에
완만하게 퍼진다. lambda_l2가 작을수록 FICR 항의 순수한 영향력이 커지고
(경계 밖 표본을 "포기"하는 경향이 강해짐), lambda_l2가 크면 L2가 여전히
지배적이라 FICR 항의 효과가 희석된다.

group2(FICR objective가 실제 배포된 유일한 그룹, ficr_weight=0.008 고정)를
대상으로 T x lambda_l2 2D 그리드서치.

실행: python3 scripts/tune_ficr_objective_hparams.py <group_id> [ficr_weight]
결과 누적: experiments/baseline_lgbm/loyo_ficr_hparams_results.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.ficr_objective import make_ficr_objective
from src.metrics import CAPACITY_KWH
from validate_loyo import OUT_DIR, RECIPE_CHOICE, load_params, run_group
from validate_loyo_candidates import get_baseline_stats

RESULTS_PATH = OUT_DIR / "loyo_ficr_hparams_results.json"

T_GRID = [0.005, 0.01, 0.02, 0.04]
LAMBDA_L2_GRID = [0.3, 0.5, 1.0, 2.0]
DEFAULT_FICR_WEIGHT = {2: 0.008, 3: 0.001}


def run_combo(group_id: int, ficr_weight: float, T: float, lambda_l2: float):
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    recipe = RECIPE_CHOICE[group_id]
    base_params = load_params(group_id, recipe)
    params = dict(base_params)
    params["objective"] = make_ficr_objective(capacity, ficr_weight=ficr_weight, lambda_l2=lambda_l2, T=T)
    return run_group(group_id, model_params=params)


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    group_id = int(args[0])
    ficr_weight = float(args[1]) if len(args) > 1 else DEFAULT_FICR_WEIGHT[group_id]

    Ts = T_GRID
    lambdas = LAMBDA_L2_GRID
    if len(args) > 2:
        Ts = [float(x) for x in args[2].split(",")]
    if len(args) > 3:
        lambdas = [float(x) for x in args[3].split(",")]

    existing = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.setdefault(str(group_id), {})

    baseline_stats = get_baseline_stats()
    base_mean, base_std, _ = baseline_stats[group_id]
    print(f"\n=== group{group_id} L2 baseline mean={base_mean:.4f} std={base_std:.4f} "
          f"(ficr_weight={ficr_weight} 고정) ===")

    for T in Ts:
        for lam in lambdas:
            key = f"T{T}_lam{lam}"
            fold_results = run_combo(group_id, ficr_weight, T, lam)
            scores = np.array([r["score"] for r in fold_results])
            ficrs = np.array([r["ficr"] for r in fold_results])
            nmaes = np.array([r["nmae"] for r in fold_results])
            mean = float(scores.mean())
            std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
            delta = mean - base_mean
            verdict = "신뢰 가능(노이즈 초과)" if abs(delta) > base_std else "노이즈 수준"
            print(f"  T={T:.3f} lambda_l2={lam:.2f}: score={mean:.4f}(std={std:.4f}, delta={delta:+.4f}) "
                  f"FICR={ficrs.mean():.4f} NMAE={nmaes.mean():.4f}  [{verdict}]")
            existing[str(group_id)][key] = {
                "T": T, "lambda_l2": lam, "ficr_weight": ficr_weight,
                "mean": mean, "std": std, "delta": delta, "verdict": verdict,
                "ficr_mean": float(ficrs.mean()), "nmae_mean": float(nmaes.mean()),
                "fold_scores": scores.tolist(),
            }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장(누적): {RESULTS_PATH}")

    print(f"\n=== group{group_id} 요약 (delta 기준 정렬) ===")
    rows = sorted(existing[str(group_id)].values(), key=lambda r: -r["delta"])
    for r in rows[:10]:
        print(f"  T={r['T']:.3f} lambda_l2={r['lambda_l2']:.2f} ficr_weight={r['ficr_weight']}: "
              f"delta={r['delta']:+.4f} FICR={r['ficr_mean']:.4f} NMAE={r['nmae_mean']:.4f} [{r['verdict']}]")


if __name__ == "__main__":
    main()
