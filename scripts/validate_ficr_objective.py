"""
FICR 개선 시도 2단계 — 진짜 계단형(6%/8% 이중 sigmoid) custom objective LOYO 검증.

배경: rated_output_investigation.md 24번 섹션(scaled Huber, 기각)의 후속.
Huber는 경계 밖에서도 오차가 커질수록 페널티가 계속 커지는 선형 구조라
FICR의 진짜 계단형 보상(8% 넘으면 오차 크기 무관하게 전부 0원)을 반영하지
못했다. `src/ficr_objective.py`는 6%/8% 지점에서 매끄럽게 전환되는 두 개의
로지스틱 sigmoid로 실제 unit_price(e) 계단함수를 근사하고, FICR의 실제
발전량(actual) 가중치까지 반영한 grad/hess를 직접 유도한 custom objective다.
(순수 FICR항만 쓰면 이미 8%를 크게 넘은 표본의 gradient가 거의 0이 되어
모델이 그 표본들을 발산시킬 위험이 있어, 약한 L2 항을 안전망으로 섞음.)

이 스크립트는 프로덕션 하이퍼파라미터는 그대로 두고 objective만 교체해
ficr_weight(FICR 항의 상대적 세기)를 그리드서치한다 — lambda_l2=1.0,
T=0.01(sigmoid 폭 1%p) 고정.

실행 (레포 루트에서):
  python3 scripts/validate_ficr_objective.py <group_id> [ficr_weight ...]
결과 누적: experiments/baseline_lgbm/loyo_ficr_objective_results.json
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

RESULTS_PATH = OUT_DIR / "loyo_ficr_objective_results.json"
DEFAULT_FICR_WEIGHTS = [0.001, 0.003, 0.01, 0.03, 0.1]
LAMBDA_L2 = 1.0
T = 0.01


def run_weight(group_id: int, ficr_weight: float):
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    recipe = RECIPE_CHOICE[group_id]
    base_params = load_params(group_id, recipe)
    params = dict(base_params)
    params["objective"] = make_ficr_objective(
        capacity, ficr_weight=ficr_weight, lambda_l2=LAMBDA_L2, T=T
    )
    return run_group(group_id, model_params=params)


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    group_id = int(args[0])
    ficr_weights = [float(a) for a in args[1:]] if len(args) > 1 else DEFAULT_FICR_WEIGHTS

    existing = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.setdefault(str(group_id), {})

    baseline_stats = get_baseline_stats()
    base_mean, base_std, _ = baseline_stats[group_id]
    print(f"\n=== group{group_id} baseline(L2) mean={base_mean:.4f} std={base_std:.4f} ===")

    for fw in ficr_weights:
        print(f"\n--- ficr_weight={fw} (lambda_l2={LAMBDA_L2}, T={T}) ---")
        fold_results = run_weight(group_id, fw)
        scores = np.array([r["score"] for r in fold_results])
        ficrs = np.array([r["ficr"] for r in fold_results])
        nmaes = np.array([r["nmae"] for r in fold_results])
        mean = float(scores.mean())
        std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        delta = mean - base_mean
        verdict = "신뢰 가능(노이즈 초과)" if abs(delta) > base_std else "노이즈 수준(불확실)"
        print(f"  -> ficr_weight={fw}: score={mean:.4f}(std={std:.4f}, delta={delta:+.4f}) "
              f"FICR={ficrs.mean():.4f} NMAE={nmaes.mean():.4f}  [{verdict}]")
        existing[str(group_id)][str(fw)] = {
            "mean": mean, "std": std, "delta": delta, "verdict": verdict,
            "ficr_mean": float(ficrs.mean()), "nmae_mean": float(nmaes.mean()),
            "fold_scores": scores.tolist(), "fold_ficr": ficrs.tolist(), "fold_nmae": nmaes.tolist(),
        }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장(누적): {RESULTS_PATH}")

    print(f"\n=== group{group_id} ficr_weight별 요약 ===")
    for fw, r in sorted(existing[str(group_id)].items(), key=lambda kv: float(kv[0])):
        print(f"  ficr_weight={fw}: score={r['mean']:.4f} delta={r['delta']:+.4f} "
              f"FICR={r['ficr_mean']:.4f} NMAE={r['nmae_mean']:.4f} [{r['verdict']}]")


if __name__ == "__main__":
    main()
