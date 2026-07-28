"""
FICR 개선 시도 1단계 — capacity 스케일에 맞춘 Huber loss 재검증 (LOYO).

배경: rated_output_investigation.md 6번 섹션에서 objective를 L2/MAE/Huber로
바꿔봤을 때 Huber가 가장 나빴다(group1 0.4221, group3 0.4335) — 그런데
그때 쓴 Huber의 delta(alpha) 파라미터가 LightGBM 기본값(~1.0)이었다.
이 대회 타깃 스케일은 수천~2만 kWh인데 delta=1.0은 사실상 거의 모든 잔차가
"이상치 구간"으로 처리돼(Huber는 |잔차|>delta면 L1처럼, 이하면 L2처럼 동작)
L1과 거의 다를 바 없어지고, 그마저도 스케일이 안 맞아 그래디언트가 사실상
무의미해진 것으로 보인다 — 제대로 스케일링한 적이 없었던 것.

FICR의 정산 구간(오차율 <=6% → 4원, <=8% → 3원, 초과 → 0원, 오차율=
|pred-actual|/capacity)에 맞춰 delta를 capacity의 6~8% 부근(수천 kWh)으로
직접 스케일링하면, "이 정도 오차까지는 정밀하게(L2), 그 이상은 완만하게(L1
근사)"라는 모양이 얼추 FICR 보상구조와 비슷해진다는 가설로 재시도한다.
프로덕션 하이퍼파라미터(n_estimators/num_leaves/learning_rate 등)는 그대로
두고 objective만 huber로, delta(alpha)만 그리드서치해 순수하게 "손실함수
모양"의 효과만 분리해서 본다(6번 섹션과 같은 취지, delta만 제대로 스케일링).

실행 (레포 루트에서):
  python3 scripts/validate_huber_scaled.py <group_id> [alpha_frac ...]
  (alpha_frac 생략 시 기본 그리드 0.04~0.12 사용, capacity 대비 비율)
결과 누적: experiments/baseline_lgbm/loyo_huber_scaled_results.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.metrics import CAPACITY_KWH
from validate_loyo import OUT_DIR, load_params, RECIPE_CHOICE, run_group
from validate_loyo_candidates import get_baseline_stats

RESULTS_PATH = OUT_DIR / "loyo_huber_scaled_results.json"
DEFAULT_ALPHA_FRACS = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]


def run_alpha(group_id: int, alpha_frac: float):
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    recipe = RECIPE_CHOICE[group_id]
    base_params = load_params(group_id, recipe)
    params = dict(base_params)
    params["objective"] = "huber"
    params["alpha"] = alpha_frac * capacity
    return run_group(group_id, model_params=params)


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    group_id = int(args[0])
    alpha_fracs = [float(a) for a in args[1:]] if len(args) > 1 else DEFAULT_ALPHA_FRACS

    existing = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.setdefault(str(group_id), {})

    baseline_stats = get_baseline_stats()
    base_mean = baseline_stats[group_id][0]
    base_std = baseline_stats[group_id][1]
    print(f"\n=== group{group_id} baseline(L2) mean={base_mean:.4f} std={base_std:.4f} ===")

    for alpha_frac in alpha_fracs:
        print(f"\n--- alpha_frac={alpha_frac} (alpha={alpha_frac*CAPACITY_KWH[f'kpx_group_{group_id}']:.0f} kWh) ---")
        fold_results = run_alpha(group_id, alpha_frac)
        import numpy as np
        scores = np.array([r["score"] for r in fold_results])
        ficrs = np.array([r["ficr"] for r in fold_results])
        nmaes = np.array([r["nmae"] for r in fold_results])
        mean = float(scores.mean())
        std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        delta = mean - base_mean
        verdict = "신뢰 가능(노이즈 초과)" if abs(delta) > base_std else "노이즈 수준(불확실)"
        print(f"  -> alpha_frac={alpha_frac}: score={mean:.4f}(std={std:.4f}, delta={delta:+.4f}) "
              f"FICR={ficrs.mean():.4f} NMAE={nmaes.mean():.4f}  [{verdict}]")
        existing[str(group_id)][str(alpha_frac)] = {
            "mean": mean, "std": std, "delta": delta, "verdict": verdict,
            "ficr_mean": float(ficrs.mean()), "nmae_mean": float(nmaes.mean()),
            "fold_scores": scores.tolist(), "fold_ficr": ficrs.tolist(), "fold_nmae": nmaes.tolist(),
        }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장(누적): {RESULTS_PATH}")

    print(f"\n=== group{group_id} alpha_frac별 요약 ===")
    for af, r in sorted(existing[str(group_id)].items(), key=lambda kv: float(kv[0])):
        print(f"  alpha_frac={af}: score={r['mean']:.4f} delta={r['delta']:+.4f} "
              f"FICR={r.get('ficr_mean', float('nan')):.4f} NMAE={r.get('nmae_mean', float('nan')):.4f} "
              f"[{r['verdict']}]")


if __name__ == "__main__":
    main()
