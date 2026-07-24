"""
대회 공식 평가 산식 재현.

출처: DACON 코드공유 "평가 산식 코드"
https://dacon.io/competitions/official/236727/codeshare/14035
Score = 0.5 x (1-NMAE) + 0.5 x (FICR)

공식 예시 코드(위 링크)를 그대로 옮기되, 대회 개요 페이지(평가 탭)에 별도로
명시된 규칙 하나를 추가했다:
  "평가는 실제 발전량이 설비용량의 10% 이상인 시간대만 대상으로 합니다."
이 필터는 코드공유의 예시 코드에는 생략되어 있어("핵심 계산 로직만 포함한
예시 코드"라고 명시됨), 이 모듈에서는 기본적으로 적용한다
(`min_actual_ratio=0.10`). 완전히 동일한 재현이 필요하면 0.0으로 끄면 된다.

정산 단가(FICR) 기준: 오차율(NMAE) <= 6% → 4원/kWh, <= 8% → 3원/kWh, 초과 → 0원.
"""
import numpy as np
import pandas as pd

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}


def group_nmae_ficr(
    actual: np.ndarray,
    forecast: np.ndarray,
    capacity: float,
    min_actual_ratio: float = 0.10,
) -> tuple[float, float, int]:
    """단일 그룹에 대한 NMAE, FICR을 계산한다.

    Returns
    -------
    nmae: float
    ficr: float
    n_eval: 평가 대상(설비용량의 min_actual_ratio 이상)이었던 시간 수
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    mask = actual >= min_actual_ratio * capacity
    if mask.sum() == 0:
        return np.nan, np.nan, 0

    a = actual[mask]
    f = forecast[mask]

    error_rate = np.abs(f - a) / capacity
    nmae = float(np.mean(error_rate))

    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    earned_settlement = np.sum(a * unit_price)
    max_settlement = np.sum(a * 4.0)
    ficr = float(earned_settlement / max_settlement) if max_settlement > 0 else np.nan

    return nmae, ficr, int(mask.sum())


def competition_score(
    answer_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target_cols: list[str] = TARGET_COLS,
    capacity_kwh: dict = CAPACITY_KWH,
    min_actual_ratio: float = 0.10,
) -> dict:
    """3개 그룹 전체에 대한 공식 산식(Score, 1-NMAE, FICR)을 계산한다.

    answer_df / pred_df는 같은 행 순서(또는 동일 키로 이미 정렬됨)를 가정한다.
    한 그룹만 검증하고 싶으면 `group_nmae_ficr`를 직접 쓰거나, 나머지 그룹
    컬럼을 정답과 동일한 값으로 채워 넣어 영향이 없게 만들면 된다
    (validate_single_group 참고).
    """
    group_nmae, group_ficr, group_n = [], [], []
    for col in target_cols:
        nmae, ficr, n_eval = group_nmae_ficr(
            answer_df[col].to_numpy(dtype=float),
            pred_df[col].to_numpy(dtype=float),
            capacity_kwh[col],
            min_actual_ratio=min_actual_ratio,
        )
        group_nmae.append(nmae)
        group_ficr.append(ficr)
        group_n.append(n_eval)

    one_minus_nmae = 1 - float(np.nanmean(group_nmae))
    ficr = float(np.nanmean(group_ficr))
    total_score = 0.5 * one_minus_nmae + 0.5 * ficr

    return {
        "total_score": total_score,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "group_nmae": dict(zip(target_cols, group_nmae)),
        "group_ficr": dict(zip(target_cols, group_ficr)),
        "group_n_eval": dict(zip(target_cols, group_n)),
    }


def validate_single_group(
    actual: np.ndarray,
    forecast: np.ndarray,
    group_id: int,
    min_actual_ratio: float = 0.10,
) -> dict:
    """한 그룹만 검증할 때 쓰는 편의 함수 (로컬 holdout 검증용).

    나머지 두 그룹은 총점 산식에서 "예측=정답"으로 취급해 영향이 없게 만든 뒤,
    이 그룹의 1-NMAE / FICR 만 의미 있게 계산한다. 대회 리더보드처럼 3그룹
    평균 total_score까지 보고 싶으면 세 그룹 결과를 모아 competition_score에
    넣을 것 — 이 함수는 "내 그룹 하나"의 성능만 빠르게 보기 위한 것이다.
    """
    label_col = f"kpx_group_{group_id}"
    capacity = CAPACITY_KWH[label_col]
    nmae, ficr, n_eval = group_nmae_ficr(actual, forecast, capacity, min_actual_ratio)
    return {
        "group_id": group_id,
        "nmae": nmae,
        "one_minus_nmae": 1 - nmae if not np.isnan(nmae) else np.nan,
        "ficr": ficr,
        "n_eval": n_eval,
        "n_total": len(actual),
    }


def analyze_error_bands(
    actual: np.ndarray,
    forecast: np.ndarray,
    capacity: float,
    min_actual_ratio: float = 0.10,
) -> dict:
    """정격출력(설비용량) 대비 실제 발전량 구간별 오차율 분포를 분석한다.

    평가 대상(actual >= min_actual_ratio * capacity)만 포함. 구간은
    capacity_ratio(= actual/capacity)를 10%p 단위로 나눈다.

    Returns
    -------
    dict with:
      - "overall": {"pct_le6", "pct_6to8", "pct_over8", "mean_error_rate"}
      - "by_capacity_band": DataFrame(index=band_label) with
            n, mean_error_rate, pct_over8, mean_bias(=mean(forecast-actual)/capacity)
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    mask = actual >= min_actual_ratio * capacity
    a, f = actual[mask], forecast[mask]

    error_rate = np.abs(f - a) / capacity
    bias = (f - a) / capacity
    capacity_ratio = a / capacity

    overall = {
        "pct_le6": float(np.mean(error_rate <= 0.06)),
        "pct_6to8": float(np.mean((error_rate > 0.06) & (error_rate <= 0.08))),
        "pct_over8": float(np.mean(error_rate > 0.08)),
        "mean_error_rate": float(np.mean(error_rate)),
    }

    edges = np.arange(0.1, 1.01, 0.1)
    labels = [f"{int(edges[i]*100)}-{int(edges[i+1]*100)}%" for i in range(len(edges) - 1)]
    band_idx = np.clip(np.digitize(capacity_ratio, edges[1:-1]), 0, len(labels) - 1)

    rows = []
    for i, label in enumerate(labels):
        sel = band_idx == i
        n = int(sel.sum())
        rows.append(
            {
                "band": label,
                "n": n,
                "mean_error_rate": float(np.mean(error_rate[sel])) if n else np.nan,
                "pct_over8": float(np.mean(error_rate[sel] > 0.08)) if n else np.nan,
                "mean_bias": float(np.mean(bias[sel])) if n else np.nan,
            }
        )
    by_band = pd.DataFrame(rows).set_index("band")

    return {"overall": overall, "by_capacity_band": by_band}
