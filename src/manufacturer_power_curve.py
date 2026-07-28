"""
터빈 제작사 공식 파워커브 (외부 공개 데이터, 실험용).

`data/open/info.xlsx` 확인 결과 이 대회의 터빈 구성은:
  - group1(태백가덕산 1~6호기) / group2(7~12호기): VESTAS V126, 3.6MW/기
    (info.xlsx "설비용량(MW)"=3.6), Hub Height 117m.
  - group3(태백원동 1~5호기): UNISON U136, 4.2MW/기, Hub Height 117m.

두 모델 모두 수년 전부터 공개된 상용 제품의 고정 사양(spec)이라 시계열이
아니다 — "예측기준시점 이전에 생성·공개된 정보만 사용"이라는 대회 규칙을
자명하게 만족한다(데이터 leakage 위험 없음).

⚠️ 아직 프로덕션 레시피에는 편입되지 않았다. LOYO로 검증 중
(scripts/validate_loyo_candidates.py "manufacturer_curve" 후보). 재현성
문서화는 docs/external_data_manufacturer_power_curve.md 참고.

출처
----
1) VESTAS V126-3.45MW (가변출력 3,300~3,600kW, 이 프로젝트의 3.6MW 설정에
   해당): https://en.wind-turbine-models.com/turbines/1249-vestas-v126-3-45
   (조회 2026-07-27, wind-turbine-models.com — 업계에서 널리 인용되는
   터빈 스펙 애그리게이터). 3~22.5 m/s, 0.5 m/s 간격, IEC 표준 공기밀도
   1.225 kg/m^3 기준. 원본 표는 정격 3,450kW 기준이라, 실제 이 프로젝트의
   배치(정격 3,600kW로 출력 상향 설정, Vestas가 "V126-3.45MW power
   optimised to 3.6MW"로 판매하는 것과 동일 계열로 확인됨)에 맞춰
   3600/3450 배율로 전체 곡선을 스케일링했다 — 근사이며, 실제 3.6MW
   전용 곡선이 아니라는 한계가 있다.
2) UNISON U136-4.2MW: https://www.unison.co.kr/product/4MW_Platform_U136
   (제작사 공식 홈페이지, 조회 2026-07-27). 페이지에 게시된 파워커브 차트
   원본 데이터(3~25 m/s, 1 m/s 간격, 정격 4,200kW)를 그대로 사용 — 가장
   신뢰도 높은 출처(제작사 1차 자료).

공기밀도 보정
-------------
제작사 공식 파워커브는 관행적으로 IEC 표준 공기밀도(1.225 kg/m^3, 해수면
15도 기준)로 공표된다. 이 대회 터빈은 태백(고지대)에 있어 실제 공기밀도가
표준보다 낮을 가능성이 있다(src.features.add_physics_features가 계산하는
ldaps_air_density 참고). IEC 61400-12-1의 관행적 밀도 보정 방식을 따라,
곡선을 조회하기 전에 풍속을 "표준밀도에서의 등가 풍속"으로 변환한다:

    v_corrected = v_actual * (rho_actual / rho_standard) ** (1/3)

(발전량이 대략 rho * v^3에 비례한다는 물리적 관계에서 유도 — 같은 발전량을
내려면 밀도가 낮을수록 더 높은 풍속이 필요하므로, 표준밀도 커브에 넣을
"등가 풍속"은 실제 풍속보다 낮게 보정된다.)
"""
import numpy as np

STANDARD_AIR_DENSITY = 1.225  # kg/m^3, IEC 61400-12-1 표준

# (풍속 m/s, 출력 kW) — 출처 1) 참고, 정격 3,450kW 기준 원본.
VESTAS_V126_3450KW_CURVE = [
    (3.0, 35.0), (3.5, 101.0), (4.0, 184.0), (4.5, 283.0), (5.0, 404.0),
    (5.5, 550.0), (6.0, 725.0), (6.5, 932.0), (7.0, 1172.0), (7.5, 1446.0),
    (8.0, 1760.0), (8.5, 2104.0), (9.0, 2482.0), (9.5, 2865.0), (10.0, 3187.0),
    (10.5, 3366.0), (11.0, 3433.0), (11.5, 3448.0), (12.0, 3450.0), (12.5, 3450.0),
    (13.0, 3450.0), (13.5, 3450.0), (14.0, 3450.0), (14.5, 3450.0), (15.0, 3450.0),
    (15.5, 3450.0), (16.0, 3450.0), (16.5, 3450.0), (17.0, 3450.0), (17.5, 3450.0),
    (18.0, 3450.0), (18.5, 3450.0), (19.0, 3450.0), (19.5, 3450.0), (20.0, 3450.0),
    (20.5, 3450.0), (21.0, 3450.0), (21.5, 3450.0), (22.0, 3450.0), (22.5, 3450.0),
]
VESTAS_V126_RATED_KW_SOURCE = 3450.0
VESTAS_V126_RATED_KW_DEPLOYED = 3600.0  # info.xlsx 설비용량(3.6MW/기)에 맞춘 스케일

# (풍속 m/s, 출력 kW) — 출처 2) 참고, unison.co.kr 공식 파워커브 원본 그대로.
UNISON_U136_4200KW_CURVE = [
    (3, 79.1), (4, 193.6), (5, 396.2), (6, 720.9), (7, 1178.9),
    (8, 1783.3), (9, 2502.2), (10, 3250.4), (11, 3851.9), (12, 4126.7),
    (13, 4185.7), (14, 4196.5), (15, 4198.8), (16, 4200.0), (17, 4200.0),
    (18, 4200.0), (19, 4200.0), (20, 4200.0), (21, 4200.0), (22, 4200.0),
    (23, 4200.0), (24, 4200.0), (25, 4200.0),
]

# group_id -> 이 모듈의 커브 키
TURBINE_BY_GROUP = {1: "vestas_v126", 2: "vestas_v126", 3: "unison_u136"}


def _make_lookup(curve: list[tuple[float, float]], scale: float = 1.0):
    speeds = np.array([c[0] for c in curve], dtype=float)
    powers = np.array([c[1] for c in curve], dtype=float) * scale

    def lookup(wind_speed):
        wind_speed = np.asarray(wind_speed, dtype=float)
        return np.interp(wind_speed, speeds, powers, left=0.0, right=powers[-1])

    return lookup


_LOOKUPS = {
    "vestas_v126": _make_lookup(
        VESTAS_V126_3450KW_CURVE, scale=VESTAS_V126_RATED_KW_DEPLOYED / VESTAS_V126_RATED_KW_SOURCE
    ),
    "unison_u136": _make_lookup(UNISON_U136_4200KW_CURVE, scale=1.0),
}


def air_density_correct_speed(wind_speed, air_density, standard: float = STANDARD_AIR_DENSITY):
    """실제 공기밀도에서의 풍속을 IEC 표준밀도 등가 풍속으로 변환(IEC 61400-12-1 관행)."""
    wind_speed = np.asarray(wind_speed, dtype=float)
    ratio = np.asarray(air_density, dtype=float) / standard
    ratio = np.clip(ratio, 0.5, 1.5)  # 극단적 이상치로 인한 왜곡 방지
    return wind_speed * ratio ** (1.0 / 3.0)


def estimate_power_kw(wind_speed, air_density, turbine: str):
    """turbine: "vestas_v126" 또는 "unison_u136". 공기밀도 보정 후 커브 조회, kW(=1시간 kWh) 반환."""
    v_std = air_density_correct_speed(wind_speed, air_density)
    return _LOOKUPS[turbine](v_std)
