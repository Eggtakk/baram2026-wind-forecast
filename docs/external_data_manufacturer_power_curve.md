# 외부 공개 데이터 — 터빈 제작사 공식 파워커브

대회 규칙(대회 개요 페이지 "대회규칙 및 유의사항" 3~4절) 상 외부 데이터는
아래 조건을 만족하면 사용 가능하다:

- 예측기준시점 이전에 생성·공개·확정되어 실제로 활용 가능했던 정보일 것
- 출처, 수집 방법, 수집 시점, 사용 변수, 라이선스, 전처리 코드를 포함해
  재현 가능하게 문서화할 것 (2차 평가 대상자로 지정될 경우 제출 필요)

이 문서는 `src/manufacturer_power_curve.py`가 사용하는 터빈 제작사 공식
파워커브 데이터의 출처와 재현 방법을 기록한다.

## 왜 이 데이터가 규칙을 만족하는가

터빈 제작사의 공식 파워커브는 **터빈 설계 사양(고정값)**이지 시계열
관측치가 아니다. VESTAS V126과 UNISON U136 모두 2015~2019년경부터
공개된 상용 제품이며(아래 출처 페이지의 "등록일" 참고), 이 대회의 학습
기간(2022년~) 및 평가 기간(2025년) 어느 시점보다도 훨씬 이전에 이미
공개되어 있었다. 따라서 "예측기준시점 이전에 생성·공개된 정보"라는 조건을
모든 예측 행에 대해 자명하게 만족한다.

## 터빈 모델 확인 근거

`data/open/info.xlsx`(대회 공식 제공 파일, `info` 시트)에서 확인:

| KPX 그룹 | 명칭 | 제작사 | 모델명 | 호기 | Hub Height | 설비용량(1기) |
|---|---|---|---|---|---|---|
| 1 | 태백가덕산 | VESTAS | V126 | 1~6 | 117m | 3.6 MW |
| 2 | 태백가덕산 | VESTAS | V126 | 7~12 | 117m | 3.6 MW |
| 3 | 태백원동 | UNISON | U136 | 1~5 | 117m | 4.2 MW |

## 데이터 출처

### 1) VESTAS V126, 정격 3,450kW (그룹1/2, 프로젝트 배치는 3,600kW로 출력 상향)

- URL: https://en.wind-turbine-models.com/turbines/1249-vestas-v126-3-45
- 조회일: 2026-07-27
- 출처 성격: wind-turbine-models.com (Lucas Bauer & Silvio Matysik 운영,
  2011년부터 운영된 풍력터빈 스펙 데이터베이스, 업계에서 널리 인용됨).
  페이지 내 "등록일 27.05.2015"로 명시 — 이 대회 데이터 기간보다 훨씬
  이전부터 공개.
- 수집 방법: 브라우저로 페이지 접속 후, 파워커브 차트를 렌더링하는
  `window.myChart.data.datasets[0].data`(JS 전역 변수, Chart.js 기반)에서
  풍속 3.0~22.5 m/s(0.5 m/s 간격) x 출력(kW) 배열을 직접 추출. 원본은
  IEC 표준 공기밀도(1.225 kg/m^3) 기준.
- ⚠️ 한계: 실제 이 프로젝트의 터빈은 info.xlsx 기준 3.6MW로 설정돼 있는데,
  이 출처의 곡선은 정격 3,450kW(가변출력 범위 3,300~3,600kW 중 하나) 기준이다.
  검색 결과 Vestas가 "V126-3.45MW power optimised to 3.6MW"로 판매한
  사례가 확인되어 같은 물리적 터빈의 출력 상향 설정으로 추정되나, 3.6MW
  전용 공식 곡선 원본을 찾지 못해 **전체 곡선을 3600/3450 배율로 균일
  스케일링**하는 근사를 사용했다. 실제 곡선(특히 정격출력 도달 풍속)이
  살짝 다를 수 있다는 한계가 있음 — `src/manufacturer_power_curve.py`
  docstring에도 동일하게 명시.
- 라이선스: 사이트 자체 이용약관에 따름(공개 열람 가능한 스펙 데이터,
  상업적 재배포가 아닌 연구/분석 목적의 참고 인용). 재현이 필요하면 위
  URL에서 동일 절차로 다시 추출 가능.

### 2) UNISON U136, 정격 4,200kW (그룹3)

- URL: https://www.unison.co.kr/product/4MW_Platform_U136
- 조회일: 2026-07-27
- 출처 성격: **제작사(UNISON Co., Ltd.) 공식 홈페이지** — 가장 신뢰도 높은
  1차 자료.
- 수집 방법: 제품 페이지의 "Power Curve" 섹션을 렌더링하는 `window.chart_1`
  (Google Charts 기반, [[풍속, 출력kW], ...] 형태의 원시 배열)을 그대로
  추출. 풍속 3~25 m/s(1 m/s 간격), 정격 4,200kW.
- 페이지 명시 사양과 교차 확인: 정격출력 4,200kW / 로터직경 136m / 허브높이
  95m·**117m**(이 프로젝트와 정확히 일치) / 정격풍속 11.3m/s — info.xlsx와
  전부 일치해 올바른 모델임을 확인.
- 라이선스: 제작사 공식 제품 정보 페이지, 공개 열람 가능.

## 공기밀도 보정

두 곡선 모두 IEC 표준 공기밀도(1.225 kg/m³, 해수면 15℃) 기준으로 공표된
것으로 가정한다(제작사 공식 자료의 일반적 관행). 이 대회 터빈은 태백(고지대)
에 있어 실제 학습 데이터의 `ldaps_air_density`(src/features.py 계산)가
1.09~1.26 kg/m³로 표준보다 낮게 분포한다. IEC 61400-12-1 관행에 따라 곡선
조회 전 풍속을 표준밀도 등가 풍속으로 보정한다:

```
v_corrected = v_actual * (rho_actual / rho_standard) ** (1/3)
```

## 재현 방법

```python
from src.manufacturer_power_curve import estimate_power_kw
# wind_speed: 허브높이 풍속(m/s), air_density: kg/m^3, turbine: "vestas_v126" | "unison_u136"
estimate_power_kw(wind_speed=11.0, air_density=1.17, turbine="vestas_v126")
```

전체 원시 (풍속, 출력) 테이블은 `src/manufacturer_power_curve.py`의
`VESTAS_V126_3450KW_CURVE` / `UNISON_U136_4200KW_CURVE` 상수에 하드코딩되어
있어(외부 API 호출 없이) 코드만으로 완전히 재현 가능하다.

## 현재 상태 (2026-07-27)

`src/features.py::add_manufacturer_power_curve_feature`로 feature화해
`scripts/validate_loyo_candidates.py`의 "manufacturer_curve" 후보로 LOYO
검증했으나, 세 그룹 모두 baseline std 대비 유의미한 delta를 만들지 못해
**프로덕션 레시피에는 아직 편입하지 않았다**(상세: 
`experiments/baseline_lgbm/rated_output_investigation.md` 16번 섹션).
