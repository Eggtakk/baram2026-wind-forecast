"""
info.xlsx로부터 그룹1/2/3 config(configs/group{n}.yaml)를 자동 생성한다.

터빈 -> KPX그룹 매핑은 info.xlsx의 'KPX그룹' 컬럼이 각 그룹의 첫 터빈 행에만
채워져 있고 나머지는 비어있는 형태이므로, forward-fill로 그룹을 채운 뒤
그룹별로 묶는다. 그룹별 centroid를 기준으로 LDAPS/GFS 중 가장 가까운
격자를 선정해 config에 함께 저장한다.

재실행하면 항상 같은 결과가 나오는 결정적(deterministic) 스크립트이므로,
데이터가 바뀌지 않는 한 다시 돌릴 필요는 없다.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "open"
CONFIG_DIR = ROOT / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

N_LDAPS_GRIDS = 4
N_GFS_GRIDS = 1


def dms_to_dd(s: str) -> float:
    m = re.match(r"(\d+)\D+(\d+)\D+([\d.]+)\"?([NSEW])", s.strip())
    d, mi, se, hemi = m.groups()
    val = float(d) + float(mi) / 60 + float(se) / 3600
    return -val if hemi in ("S", "W") else val


def load_turbine_info() -> pd.DataFrame:
    df = pd.ExcelFile(DATA / "info.xlsx").parse("info", header=3)
    df.columns = [str(c).strip() for c in df.columns]
    df["KPX그룹"] = df["KPX그룹"].ffill().astype(int)
    df["그룹설비용량(MW)"] = df["그룹설비용량(MW)"].ffill().astype(float)
    coords = df["좌표(Google)"].astype(str)
    df["lat"] = coords.apply(lambda c: dms_to_dd(c.split()[0]))
    df["lon"] = coords.apply(lambda c: dms_to_dd(c.split()[1]))
    return df


def nearest_grids(csv_path: Path, target_lat: float, target_lon: float, n: int) -> list[int]:
    df = pd.read_csv(csv_path, usecols=["grid_id", "latitude", "longitude"], nrows=200_000)
    g = df.drop_duplicates("grid_id")[["grid_id", "latitude", "longitude"]].copy()
    g["dist"] = np.sqrt((g["latitude"] - target_lat) ** 2 + (g["longitude"] - target_lon) ** 2)
    return g.sort_values("dist")["grid_id"].head(n).astype(int).tolist()


def main():
    info = load_turbine_info()

    for group_id, g in info.groupby("KPX그룹"):
        manufacturer = g["제작사"].iloc[0].lower()
        turbines = [f"{manufacturer}_wtg{int(h):02d}" for h in g["호기"]]
        capacity_mw = float(g["그룹설비용량(MW)"].iloc[0])
        lat, lon = float(g["lat"].mean()), float(g["lon"].mean())

        ldaps_grids = nearest_grids(DATA / "train" / "ldaps_train.csv", lat, lon, N_LDAPS_GRIDS)
        gfs_grids = nearest_grids(DATA / "train" / "gfs_train.csv", lat, lon, N_GFS_GRIDS)

        cfg = {
            "group_id": int(group_id),
            "label_column": f"kpx_group_{int(group_id)}",
            "manufacturer": manufacturer,
            "scada_file": f"scada_{manufacturer}_train.csv",
            "turbines": turbines,
            "capacity_mw": capacity_mw,
            "capacity_kwh": capacity_mw * 1000,
            "centroid": {"lat": round(lat, 4), "lon": round(lon, 4)},
            "ldaps_grids": ldaps_grids,
            "gfs_grids": gfs_grids,
        }

        out_path = CONFIG_DIR / f"group{int(group_id)}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"wrote {out_path}")
        print(yaml.dump(cfg, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
