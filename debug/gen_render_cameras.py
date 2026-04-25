#!/usr/bin/env python
"""
為 JAX_068_007_p0201 場景計算新感測器的裁切偏移量 (crop offset)。

K 和 W2C 直接從已有的 camera_pose JSON 讀取，不需要重新計算。
本腳本只需用 RPC 計算每個新感測器在 p0201 地理位置對應的 crop offset。

Usage:
    conda activate mvsplat && cd mvsplat
    python debug/gen_render_cameras.py

Output: debug/p0201_crop_offsets.json
"""

import json
import sys
from pathlib import Path

import numpy as np

# ── 路徑設定 ──────────────────────────────────────────────────────────────
ROOT          = Path("/project/winston")
TRACK3_DIR    = ROOT / "datasets/DFC2019/Track3-RGB-1"
TRAINING_DIR  = ROOT / "datasets/DFC2019/overfit_068/training/JAX_068_014_p0304"
ENU_ORIGIN_FILE = ROOT / "datasets/DFC2019/overfit/enu_origin/JAX_068.json"
OUTPUT_FILE   = Path(__file__).parent / "p0304_crop_offsets.json"

CAMERA_POSE_DIR = ROOT / "datasets/DFC2019/overfit/camera_pose/JAX_068_cameras"

MASTER_SENSOR = "014"
TRAIN_SENSORS = {"009", "014", "020", "022"}
SCENE_RANGE   = 20.0  # near/far 計算用 (meters)


# ── RPC 工具函數 (numpy) ──────────────────────────────────────────────────

def parse_rpc_gdal(ds) -> np.ndarray:
    md = ds.GetMetadata("RPC")
    if not md:
        raise RuntimeError(f"TIF 中找不到 RPC 元數據")
    keys = ["LINE_OFF", "LINE_SCALE", "SAMP_OFF", "SAMP_SCALE",
            "LAT_OFF", "LAT_SCALE", "LONG_OFF", "LONG_SCALE",
            "HEIGHT_OFF", "HEIGHT_SCALE"]
    coeffs = [float(md.get(k, 0)) for k in keys]
    for prefix in ["LINE_NUM_COEFF", "LINE_DEN_COEFF", "SAMP_NUM_COEFF", "SAMP_DEN_COEFF"]:
        vals = [float(x) for x in md.get(prefix, "").split()]
        coeffs.extend(vals if len(vals) == 20 else vals + [0] * (20 - len(vals)))
    return np.array(coeffs, dtype=np.float64)


def _poly(c, x, y, z):
    return (c[0] + c[1]*x + c[2]*y + c[3]*z + c[4]*x*y + c[5]*x*z + c[6]*y*z
            + c[7]*x**2 + c[8]*y**2 + c[9]*z**2 + c[10]*x*y*z + c[11]*x**3
            + c[12]*x*y**2 + c[13]*x*z**2 + c[14]*x**2*y + c[15]*y**3
            + c[16]*y*z**2 + c[17]*x**2*z + c[18]*y**2*z + c[19]*z**3)


def rpc_forward(rpc, lat, lon, height):
    p, l = (lat - rpc[4]) / rpc[5], (lon - rpc[6]) / rpc[7]
    h = (height - rpc[8]) / rpc[9]
    row = _poly(rpc[10:30], l, p, h) / _poly(rpc[30:50], l, p, h) * rpc[1] + rpc[0]
    col = _poly(rpc[50:70], l, p, h) / _poly(rpc[70:90], l, p, h) * rpc[3] + rpc[2]
    return float(row), float(col)


def rpc_inverse(rpc, row, col, height, iterations=50):
    lat, lon = rpc[4], rpc[6]
    for _ in range(iterations):
        r_est, c_est = rpc_forward(rpc, lat, lon, height)
        delta = 1e-5
        r_dlat, c_dlat = rpc_forward(rpc, lat + delta, lon, height)
        r_dlon, c_dlon = rpc_forward(rpc, lat, lon + delta, height)
        dr_dlat = (r_dlat - r_est) / delta;  dc_dlat = (c_dlat - c_est) / delta
        dr_dlon = (r_dlon - r_est) / delta;  dc_dlon = (c_dlon - c_est) / delta
        det = dr_dlat * dc_dlon - dr_dlon * dc_dlat
        if abs(det) < 1e-15:
            break
        dlat = (dc_dlon * (row - r_est) - dr_dlon * (col - c_est)) / det
        dlon = (dr_dlat * (col - c_est) - dc_dlat * (row - r_est)) / det
        lat, lon = lat + dlat, lon + dlon
    return lat, lon


def near_far_from_camera_json(stem: str, rpc: np.ndarray, enu_origin: list,
                               scene_range: float = SCENE_RANGE) -> tuple:
    """
    從 camera_pose JSON 讀取 c2w[2,3]（相機 ENU-Z），
    和 training_step 相同的邏輯計算 near/far。

    dist_eff = (h_enu_origin + cam_z) - h_off_rpc
    near     = max(1.0, dist_eff - scene_range)
    far      = max(near + 1, dist_eff + scene_range)
    """
    cam_json = CAMERA_POSE_DIR / f"{stem}.json"
    with open(cam_json) as f:
        cam = json.load(f)
    W2C = np.array(cam["W2C"], dtype=np.float64).reshape(4, 4)
    C2W = np.linalg.inv(W2C)
    cam_z = float(C2W[2, 3])          # camera ENU-Z above ENU origin

    h_enu  = float(enu_origin[2])     # ENU origin MSL height
    h_off  = float(rpc[8])            # HEIGHT_OFF from RPC (ground MSL)
    cam_msl    = h_enu + cam_z
    dist_eff   = cam_msl - h_off

    near = max(1.0, dist_eff - scene_range)
    far  = max(near + 1.0, dist_eff + scene_range)
    return near, far


def main():
    from osgeo import gdal

    with open(ENU_ORIGIN_FILE) as f:
        enu_origin = json.load(f)
    print(f"ENU origin: lat={enu_origin[0]:.6f}, lon={enu_origin[1]:.6f}, h={enu_origin[2]:.2f} m")

    # master 的裁切位置
    with open(TRAINING_DIR / "crop_offsets.json") as f:
        crop_offsets = json.load(f)
    master_col = crop_offsets[f"JAX_068_{MASTER_SENSOR}_RGB"]["col_start"]
    master_row = crop_offsets[f"JAX_068_{MASTER_SENSOR}_RGB"]["row_start"]
    print(f"Master (007) crop: row={master_row}, col={master_col}")

    # master RPC → 地理錨點
    ds_master = gdal.Open(str(TRACK3_DIR / f"JAX_068_{MASTER_SENSOR}_RGB.tif"))
    rpc_master = parse_rpc_gdal(ds_master)
    del ds_master
    h_ground = float(rpc_master[8])
    lat_anchor, lon_anchor = rpc_inverse(rpc_master, master_row, master_col, h_ground)
    print(f"Geographic anchor: lat={lat_anchor:.8f}, lon={lon_anchor:.8f}, h={h_ground:.2f} m\n")

    result = {}
    for tif_path in sorted(TRACK3_DIR.glob("JAX_068_*_RGB.tif")):
        sensor = tif_path.stem.split("_")[2]
        if sensor in TRAIN_SENSORS:
            print(f"  skip {sensor} (training sensor)")
            continue

        ds  = gdal.Open(str(tif_path))
        rpc = parse_rpc_gdal(ds)
        del ds

        row_start, col_start = rpc_forward(rpc, lat_anchor, lon_anchor, h_ground)
        row_start = int(round(row_start))
        col_start = int(round(col_start))

        near, far = near_far_from_camera_json(tif_path.stem, rpc, enu_origin)

        result[tif_path.stem] = {
            "col_start": col_start,
            "row_start": row_start,
            "near": round(near, 2),
            "far":  round(far,  2),
        }
        print(f"  {tif_path.stem}: col={col_start}, row={row_start}, near={near:.1f}, far={far:.1f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
