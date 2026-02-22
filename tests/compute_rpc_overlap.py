import os
from dataclasses import dataclass
from typing import List

import numpy as np
from osgeo import gdal


# ---- RPC utilities (copied from geographic_cropping.py, same conventions) ----

def parse_rpc(ds):
    md = ds.GetMetadata('RPC')
    if not md:
        return np.zeros(90, dtype=np.float64)
    keys = [
        'LINE_OFF', 'LINE_SCALE', 'SAMP_OFF', 'SAMP_SCALE',
        'LAT_OFF', 'LAT_SCALE', 'LONG_OFF', 'LONG_SCALE',
        'HEIGHT_OFF', 'HEIGHT_SCALE',
    ]
    coeffs = [float(md.get(k, 0)) for k in keys]
    for prefix in ['LINE_NUM_COEFF', 'LINE_DEN_COEFF', 'SAMP_NUM_COEFF', 'SAMP_DEN_COEFF']:
        val_str = md.get(prefix, "")
        vals = [float(x) for x in val_str.split()]
        coeffs.extend(vals if len(vals) == 20 else vals + [0] * (20 - len(vals)))
    return np.array(coeffs, dtype=np.float64)


def poly(c, x, y, z):
    return (
        c[0] + c[1] * x + c[2] * y + c[3] * z + c[4] * x * y + c[5] * x * z + c[6] * y * z
        + c[7] * x * x + c[8] * y * y + c[9] * z * z
        + c[10] * x * y * z + c[11] * x * x * x + c[12] * x * y * y + c[13] * x * z * z
        + c[14] * x * x * y + c[15] * y * y * y + c[16] * y * z * z
        + c[17] * x * x * z + c[18] * y * y * z + c[19] * z * z * z
    )


def rpc_forward(rpc, lat, lon, height):
    line_off, line_scale, samp_off, samp_scale, lat_off, lat_scale, long_off, long_scale, h_off, h_scale = rpc[:10]
    P = (lat - lat_off) / lat_scale
    L = (lon - long_off) / long_scale
    H = (height - h_off) / h_scale
    row = (poly(rpc[10:30], L, P, H) / poly(rpc[30:50], L, P, H)) * line_scale + line_off
    col = (poly(rpc[50:70], L, P, H) / poly(rpc[70:90], L, P, H)) * samp_scale + samp_off
    return row, col


def rpc_inverse(rpc, row, col, height, iters: int = 10):
    lat, lon = rpc[4], rpc[6]
    for _ in range(iters):
        r_est, c_est = rpc_forward(rpc, lat, lon, height)
        delta = 1e-5
        r_dlat, c_dlat = rpc_forward(rpc, lat + delta, lon, height)
        r_dlon, c_dlon = rpc_forward(rpc, lat, lon + delta, height)
        dr_dlat, dc_dlat = (r_dlat - r_est) / delta, (c_dlat - c_est) / delta
        dr_dlon, dc_dlon = (r_dlon - r_est) / delta, (c_dlon - c_est) / delta
        det = dr_dlat * dc_dlon - dr_dlon * dc_dlat
        if abs(det) < 1e-15:
            break
        dlat = (dc_dlon * (row - r_est) - dr_dlon * (col - c_est)) / det
        dlon = (dr_dlat * (col - c_est) - dc_dlat * (row - r_est)) / det
        lat, lon = lat + dlat, lon + dlon
    return lat, lon


# ---- Footprint + overlap computation ----

@dataclass
class Footprint:
    path: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def lat_center(self) -> float:
        return 0.5 * (self.lat_min + self.lat_max)


R_EARTH = 6378137.0
PI = np.pi


def bbox_area_m2(fp: Footprint) -> float:
    dlat_rad = (fp.lat_max - fp.lat_min) * PI / 180.0
    dlon_rad = (fp.lon_max - fp.lon_min) * PI / 180.0
    lat_mid_rad = fp.lat_center * PI / 180.0
    north = dlat_rad * R_EARTH
    east = dlon_rad * R_EARTH * np.cos(lat_mid_rad)
    area = abs(north * east)
    return float(max(area, 0.0))


def bbox_intersection(fp1: Footprint, fp2: Footprint):
    lat_min = max(fp1.lat_min, fp2.lat_min)
    lat_max = min(fp1.lat_max, fp2.lat_max)
    lon_min = max(fp1.lon_min, fp2.lon_min)
    lon_max = min(fp1.lon_max, fp2.lon_max)
    if lat_max <= lat_min or lon_max <= lon_min:
        return None
    return Footprint(path=f"INTER({os.path.basename(fp1.path)},{os.path.basename(fp2.path)})",
                     lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


def compute_footprint(path: str) -> Footprint:
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Failed to open {path}")

    W, H = ds.RasterXSize, ds.RasterYSize
    rpc = parse_rpc(ds)
    h_ground = rpc[8]  # HEIGHT_OFF

    # Use 4 corner pixels (row, col)
    corners = [
        (0.0, 0.0),          # top-left
        (0.0, W - 1.0),      # top-right
        (H - 1.0, 0.0),      # bottom-left
        (H - 1.0, W - 1.0),  # bottom-right
    ]

    lats = []
    lons = []
    for row, col in corners:
        lat, lon = rpc_inverse(rpc, row, col, h_ground)
        lats.append(lat)
        lons.append(lon)

    lats = np.array(lats, dtype=np.float64)
    lons = np.array(lons, dtype=np.float64)

    return Footprint(
        path=path,
        lat_min=float(lats.min()),
        lat_max=float(lats.max()),
        lon_min=float(lons.min()),
        lon_max=float(lons.max()),
    )


def inspect_overlap(tif_paths: List[str]):
    footprints: List[Footprint] = []
    for p in tif_paths:
        if not os.path.exists(p):
            print(f"File not found, skip: {p}")
            continue
        fp = compute_footprint(p)
        footprints.append(fp)

    if not footprints:
        print("No valid footprints.")
        return

    print("=== RPC Footprint (approx) ===")
    for fp in footprints:
        print(
            f"{os.path.basename(fp.path)}:\n"
            f"  lat [{fp.lat_min:.9f}, {fp.lat_max:.9f}]\n"
            f"  lon [{fp.lon_min:.9f}, {fp.lon_max:.9f}]\n"
            f"  area ≈ {bbox_area_m2(fp):.2f} m^2"
        )

    base = footprints[0]
    base_area = bbox_area_m2(base)
    print("\n=== Overlap vs base (" + os.path.basename(base.path) + ") ===")
    for fp in footprints[1:]:
        inter = bbox_intersection(base, fp)
        if inter is None:
            print(f"{os.path.basename(fp.path)}: no overlap with base")
            continue
        inter_area = bbox_area_m2(inter)
        ratio = inter_area / base_area if base_area > 0 else 0.0
        print(
            f"{os.path.basename(fp.path)}:\n"
            f"  overlap area ≈ {inter_area:.2f} m^2\n"
            f"  overlap ratio (intersection / base) ≈ {ratio * 100.0:.2f}%"
        )

    # Intersection of all footprints (optional)
    inter_all = footprints[0]
    for fp in footprints[1:]:
        tmp = bbox_intersection(inter_all, fp)
        if tmp is None:
            inter_all = None
            break
        inter_all = tmp

    print("\n=== Intersection of all images ===")
    if inter_all is None:
        print("No common intersection among all images.")
    else:
        inter_all_area = bbox_area_m2(inter_all)
        ratio_all = inter_all_area / base_area if base_area > 0 else 0.0
        print(
            f"Common area ≈ {inter_all_area:.2f} m^2\n"
            f"Common / base ratio ≈ {ratio_all * 100.0:.2f}%"
        )


def main():
    base_dir = "/project/winston/datasets/DFC2019/overfit/training/JAX_004_p0004"
    tif_names = [
        "JAX_004_006_RGB.tif",
        "JAX_004_009_RGB.tif",
        "JAX_004_012_RGB.tif",
        "JAX_004_014_RGB.tif",
    ]
    tif_paths = [os.path.join(base_dir, n) for n in tif_names]
    inspect_overlap(tif_paths)


if __name__ == "__main__":
    main()
