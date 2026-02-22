import os
from osgeo import gdal
import numpy as np


def parse_rpc(ds) -> np.ndarray:
    """Parse RPC metadata from a GDAL dataset into a 90-dim coefficient array.

    Layout matches src.geometry.rpc.RPC:
    0-9:  LINE_OFF, LINE_SCALE, SAMP_OFF, SAMP_SCALE,
          LAT_OFF, LAT_SCALE, LONG_OFF, LONG_SCALE,
          HEIGHT_OFF, HEIGHT_SCALE
    10-29: LINE_NUM_COEFF (20 terms)
    30-49: LINE_DEN_COEFF (20 terms)
    50-69: SAMP_NUM_COEFF (20 terms)
    70-89: SAMP_DEN_COEFF (20 terms)
    """
    keys = [
        "LINE_OFF",
        "LINE_SCALE",
        "SAMP_OFF",
        "SAMP_SCALE",
        "LAT_OFF",
        "LAT_SCALE",
        "LONG_OFF",
        "LONG_SCALE",
        "HEIGHT_OFF",
        "HEIGHT_SCALE",
    ]

    coeffs = []
    for k in keys:
        val = ds.GetMetadataItem(k, "RPC")
        if val is None:
            val = ds.GetMetadata("RPC").get(k, 0)
        coeffs.append(float(val))

    for prefix in [
        "LINE_NUM_COEFF",
        "LINE_DEN_COEFF",
        "SAMP_NUM_COEFF",
        "SAMP_DEN_COEFF",
    ]:
        val_str = ds.GetMetadataItem(prefix, "RPC")
        if val_str is None:
            val_str = ds.GetMetadata("RPC").get(prefix, "")
        vals = [float(x) for x in val_str.split()]
        if len(vals) < 20:
            vals = vals + [0.0] * (20 - len(vals))
        coeffs.extend(vals[:20])

    return np.array(coeffs, dtype=np.float64)


def inspect_rpc_geolocation(tif_paths):
    infos = []

    for path in tif_paths:
        if not os.path.exists(path):
            print(f"File not found, skip: {path}")
            continue

        ds = gdal.Open(path)
        if ds is None:
            print(f"GDAL failed to open: {path}")
            continue

        coeffs = parse_rpc(ds)

        line_off = coeffs[0]
        samp_off = coeffs[2]
        lat_off = coeffs[4]
        long_off = coeffs[6]
        height_off = coeffs[8]

        infos.append(
            {
                "path": path,
                "lat_off": lat_off,
                "long_off": long_off,
                "height_off": height_off,
            }
        )

    if not infos:
        print("No valid RPC info collected.")
        return

    print("=== RPC Geolocation Summary (using LAT_OFF/LONG_OFF) ===")
    base = infos[0]
    print(
        f"Base image: {os.path.basename(base['path'])} | "
        f"LAT_OFF={base['lat_off']:.9f}, LONG_OFF={base['long_off']:.9f}, HEIGHT_OFF={base['height_off']:.3f}"
    )

    for info in infos:
        d_lat_off = info["lat_off"] - base["lat_off"]
        d_lon_off = info["long_off"] - base["long_off"]
        print(
            f"Image: {os.path.basename(info['path'])}\n"
            f"  LAT_OFF={info['lat_off']:.9f}, LONG_OFF={info['long_off']:.9f}, HEIGHT_OFF={info['height_off']:.3f}\n"
            f"  ΔLAT_OFF={d_lat_off:.9e}, ΔLONG_OFF={d_lon_off:.9e}"
        )

    # Simple conclusion: check if all LAT_OFF/LONG_OFF are (almost) identical
    tol_deg = 1e-6  # ~ sub-meter
    same = True
    for info in infos[1:]:
        if abs(info["lat_off"] - base["lat_off"]) > tol_deg or abs(
            info["long_off"] - base["long_off"]
        ) > tol_deg:
            same = False
            break

    if same:
        print("\nConclusion: All images share (practically) the same geolocation.")
    else:
        print("\nConclusion: Images have different geolocations (beyond tolerance).")


def main():
    base_dir = "/project/winston/datasets/DFC2019/overfit/training/JAX_004_p0004"
    tif_names = [
        "JAX_004_006_RGB.tif",
        "JAX_004_009_RGB.tif",
        "JAX_004_012_RGB.tif",
        "JAX_004_014_RGB.tif",
    ]
    tif_paths = [os.path.join(base_dir, n) for n in tif_names]

    inspect_rpc_geolocation(tif_paths)


if __name__ == "__main__":
    main()
