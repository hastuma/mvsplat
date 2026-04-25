"""
Generate a DSM (Digital Surface Model) GeoTIFF from a DFC2019 Track4 PC3 file.

The PC3 format is: X, Y, Z, Intensity, ReturnNumber  (comma-separated)
X and Y are assumed to be in the same ENU local frame as SkySplat
(East and North offsets in metres from the ENU origin).
Z is height above the PC3 reference (scene-relative metres).

Output: a single-band float32 GeoTIFF where each pixel = max Z of all
LiDAR points that fall within that grid cell (surface DSM).

Usage:
    python tools/generate_dsm_from_pc3.py \
        --input  /project/winston/datasets/DFC2019/Track4/JAX_004_PC3.txt \
        --output /project/winston/datasets/DFC2019/dsm_gt/JAX_004_DSM.tif \
        --resolution 0.5          # metres per pixel (default 0.5 = GSD of RGB)
        [--x_min -400]            # optional bounding box override (metres)
        [--x_max  400]
        [--y_min -400]
        [--y_max  400]

Dependencies:  numpy, (optional) osgeo.gdal for GeoTIFF output.
               Falls back to saving a plain numpy .npy file if GDAL unavailable.
"""

import argparse
from pathlib import Path

import numpy as np


def load_pc3(path: str):
    print(f"Loading {path} ...")
    data = np.loadtxt(path, delimiter=",", usecols=(0, 1, 2), dtype=np.float32)
    print(f"  {len(data):,} points  X∈[{data[:,0].min():.1f}, {data[:,0].max():.1f}]"
          f"  Y∈[{data[:,1].min():.1f}, {data[:,1].max():.1f}]"
          f"  Z∈[{data[:,2].min():.2f}, {data[:,2].max():.2f}]")
    return data


def rasterize_max_z(xyz: np.ndarray, resolution: float,
                    x_min, x_max, y_min, y_max) -> np.ndarray:
    """
    Rasterize point cloud onto regular grid by taking max Z per cell.
    Returns DSM array [rows, cols], NaN where no points fall.
    """
    cols = int(np.ceil((x_max - x_min) / resolution))
    rows = int(np.ceil((y_max - y_min) / resolution))
    print(f"  Grid: {rows} rows × {cols} cols  ({resolution} m/px)")

    # Initialize with -inf so np.maximum.at works correctly; unvisited cells → NaN at end
    dsm = np.full((rows, cols), -np.inf, dtype=np.float64)

    x, y, z = xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64), xyz[:, 2].astype(np.float64)

    # Filter to bounding box
    mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
    x, y, z = x[mask], y[mask], z[mask]
    print(f"  {mask.sum():,} points inside bounding box")

    # Map to pixel indices
    col_idx = ((x - x_min) / resolution).astype(np.int64)
    # Y-axis: larger Y = smaller row index (north-up)
    row_idx = ((y_max - y) / resolution).astype(np.int64)

    # Clamp (safety)
    col_idx = np.clip(col_idx, 0, cols - 1)
    row_idx = np.clip(row_idx, 0, rows - 1)

    # Scatter-max: take max Z per cell (surface DSM)
    idx_flat = row_idx * cols + col_idx
    np.maximum.at(dsm.ravel(), idx_flat, z)

    # Replace unvisited cells (-inf) with NaN
    dsm[dsm == -np.inf] = np.nan
    dsm = dsm.astype(np.float32)

    valid = np.isfinite(dsm)
    print(f"  Valid cells: {valid.sum():,} / {rows * cols:,}"
          f"  ({100 * valid.mean():.1f}%)")
    print(f"  DSM Z range: {dsm[valid].min():.2f} to {dsm[valid].max():.2f} m")
    return dsm


def save_geotiff(dsm: np.ndarray, output_path: str,
                 x_min: float, y_max: float, resolution: float):
    """Save DSM as GeoTIFF. Falls back to .npy if GDAL is not installed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from osgeo import gdal, osr
        driver = gdal.GetDriverByName("GTiff")
        rows, cols = dsm.shape
        ds = driver.Create(str(output_path), cols, rows, 1, gdal.GDT_Float32,
                           options=["COMPRESS=LZW", "TILED=YES"])
        # Geotransform: (x_origin, pixel_width, 0, y_origin, 0, -pixel_height)
        # x_origin = x_min, y_origin = y_max (top-left corner)
        ds.SetGeoTransform([x_min, resolution, 0, y_max, 0, -resolution])
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(np.nan)
        band.WriteArray(dsm)
        ds.FlushCache()
        ds = None
        print(f"  Saved GeoTIFF → {output_path}")
    except ImportError:
        npy_path = output_path.with_suffix(".npy")
        np.save(npy_path, dsm)
        print(f"  GDAL not available. Saved numpy array → {npy_path}")
        print(f"  Metadata: x_min={x_min}, y_max={y_max}, resolution={resolution}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--resolution", type=float, default=0.5,
                        help="Metres per pixel (default 0.5 to match RGB GSD)")
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    args = parser.parse_args()

    xyz = load_pc3(args.input)

    x_min = args.x_min if args.x_min is not None else float(xyz[:, 0].min())
    x_max = args.x_max if args.x_max is not None else float(xyz[:, 0].max())
    y_min = args.y_min if args.y_min is not None else float(xyz[:, 1].min())
    y_max = args.y_max if args.y_max is not None else float(xyz[:, 1].max())

    print(f"\nRasterizing to {args.resolution} m/px grid ...")
    dsm = rasterize_max_z(xyz, args.resolution, x_min, x_max, y_min, y_max)

    print(f"\nSaving ...")
    save_geotiff(dsm, args.output, x_min, y_max, args.resolution)


if __name__ == "__main__":
    main()
