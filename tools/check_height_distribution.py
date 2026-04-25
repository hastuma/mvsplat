"""
Level-1 height sanity check for SkySplat.

Compares the Z distribution of predicted Gaussian positions against
ground-truth LiDAR points from a DFC2019 Track4 PC3 file.

Usage:
    python tools/check_height_distribution.py \
        --gaussian_pt  outputs/2026-04-18/00-43-55/gaussians_ply/step_000350_full.pt \
        --pc3          /project/winston/datasets/DFC2019/Track4/JAX_004_PC3.txt \
        --enu_origin   /project/winston/datasets/DFC2019/overfit/enu_origin/JAX_004.json \
        [--out_png     tools/height_check.png]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_pc3(path: str) -> np.ndarray:
    """Load Track4 PC3 file.  Columns: X, Y, Z, Intensity, ReturnNum."""
    return np.loadtxt(path, delimiter=",", usecols=(0, 1, 2))


def load_gaussians(pt_path: str, enu_origin_h: float):
    """
    Load saved Gaussian .pt file and return:
      - enu_z:  ENU-Up component (metres above ENU origin)
      - abs_alt: absolute WGS84 ellipsoidal altitude (enu_origin_h + enu_z)
    """
    data = torch.load(pt_path, map_location="cpu")
    means = data["means"]           # [N, 3]  ENU (East, North, Up)
    enu_z = means[:, 2].numpy()
    abs_alt = enu_origin_h + enu_z
    scene = data.get("scene", "unknown")
    step  = data.get("step", -1)
    return enu_z, abs_alt, scene, step


def print_stats(label: str, arr: np.ndarray):
    print(f"  {label:30s}  min={arr.min():8.2f}  max={arr.max():8.2f}"
          f"  mean={arr.mean():8.2f}  std={arr.std():6.2f}  n={len(arr):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian_pt",  required=True)
    parser.add_argument("--pc3",          required=True)
    parser.add_argument("--enu_origin",   required=True,
                        help="JSON file: [lat, lon, h_wgs84]")
    parser.add_argument("--out_png",      default=None,
                        help="Optional: save histogram plot to this path")
    args = parser.parse_args()

    # ── Load ENU origin ──────────────────────────────────────────────────────
    with open(args.enu_origin) as f:
        enu_data = json.load(f)
    enu_h = float(enu_data[2])        # WGS84 ellipsoidal height of ENU origin
    print(f"\nENU origin height (WGS84):  {enu_h:.3f} m")

    # ── Load Gaussians ────────────────────────────────────────────────────────
    enu_z, abs_alt, scene, step = load_gaussians(args.gaussian_pt, enu_h)
    print(f"Gaussian file: {Path(args.gaussian_pt).name}  |  scene={scene}  step={step}")

    # ── Load PC3 ground truth ─────────────────────────────────────────────────
    pts = load_pc3(args.pc3)
    pc3_z = pts[:, 2]
    pc3_x = pts[:, 0]
    pc3_y = pts[:, 1]
    print(f"PC3 file: {Path(args.pc3).name}  |  {len(pc3_z):,} points")

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n── Height Statistics ─────────────────────────────────────────────")
    print_stats("Gaussian ENU-Z (above origin)", enu_z)
    print_stats("Gaussian absolute alt (WGS84)", abs_alt)
    print_stats("PC3 Z (scene-relative metres)", pc3_z)
    print_stats("PC3 X", pc3_x)
    print_stats("PC3 Y", pc3_y)

    # ── Interpretation ────────────────────────────────────────────────────────
    print("\n── Interpretation ────────────────────────────────────────────────")
    # If the PC3 coordinate origin matches the ENU origin, then:
    #   PC3_Z  ≈  Gaussian ENU-Z   (both heights above the same reference)
    # Check overlap:
    g_min, g_max = enu_z.min(), enu_z.max()
    p_min, p_max = pc3_z.min(), pc3_z.max()
    overlap_min = max(g_min, p_min)
    overlap_max = min(g_max, p_max)
    if overlap_min < overlap_max:
        print(f"  ✓ ENU-Z and PC3-Z ranges OVERLAP: [{overlap_min:.1f}, {overlap_max:.1f}] m")
    else:
        gap = overlap_min - overlap_max
        print(f"  ✗ ENU-Z and PC3-Z ranges DO NOT OVERLAP (gap = {gap:.1f} m)")
        print(f"    ENU-Z: [{g_min:.1f}, {g_max:.1f}]   PC3-Z: [{p_min:.1f}, {p_max:.1f}]")
        print(f"    → Likely a coordinate offset. Check if PC3 origin ≠ ENU origin.")

    mean_diff = enu_z.mean() - pc3_z.mean()
    print(f"  Mean(ENU-Z) - Mean(PC3-Z) = {mean_diff:+.2f} m")
    print(f"  If this is close to 0 → same height reference. "
          f"If close to {-enu_h:.1f} m → PC3 is in absolute WGS84.")

    # ── Optional plot ─────────────────────────────────────────────────────────
    if args.out_png:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].hist(enu_z, bins=100, color="steelblue", alpha=0.7, label="Gaussian ENU-Z")
            axes[0].hist(pc3_z, bins=100, color="orange", alpha=0.7, label="PC3-Z (GT)")
            axes[0].set_xlabel("Height above ENU origin (m)")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Height Distribution Comparison")
            axes[0].legend()

            axes[1].hist(abs_alt, bins=100, color="steelblue", alpha=0.7, label="Gaussian abs alt (WGS84)")
            axes[1].axvline(enu_h, color="red", linestyle="--", label=f"ENU origin h = {enu_h:.1f} m")
            axes[1].set_xlabel("WGS84 Ellipsoidal Altitude (m)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Absolute Altitude (Gaussian only)")
            axes[1].legend()

            plt.tight_layout()
            Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.out_png, dpi=150)
            print(f"\nHistogram saved → {args.out_png}")
        except ImportError:
            print("\n[matplotlib not available, skipping plot]")


if __name__ == "__main__":
    main()
