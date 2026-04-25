#!/usr/bin/env python
"""
用已訓練的 Gaussians (.pt) + 現有 camera_pose JSON 渲染新視角。

流程：
  1. 從 camera_pose/JAX_004_cameras/ 讀取 K (full-image) 和 W2C
  2. 從 camera_pose/skewed_camera/ 讀取 K_orig 計算 skew_params
  3. 從 --offsets 讀取每個感測器的 crop offset 和 near/far
  4. 呼叫 decoder.forward(gaussians, C2W, K_full, near, far,
                          (256, 256), skew_params, crop_offsets)
  5. 存成 PNG

Usage (在 mvsplat/ 目錄下):
    conda activate mvsplat && cd mvsplat
    python debug/render_cameras.py \
        --pt outputs/2026-04-19/01-33-49/gaussians_ply/step_011100_full.pt \
        --cameras-dir /project/winston/datasets/DFC2019/overfit_068/camera_pose/JAX_068_cameras \
        --skewed-dir  /project/winston/datasets/DFC2019/overfit_068/camera_pose/skewed_camera \
        --offsets     debug/p0304_crop_offsets.json \
        --output      debug/rendered_p0304
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision

sys.path.insert(0, str(Path(__file__).parent.parent))

from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.model.types import Gaussians


def load_gaussians(pt_path: Path, device: torch.device) -> Gaussians:
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    return Gaussians(
        means=data["means"].unsqueeze(0).to(device),
        covariances=data["covariances"].unsqueeze(0).to(device),
        harmonics=data["harmonics"].unsqueeze(0).to(device),
        opacities=data["opacities"].unsqueeze(0).to(device),
        scales=data["scales"].unsqueeze(0).to(device),
        rotations=data["rotations"].unsqueeze(0).to(device),
    )


def load_camera(cameras_dir: Path, skewed_dir: Path, stem: str, device: torch.device):
    """
    cameras_dir: JAX_004_cameras/ (skew-free, s=0)
    skewed_dir:  skewed_camera/   (original with skew)

    Returns:
        c2w        [4, 4]   camera-to-world
        K_full     [3, 3]   full-image intrinsics (s=0)
        skew_params [4]     (s, fy, cy_full, delta_cx)
    """
    with open(cameras_dir / f"{stem}.json") as f:
        cam_s0 = json.load(f)
    with open(skewed_dir / f"{stem}.json") as f:
        cam_orig = json.load(f)

    K_s0   = torch.tensor(cam_s0["K"],   dtype=torch.float64).reshape(4, 4)
    W2C    = torch.tensor(cam_s0["W2C"], dtype=torch.float64).reshape(4, 4)
    K_orig = torch.tensor(cam_orig["K"], dtype=torch.float64).reshape(4, 4)

    C2W = torch.inverse(W2C).to(torch.float32)

    K_full = torch.tensor([
        [K_s0[0, 0].item(), 0.0,               K_s0[0, 2].item()],
        [0.0,               K_s0[1, 1].item(), K_s0[1, 2].item()],
        [0.0,               0.0,               1.0              ],
    ], dtype=torch.float32)

    # skew_params = (s, fy, cy_full, delta_cx)
    s_val    = float(K_orig[0, 1])
    fy_val   = float(K_orig[1, 1])
    cy_val   = float(K_orig[1, 2])
    delta_cx = float(K_s0[0, 2]) - float(K_orig[0, 2])
    skew_params = torch.tensor([s_val, fy_val, cy_val, delta_cx], dtype=torch.float32)

    return C2W.to(device), K_full.to(device), skew_params.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--cameras-dir", required=True, type=Path,
                        help="camera_pose/JAX_004_cameras/ (skew-free)")
    parser.add_argument("--skewed-dir",  required=True, type=Path,
                        help="camera_pose/skewed_camera/ (original with skew)")
    parser.add_argument("--offsets", required=True, type=Path,
                        help="JSON produced by gen_render_cameras.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256],
                        metavar=("H", "W"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    h, w   = args.image_size

    # ── Decoder ────────────────────────────────────────────────────────────
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from src.config import load_typed_root_config
    from src.model.decoder import get_decoder
    GlobalHydra.instance().clear()
    config_dir = str(Path(__file__).parent.parent / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_dict = compose(config_name="main", overrides=["+experiment=dfc2019"])
    cfg = load_typed_root_config(cfg_dict)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset).to(device).eval()

    # ── Gaussians ──────────────────────────────────────────────────────────
    print(f"Loading Gaussians from {args.pt}")
    gaussians = load_gaussians(args.pt, device)
    print(f"  {gaussians.means.shape[1]:,} Gaussians  scene={torch.load(args.pt, map_location='cpu', weights_only=False).get('scene','?')}")

    # ── Crop offsets ────────────────────────────────────────────────────────
    with open(args.offsets) as f:
        offsets = json.load(f)
    print(f"Rendering {len(offsets)} cameras from {args.offsets.name}")

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Render ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        for i, (stem, off) in enumerate(offsets.items()):
            c2w, K_full, skew_params = load_camera(
                args.cameras_dir, args.skewed_dir, stem, device)

            col_start = float(off["col_start"])
            row_start = float(off["row_start"])
            near_val  = float(off["near"])
            far_val   = float(off["far"])

            # [B=1, V=1, ...]
            extrinsics  = c2w.unsqueeze(0).unsqueeze(0)           # [1,1,4,4]
            intrinsics  = K_full.unsqueeze(0).unsqueeze(0)        # [1,1,3,3]
            skew_p      = skew_params.unsqueeze(0).unsqueeze(0)   # [1,1,4]
            crop_off    = torch.tensor([[col_start, row_start]],
                                       device=device).unsqueeze(0) # [1,1,2]
            near = torch.tensor([[near_val]], device=device)
            far  = torch.tensor([[far_val]],  device=device)

            output = decoder.forward(
                gaussians, extrinsics, intrinsics, near, far, (h, w),
                skew_params=skew_p,
                crop_offsets=crop_off,
            )
            rgb = output.color[0, 0].clamp(0, 1)  # [3, H, W]

            out_path = args.output / f"{stem}.png"
            torchvision.utils.save_image(rgb, out_path)
            print(f"  [{i+1}/{len(offsets)}] {stem}  col={int(col_start)}, row={int(row_start)} → {out_path.name}")

    print(f"\nDone. {len(offsets)} images → {args.output}/")


if __name__ == "__main__":
    main()
