from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json

from ..geometry.rpc import RPC
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    damv2_encoder: str = "vitl"
    damv2_checkpoint: str = ""   # empty = disabled
    damv2_observe_every_n_steps: int = 10
    damv2_loss_weight: float = 0.0       # > 0 enables Pearson loss
    damv2_loss_warmup_steps: int = 0     # steps before DAMV2 loss kicks in


def _pearson_corr(x: Tensor, y: Tensor) -> Tensor:
    """Pearson correlation coefficient between two spatial maps.

    Args:
        x: [B, H, W]
        y: [B, H, W]
    Returns:
        corr: [B]  values in [-1, 1]
    """
    B = x.shape[0]
    x = x.reshape(B, -1).float()   # [B, N]
    y = y.reshape(B, -1).float()
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    corr = (x * y).sum(dim=1) / (
        x.norm(dim=1) * y.norm(dim=1) + 1e-8
    )
    return corr  # [B]


@torch.no_grad()
def _damv2_observe(
    damv2,
    batch: dict,
    vis_dump: dict,
    output_dir,
    global_step: int,
) -> None:
    """Compute and log Pearson correlation between cost-volume depth and DAMV2
    depth for each context view.  Observation only — no gradient flows here.

    Sign convention:
        cost_volume depth  = camera-to-surface distance (metres)
                           = larger for LOW terrain, smaller for tall buildings
        DAMV2 depth        = relative depth (larger = farther from camera)
                           = same direction as cost_volume distance

    Expected result: positive Pearson correlation.
    If the logged 'depth/pearson_avg' is consistently negative, the sign
    convention differs and the loss term should use -(corr) instead of corr.
    """
    import torchvision

    device = batch["context"]["image"].device
    ctx_imgs = batch["context"]["image"]          # [B, V, 3, H, W]
    B, V, _, H, W = ctx_imgs.shape

    # cost_vol_depth: [B*V, 1, H, W], ordering is (b0v0, b0v1, ..., b0vV, b1v0, ...)
    # i.e. batch-major → reshape to [B, V, H, W]
    depth_hires = vis_dump["depth_highres"].detach().squeeze(1)  # [B*V, H, W]
    depth_hires = depth_hires.reshape(B, V, H, W)                # [B, V, H, W]

    corrs_per_view = []
    vis_rows = []  # for saving a comparison image

    for v in range(V):
        ctx_v = ctx_imgs[:, v]          # [B, 3, H, W]  values in [0,1]
        cv_depth_v = depth_hires[:, v]  # [B, H, W]  cost-volume depth (metres)

        damv2_depth_v = damv2(ctx_v.to(device))   # [B, H, W]  relative depth

        corr_v = _pearson_corr(cv_depth_v, damv2_depth_v)   # [B]
        corrs_per_view.append(corr_v)

        # Log per-view scalar to WandB
        try:
            wandb.log(
                {f"depth/pearson_view{v}": corr_v.mean().item()},
                step=global_step,
            )
        except Exception:
            pass

        # Build one row of the visualization grid: [ctx_rgb | cv_depth | da_depth]
        if B > 0:
            def _norm_to_01(t):
                t = t[0].cpu()   # [H, W]
                mn, mx = t.min(), t.max()
                return ((t - mn) / (mx - mn + 1e-8)).unsqueeze(0).expand(3, -1, -1)

            row = torch.cat([
                ctx_v[0].cpu(),           # RGB image
                _norm_to_01(cv_depth_v),  # cost-volume depth (greyscale as RGB)
                _norm_to_01(damv2_depth_v),  # DAMV2 depth
            ], dim=-1)   # [3, H, 3W]
            vis_rows.append(row)

    # Average Pearson across all views
    corrs_all = torch.stack(corrs_per_view, dim=0)   # [V, B]
    pearson_avg = corrs_all.mean().item()

    try:
        wandb.log({"depth/pearson_avg": pearson_avg}, step=global_step)
    except Exception:
        pass

    print(
        f"[DAMV2 observe] step={global_step}  "
        f"pearson/view={[f'{c.mean().item():.3f}' for c in corrs_per_view]}  "
        f"avg={pearson_avg:.3f}"
    )

    # Save side-by-side visualization: one image per context view
    # Each image: [RGB | cv_depth | da_depth] horizontally concatenated
    if vis_rows:
        vis_damv2_dir = output_dir / "vis_damv2"
        vis_damv2_dir.mkdir(parents=True, exist_ok=True)
        for vi, row in enumerate(vis_rows):
            torchvision.utils.save_image(
                row,
                vis_damv2_dir / f"step_{global_step:0>6}_ctx{vi}.png",
            )


def _damv2_pearson_loss(
    damv2,
    ctx_imgs: Tensor,
    cv_depth_map: Tensor,
) -> Tensor:
    """Differentiable Pearson correlation loss between cost-volume depth and DAMV2.

    DAMV2 output is always detached (FrozenDAMV2 uses @torch.no_grad internally),
    so gradients flow only through cv_depth_map back to the encoder.

    Args:
        ctx_imgs:     [B, V, 3, H, W]  context RGB, values in [0, 1]
        cv_depth_map: [B, V, H, W]     cost-volume depth with gradient
    Returns:
        loss: scalar  -mean(pearson across views and batch)
    """
    B, V, _, _, _ = ctx_imgs.shape
    corrs = []
    for v in range(V):
        damv2_v = damv2(ctx_imgs[:, v])              # [B, H, W]  detached
        corrs.append(_pearson_corr(cv_depth_map[:, v], damv2_v))  # [B]
    return -torch.stack(corrs).mean()  # minimize = maximize pearson


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    output_dir: Path

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        output_dir: Path,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.losses = nn.ModuleList(losses)
        self.step_tracker = step_tracker
        self.output_dir = output_dir
        self.data_shim = get_data_shim(self.encoder)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        # Frozen DAMV2 for depth correlation observation.
        self.damv2 = None
        if train_cfg.damv2_checkpoint:
            from .encoder.damv2_wrapper import FrozenDAMV2
            self.damv2 = FrozenDAMV2(train_cfg.damv2_encoder, train_cfg.damv2_checkpoint)
            print(f"[SkySplat] Loaded frozen DAMV2 ({train_cfg.damv2_encoder}) from {train_cfg.damv2_checkpoint}")

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def _load_json_cameras(self, batch_views: dict, device) -> tuple:
        """Load JSON cameras for a batch of views (context or target).

        Returns:
            c2ws:      [B*V, 4, 4]  camera-to-world in ENU
            K_patches: [B*V, 3, 3]  intrinsics with patch-offset-corrected principal point
            cam_z:     [B*V]        c2w[2,3] = camera ENU-Z (height above ENU origin)
        """
        image_names = batch_views["image_name"]   # list[V] of list[B]
        col_start_list = batch_views.get("col_start", None)
        row_start_list = batch_views.get("row_start", None)

        B = len(image_names[0])
        V = len(image_names)

        dataset_root = Path(get_cfg()["dataset"]["roots"][0])

        c2ws, K_patches, cam_zs = [], [], []
        for b_idx in range(B):
            for v_idx in range(V):
                img_name = image_names[v_idx][b_idx]
                scene_prefix = "_".join(str(img_name).split("_")[:2])
                cam_json = dataset_root / "camera_pose" / f"{scene_prefix}_cameras" / f"{img_name}.json"
                with open(cam_json, "r") as f:
                    cam_data = json.load(f)

                # 用 float64 讀取以確保矩陣逆運算精度，最後轉回 float32
                K_full = torch.tensor(cam_data["K"], device=device, dtype=torch.float64).reshape(4, 4)
                w2c    = torch.tensor(cam_data["W2C"], device=device, dtype=torch.float64).reshape(4, 4)

                col_start = col_start_list[v_idx][b_idx].item()
                row_start = row_start_list[v_idx][b_idx].item()

                K_patch = torch.tensor([
                    [K_full[0, 0].item(), 0.0,                              K_full[0, 2].item() ],
                    [0.0,                 K_full[1, 1].item(),              K_full[1, 2].item() ],
                    [0.0,                 0.0,                              1.0],
                ], device=device, dtype=torch.float32)

                c2w = torch.inverse(w2c).to(torch.float32)
                c2ws.append(c2w)
                K_patches.append(K_patch)
                cam_zs.append(c2w[2, 3])

        return (
            torch.stack(c2ws),       # [B*V, 4, 4] float32
            torch.stack(K_patches),  # [B*V, 3, 3] float32
            torch.stack(cam_zs),     # [B*V]        float32
        )

    def _load_target_cameras_full(self, batch_views: dict, device) -> tuple:
        """Load JSON cameras for target views with full-image K (no crop adjustment).

        Loads both K_s0 (skew=0, full image) and K_orig (original with skew) to compute
        the inverse skew warp parameters needed by render_cuda.

        Returns:
            c2ws:         [B*V, 4, 4]  camera-to-world (float32)
            K_fulls:      [B*V, 3, 3]  full-image s=0 intrinsics, no crop offset (float32)
            skew_params:  [B*V, 4]     (s, fy, cy, delta_cx) per camera (float32)
            crop_offsets: [B*V, 2]     (col_start, row_start) in pixels (float32)
            cam_zs:       [B*V]        c2w[2,3] camera ENU-Z height (float32)
        """
        image_names  = batch_views["image_name"]
        col_start_list = batch_views.get("col_start", None)
        row_start_list = batch_views.get("row_start", None)

        px_list = batch_views.get("px", None)
        py_list = batch_views.get("py", None)

        B = len(image_names[0])
        V = len(image_names)

        dataset_root = Path(get_cfg()["dataset"]["roots"][0])
        c2ws, K_fulls, skew_params_list, crop_offsets_list, cam_zs = [], [], [], [], []
        for b_idx in range(B):
            for v_idx in range(V):
                img_name = image_names[v_idx][b_idx]
                scene_prefix = "_".join(str(img_name).split("_")[:2])

                # s=0 camera (skew already removed by skew_correct.py)
                cam_json_s0 = dataset_root / "camera_pose" / f"{scene_prefix}_cameras" / f"{img_name}.json"
                with open(cam_json_s0, "r") as f:
                    cam_data_s0 = json.load(f)
                K_s0 = torch.tensor(cam_data_s0["K"], device=device, dtype=torch.float64).reshape(4, 4)
                w2c  = torch.tensor(cam_data_s0["W2C"], device=device, dtype=torch.float64).reshape(4, 4)

                # original camera (skew ≠ 0, needed for inverse warp params)
                cam_json_orig = dataset_root / "camera_pose" / "skewed_camera" / f"{img_name}.json"
                with open(cam_json_orig, "r") as f:
                    cam_data_orig = json.load(f)
                K_orig = torch.tensor(cam_data_orig["K"], device=device, dtype=torch.float64).reshape(4, 4)

                # Full-image s=0 intrinsics (3×3, no crop adjustment)
                K_full_3x3 = torch.tensor([
                    [K_s0[0, 0].item(), 0.0,               K_s0[0, 2].item()],
                    [0.0,               K_s0[1, 1].item(), K_s0[1, 2].item()],
                    [0.0,               0.0,               1.0              ],
                ], device=device, dtype=torch.float32)

                # Skew params for inverse warp: (s, fy, cy_full, delta_cx)
                s_val    = float(K_orig[0, 1])
                fy_val   = float(K_orig[1, 1])
                cy_val   = float(K_orig[1, 2])
                cx_orig  = float(K_orig[0, 2])
                cx_s0    = float(K_s0[0, 2])
                delta_cx = cx_s0 - cx_orig
                skew_param = torch.tensor(
                    [s_val, fy_val, cy_val, delta_cx], device=device, dtype=torch.float32
                )

                # Crop offset (in full-image pixel coordinates)
                if col_start_list is not None and row_start_list is not None:
                    col_s = col_start_list[v_idx][b_idx].item()
                    row_s = row_start_list[v_idx][b_idx].item()
                elif px_list is not None and py_list is not None:
                    col_s = px_list[v_idx][b_idx].item() * 256
                    row_s = py_list[v_idx][b_idx].item() * 256
                else:
                    col_s, row_s = 0, 0
                crop_offset = torch.tensor([col_s, row_s], device=device, dtype=torch.float32)

                c2w = torch.inverse(w2c).to(torch.float32)
                c2ws.append(c2w)
                K_fulls.append(K_full_3x3)
                skew_params_list.append(skew_param)
                crop_offsets_list.append(crop_offset)
                cam_zs.append(c2w[2, 3])

        return (
            torch.stack(c2ws),               # [B*V, 4, 4]
            torch.stack(K_fulls),            # [B*V, 3, 3]
            torch.stack(skew_params_list),   # [B*V, 4]
            torch.stack(crop_offsets_list),  # [B*V, 2]
            torch.stack(cam_zs),             # [B*V]
        )

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        # Ensure Target Extrinsics match the RPC coordinate system defined by Context
        if "rpc" in batch["context"] and "rpc" in batch["target"]:
            b_tgt, v_tgt, _ = batch["target"]["rpc"].shape
            device = batch["target"]["rpc"].device

            # Load full-image K_s0 (no crop adjustment) + skew params + crop offsets
            c2w_tgt, K_tgt, skew_params_tgt, crop_offsets_tgt, distance_tgt_flat = \
                self._load_target_cameras_full(batch["target"], device)

            batch["target"]["extrinsics"]   = rearrange(c2w_tgt,          "(b v) i j -> b v i j", b=b_tgt, v=v_tgt)
            batch["target"]["intrinsics"]   = rearrange(K_tgt,            "(b v) i j -> b v i j", b=b_tgt, v=v_tgt)
            batch["target"]["skew_params"]  = rearrange(skew_params_tgt,  "(b v) d   -> b v d",   b=b_tgt, v=v_tgt)
            batch["target"]["crop_offsets"] = rearrange(crop_offsets_tgt, "(b v) d   -> b v d",   b=b_tgt, v=v_tgt)
            distance_tgt = rearrange(distance_tgt_flat, "(b v) -> b v", b=b_tgt, v=v_tgt)

            # Load crop-adjusted K_s0 for context views (encoder uses 256×256 patches)
            b_ctx, v_ctx, _ = batch["context"]["rpc"].shape
            # c2w_ctx, K_ctx, skew_params_ctx, crop_offsets_ctx, _ = self._load_json_cameras(batch["context"], device)
            c2w_ctx, K_ctx, skew_params_ctx, crop_offsets_ctx, _ = self._load_target_cameras_full(batch["context"], device)
            batch["context"]["extrinsics"] = rearrange(c2w_ctx, "(b v) i j -> b v i j", b=b_ctx, v=v_ctx)
            batch["context"]["intrinsics"] = rearrange(K_ctx,   "(b v) i j -> b v i j", b=b_ctx, v=v_ctx)
            batch["context"]["skew_params"]  = rearrange(skew_params_ctx,  "(b v) d   -> b v d",   b=b_ctx, v=v_ctx)
            batch["context"]["crop_offsets"] = rearrange(crop_offsets_ctx, "(b v) d   -> b v d",   b=b_ctx, v=v_ctx)

        gaussians, vis_dump = self.encoder(batch["context"], self.global_step, False, scene_names=batch["scene"])

        if "rpc" in batch["context"] and "rpc" in batch["target"]:
            # camera MSL = h_enu_origin + c2w[2,3]
            # near/far = distance from camera to altitude planes at h_off ± 20m
            h_off_tgt = batch["target"]["rpc"][:, :, 8].to(torch.float32)  # HEIGHT_OFF [B, V]
            h_enu_tgt = batch["target"]["enu_origin"][:, 0, 2].to(
                device=h_off_tgt.device, dtype=torch.float32)               # ENU origin MSL [B]
            cam_msl_tgt = h_enu_tgt.unsqueeze(1) + distance_tgt.to(torch.float32)  # [B, V]
            dist_eff_tgt = cam_msl_tgt - h_off_tgt                         # effective distance [B, V]
            scene_range = 20.0
            render_near = (dist_eff_tgt - scene_range).clamp(min=1.0)
            render_far  = (dist_eff_tgt + scene_range).clamp(min=render_near + 1.0)
        else:
            render_near = batch["target"]["near"]
            render_far = batch["target"]["far"]
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            render_near,
            render_far,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
            skew_params=batch["target"].get("skew_params"),
            crop_offsets=batch["target"].get("crop_offsets"),
        )
        target_gt = batch["target"]["image"]

        # --- DEBUG: Save Rendered Image, Depth Maps and Log Stats ---
        if self.global_rank == 0 and self.global_step % 10 == 0:
            import torchvision
            from pathlib import Path
            import torch.nn.functional as F
            
            # 1. Setup Directories
            vis_loss_dir = self.output_dir / "vis_loss"
            vis_depth_dir = self.output_dir / "vis_depth"
            vis_feat_dir = self.output_dir / "vis_feature"
            vis_loss_dir.mkdir(parents=True, exist_ok=True)
            vis_depth_dir.mkdir(parents=True, exist_ok=True)
            vis_feat_dir.mkdir(parents=True, exist_ok=True)

            # Render Context (Training) Views and Save Comparison (every 100 steps)
            if self.global_step % 100 == 0 and "image_name" in batch["context"] and "rpc" in batch["context"]:
                vis_ctx_dir = self.output_dir / "vis_context"
                vis_ctx_dir.mkdir(parents=True, exist_ok=True)

                ctx_device = batch["context"]["rpc"].device
                b_ctx_r, v_ctx_r, _ = batch["context"]["rpc"].shape
                ctx_name_r = batch["context"]["image_name"]
                ctx_px_r = batch["context"]["px"]
                ctx_py_r = batch["context"]["py"]

                K_ctx_list, c2w_ctx_list, dist_ctx_list = [], [], []
                for b_i in range(b_ctx_r):
                    for v_i in range(v_ctx_r):
                        img_name_c = ctx_name_r[v_i][b_i]
                        scene_parts_c = str(img_name_c).split("_")
                        scene_prefix_c = "_".join(scene_parts_c[:2])
                        cam_json_c = Path(get_cfg()["dataset"]["roots"][0]) / "camera_pose" / f"{scene_prefix_c}_cameras" / f"{img_name_c}.json"
                        with open(cam_json_c, "r") as fc:
                            cam_data_c = json.load(fc)

                        K_full_c = torch.tensor(cam_data_c["K"], device=ctx_device, dtype=torch.float64).reshape(4, 4)
                        w2c_c = torch.tensor(cam_data_c["W2C"], device=ctx_device, dtype=torch.float64).reshape(4, 4)

                        cx_global = K_full_c[0, 2]
                        cy_global = K_full_c[1, 2]
                        K_patch_c = torch.tensor([
                            [K_full_c[0, 0], 0.0,             cx_global],
                            [0.0,            K_full_c[1, 1],  cy_global],
                            [0.0,            0.0,             1.0 ],
                        ], device=ctx_device, dtype=torch.float64)
                        
                        K_ctx_list.append(K_patch_c)
                        c2w_ctx_list.append(torch.inverse(w2c_c))
                        dist_ctx_list.append(w2c_c[2, 3].abs())

                K_ctx_t = torch.stack(K_ctx_list)
                c2w_ctx_t = torch.stack(c2w_ctx_list)
                dist_ctx_t = torch.stack(dist_ctx_list)

                ctx_extr = rearrange(c2w_ctx_t, "(b v) i j -> b v i j", b=b_ctx_r, v=v_ctx_r)
                ctx_intr = rearrange(K_ctx_t, "(b v) i j -> b v i j", b=b_ctx_r, v=v_ctx_r)
                dist_ctx = rearrange(dist_ctx_t, "(b v) -> b v", b=b_ctx_r, v=v_ctx_r)

                h_off_ctx = batch["context"]["rpc"][:, :, 8]
                ctx_near = (h_off_ctx + dist_ctx - (h_off_ctx + 50000.0)).clamp(min=1.0)
                ctx_far  = (h_off_ctx + dist_ctx - (h_off_ctx - 50000.0)).clamp(min=ctx_near + 1.0)

                with torch.no_grad():
                    output_ctx = self.decoder.forward(
                        gaussians,
                        ctx_extr,
                        ctx_intr,
                        ctx_near,
                        ctx_far,
                        (h, w),
                        depth_mode=None,
                        # there should be the following skew_param and crop_offsets to send in this function too 
                        skew_params=batch["context"].get("skew_params"),
                        crop_offsets=batch["context"].get("crop_offsets"),
                    )

                context_gt = batch["context"]["image"]
                for vi in range(v_ctx_r):
                    ctx_compare = torch.cat([context_gt[0, vi], output_ctx.color[0, vi]], dim=-1)
                    torchvision.utils.save_image(ctx_compare, vis_ctx_dir / f"step_{self.global_step:0>6}_ctx{vi}.png")
            # 2. Save RGB comparison
            img_loss = torch.cat([target_gt[0, 0], output.color[0, 0]], dim=-1)
            
            
            torchvision.utils.save_image(img_loss, vis_loss_dir / f"step_{self.global_step:0>6}.png")
            
            # 3. Save Predict Height Maps (Concatenate Low-res and High-res)
            # Both are [Batch*View, 1, H, W]. Take the first view of the first batch.
            d_low = vis_dump["depth_lowres"][0, 0]   # [H/4, W/4]
            d_high = vis_dump["depth_highres"][0, 0] # [H, W]
            
            # Normalize for visualization [min, max] -> [0, 1]
            d_min, d_max = d_high.min(), d_high.max()
            if d_max > d_min:
                d_low_norm = (d_low - d_min) / (d_max - d_min)
                d_high_norm = (d_high - d_min) / (d_max - d_min)
            else:
                d_low_norm, d_high_norm = d_low * 0, d_high * 0
                
            # Upscale low-res for concatenation
            d_low_up = F.interpolate(d_low_norm.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze()
            depth_cat = torch.cat([d_low_up, d_high_norm], dim=-1) # [H, 2W]
            torchvision.utils.save_image(depth_cat, vis_depth_dir / f"step_{self.global_step:0>6}.png")
            
            # 3.5 Save CNN Features (first 3 channels as RGB)
            if "cnn_features" in vis_dump:
                feat = vis_dump["cnn_features"][0, 0, :3].detach().cpu()
                # Normalize feature for vis
                f_min, f_max = feat.min(), feat.max()
                feat = (feat - f_min) / (f_max - f_min + 1e-8)
                torchvision.utils.save_image(feat, vis_feat_dir / f"step_{self.global_step:0>6}.png")


        # Compute metrics.
        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        
        # Opacity regularization: 防止所有 Gaussian 變透明（opacity → 0）
        opacity_mean = gaussians.opacities.mean()
        opacity_reg_weight = 0.1
        opacity_target = 0.3
        opacity_reg = opacity_reg_weight * torch.relu(opacity_target - opacity_mean)
        total_loss = total_loss + opacity_reg

        # ---------- DAMV2 Pearson Loss ----------
        if (
            self.damv2 is not None
            and self.train_cfg.damv2_loss_weight > 0
            and self.global_step >= self.train_cfg.damv2_loss_warmup_steps
            and "depth_highres" in vis_dump
        ):
            B_img, V_img, _, H_img, W_img = batch["context"]["image"].shape
            depth_hires = vis_dump["depth_highres"].squeeze(1)              # [B*V, H, W]
            cv_depth_map = depth_hires.reshape(B_img, V_img, H_img, W_img) # [B, V, H, W]
            damv2_loss = _damv2_pearson_loss(
                self.damv2,
                batch["context"]["image"],
                cv_depth_map,
            )
            self.log("loss/damv2_pearson", damv2_loss)
            try:
                wandb.log({"loss/damv2_pearson": damv2_loss.item()}, step=self.global_step)
            except Exception:
                pass
            total_loss = total_loss + self.train_cfg.damv2_loss_weight * damv2_loss

        self.log("loss/total", total_loss)

        # ---------- DAMV2 Pearson Observation (visualisation + per-view correlation log) ----------
        if (
            self.damv2 is not None
            and self.global_rank == 0
            and self.global_step % self.train_cfg.damv2_observe_every_n_steps == 0
            and "depth_highres" in vis_dump
        ):
            _damv2_observe(
                damv2=self.damv2,
                batch=batch,
                vis_dump=vis_dump,
                output_dir=self.output_dir,
                global_step=self.global_step,
            )

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"loss = {total_loss:.6f}"
            )
            
            # --- PLY Export for 3D Visualization ---
            if self.global_step % 50 == 0:
                from .ply_export import export_ply
                ply_dir = self.output_dir / "gaussians_ply"
                ply_dir.mkdir(parents=True, exist_ok=True)
                ply_path = ply_dir / f"step_{self.global_step:0>6}.ply"
                
                # Gaussians object returned by the encoder is already flattened [B, N, ...]
                b_idx = 0
                means = gaussians.means[b_idx]
                scales = gaussians.scales[b_idx]
                rotations = gaussians.rotations[b_idx]
                opacities = gaussians.opacities[b_idx] # Should be 1D (N)
                sh = gaussians.harmonics[b_idx]        # Should be (N, 3, D_SH)
                
                export_ply(
                    batch["target"]["extrinsics"][b_idx, 0], # Reference camera
                    means,
                    scales,
                    rotations,
                    sh, 
                    opacities,
                    ply_path
                )

                
                # 額外保存完整 3DGS 數據為 .pt 格式（方便 Python 分析）
                pt_path = ply_dir / f"step_{self.global_step:0>6}_full.pt"
                gaussians_data = {
                    "means": means.detach().cpu(),           # [N, 3] 位置
                    "scales": scales.detach().cpu(),         # [N, 3] 大小
                    "rotations": rotations.detach().cpu(),   # [N, 4] 四元數
                    "opacities": opacities.detach().cpu(),   # [N] 透明度
                    "harmonics": sh.detach().cpu(),          # [N, 3, D_SH] SH 係數
                    "covariances": gaussians.covariances[b_idx].detach().cpu(),  # [N, 3, 3]
                    # 額外資訊
                    "step": self.global_step,
                    "scene": batch["scene"][b_idx] if "scene" in batch else "unknown",
                    "target_extrinsics": batch["target"]["extrinsics"][b_idx].detach().cpu(),
                    "context_extrinsics": batch["context"]["extrinsics"][b_idx].detach().cpu(),
                }
                torch.save(gaussians_data, pt_path)

        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Fix Target Camera (same pipeline as training_step)
        render_near = batch["target"]["near"]
        render_far = batch["target"]["far"]
        if "rpc" in batch["context"] and "rpc" in batch["target"]:
            b_tgt, v_tgt, _ = batch["target"]["rpc"].shape
            device = batch["target"]["rpc"].device

            # Load full-image K_s0 (no crop adjustment) + skew params + crop offsets
            c2w_tgt, K_tgt, skew_params_tgt, crop_offsets_tgt, distance_tgt_flat = \
                self._load_target_cameras_full(batch["target"], device)

            batch["target"]["extrinsics"]   = rearrange(c2w_tgt,          "(b v) i j -> b v i j", b=b_tgt, v=v_tgt)
            batch["target"]["intrinsics"]   = rearrange(K_tgt,            "(b v) i j -> b v i j", b=b_tgt, v=v_tgt)
            batch["target"]["skew_params"]  = rearrange(skew_params_tgt,  "(b v) d   -> b v d",   b=b_tgt, v=v_tgt)
            batch["target"]["crop_offsets"] = rearrange(crop_offsets_tgt, "(b v) d   -> b v d",   b=b_tgt, v=v_tgt)
            distance_tgt = rearrange(distance_tgt_flat, "(b v) -> b v", b=b_tgt, v=v_tgt)

            # camera MSL = h_enu_origin + c2w[2,3]; near/far = dist ± 20m altitude sweep
            h_off_tgt = batch["target"]["rpc"][:, :, 8].to(torch.float32)
            h_enu_tgt = batch["target"]["enu_origin"][:, 0, 2].to(
                device=h_off_tgt.device, dtype=torch.float32)
            cam_msl_tgt = h_enu_tgt.unsqueeze(1) + distance_tgt.to(torch.float32)
            dist_eff_tgt = cam_msl_tgt - h_off_tgt
            scene_range = 20.0
            render_near = (dist_eff_tgt - scene_range).clamp(min=1.0)
            render_far  = (dist_eff_tgt + scene_range).clamp(min=render_near + 1.0)

            # Load crop-adjusted K_s0 for context views (encoder uses 256×256 patches)
            b_ctx, v_ctx, _ = batch["context"]["rpc"].shape
            c2w_ctx, K_ctx, _ = self._load_json_cameras(batch["context"], device)
            batch["context"]["extrinsics"] = rearrange(c2w_ctx, "(b v) i j -> b v i j", b=b_ctx, v=v_ctx)
            batch["context"]["intrinsics"] = rearrange(K_ctx,   "(b v) i j -> b v i j", b=b_ctx, v=v_ctx)

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians, _ = self.encoder(
                batch["context"], self.global_step, False, scene_names=batch["scene"]
            )

        # Export PLY if enabled in encoder config
        # Note: This relies on the hacky export_ply inside visualizer.visualize
        if self.train_cfg.extended_visualization or (
            self.encoder_visualizer is not None and getattr(self.encoder_visualizer.cfg, "export_ply", False)
        ):
             self.encoder_visualizer.visualize(batch["context"], self.global_step)
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                render_near,
                render_far,
                (h, w),
                depth_mode=None,
                skew_params=batch["target"].get("skew_params"),
                crop_offsets=batch["target"].get("crop_offsets"),
            )

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]

        # Save images.
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], images_prob):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        # Save context (training) images for inspection.
        if self.test_cfg.save_image and "image_name" in batch["context"] and "rpc" in batch["context"]:
            ctx_device = batch["context"]["rpc"].device
            ctx_dataset_root = Path(get_cfg()["dataset"]["roots"][0])
            v_ctx_t = batch["context"]["rpc"].shape[1]
            ctx_name_t = batch["context"]["image_name"]
            ctx_px_t = batch["context"]["px"]
            ctx_py_t = batch["context"]["py"]

            K_ctx_t_list, c2w_ctx_t_list, dist_ctx_t_list = [], [], []
            for v_i in range(v_ctx_t):
                img_name_c = ctx_name_t[v_i][0]
                px_val_c = ctx_px_t[v_i][0].item() if hasattr(ctx_px_t[v_i][0], "item") else int(ctx_px_t[v_i][0])
                py_val_c = ctx_py_t[v_i][0].item() if hasattr(ctx_py_t[v_i][0], "item") else int(ctx_py_t[v_i][0])

                scene_parts_c = str(img_name_c).split("_")
                scene_prefix_c = "_".join(scene_parts_c[:2])
                cam_json_c = ctx_dataset_root / "camera_pose" / f"{scene_prefix_c}_cameras" / f"{img_name_c}.json"
                with open(cam_json_c, "r") as fc:
                    cam_data_c = json.load(fc)

                K_full_c = torch.tensor(cam_data_c["K"], device=ctx_device, dtype=torch.float64).reshape(4, 4)
                w2c_c = torch.tensor(cam_data_c["W2C"], device=ctx_device, dtype=torch.float64).reshape(4, 4)

                cx_p = K_full_c[0, 2] - px_val_c * 256
                cy_p = K_full_c[1, 2] - py_val_c * 256
                K_patch_c = torch.tensor([
                    [K_full_c[0, 0], 0.0,            cx_p],
                    [0.0,            K_full_c[1, 1], cy_p],
                    [0.0,            0.0,            1.0 ],
                ], device=ctx_device, dtype=torch.float64)

                K_ctx_t_list.append(K_patch_c)
                c2w_ctx_t_list.append(torch.inverse(w2c_c))
                dist_ctx_t_list.append(w2c_c[2, 3].abs())

            ctx_extr = rearrange(torch.stack(c2w_ctx_t_list), "v i j -> 1 v i j")
            ctx_intr = rearrange(torch.stack(K_ctx_t_list),   "v i j -> 1 v i j")
            dist_ctx = rearrange(torch.stack(dist_ctx_t_list), "v -> 1 v")

            h_off_ctx = batch["context"]["rpc"][:, :, 8]
            ctx_near = (h_off_ctx + dist_ctx - (h_off_ctx + 50000.0)).clamp(min=1.0)
            ctx_far  = (h_off_ctx + dist_ctx - (h_off_ctx - 50000.0)).clamp(min=ctx_near + 1.0)

            output_ctx = self.decoder.forward(
                gaussians,
                ctx_extr,
                ctx_intr,
                ctx_near,
                ctx_far,
                (h, w),
                depth_mode=None,
            )

            for index, color in zip(batch["context"]["index"][0], output_ctx.color[0]):
                save_image(color, path / scene / f"context_color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = images_prob

            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Set up RPC cameras (same pipeline as training_step)
        render_near = batch["target"]["near"]
        render_far = batch["target"]["far"]
        if "rpc" in batch["context"] and "rpc" in batch["target"]:
            b_tgt, v_tgt, _ = batch["target"]["rpc"].shape
            device = batch["target"]["rpc"].device

            c2w_tgt, K_tgt, skew_params_tgt, crop_offsets_tgt, distance_tgt_flat = \
                self._load_target_cameras_full(batch["target"], device)

            batch["target"]["extrinsics"]   = rearrange(c2w_tgt,          "(b v) i j -> b v i j", b=b_tgt, v=v_tgt)
            batch["target"]["intrinsics"]   = rearrange(K_tgt,            "(b v) i j -> b v i j", b=b_tgt, v=v_tgt)
            batch["target"]["skew_params"]  = rearrange(skew_params_tgt,  "(b v) d   -> b v d",   b=b_tgt, v=v_tgt)
            batch["target"]["crop_offsets"] = rearrange(crop_offsets_tgt, "(b v) d   -> b v d",   b=b_tgt, v=v_tgt)
            distance_tgt = rearrange(distance_tgt_flat, "(b v) -> b v", b=b_tgt, v=v_tgt)

            h_off_tgt = batch["target"]["rpc"][:, :, 8].to(torch.float32)
            h_enu_tgt = batch["target"]["enu_origin"][:, 0, 2].to(
                device=h_off_tgt.device, dtype=torch.float32)
            cam_msl_tgt = h_enu_tgt.unsqueeze(1) + distance_tgt.to(torch.float32)
            dist_eff_tgt = cam_msl_tgt - h_off_tgt
            scene_range = 20.0
            render_near = (dist_eff_tgt - scene_range).clamp(min=1.0)
            render_far  = (dist_eff_tgt + scene_range).clamp(min=render_near + 1.0)

            b_ctx, v_ctx, _ = batch["context"]["rpc"].shape
            c2w_ctx, K_ctx, skew_params_ctx, crop_offsets_ctx, _ = \
                self._load_target_cameras_full(batch["context"], device)
            batch["context"]["extrinsics"] = rearrange(c2w_ctx, "(b v) i j -> b v i j", b=b_ctx, v=v_ctx)
            batch["context"]["intrinsics"] = rearrange(K_ctx,   "(b v) i j -> b v i j", b=b_ctx, v=v_ctx)
            batch["context"]["skew_params"]  = rearrange(skew_params_ctx,  "(b v) d   -> b v d",   b=b_ctx, v=v_ctx)
            batch["context"]["crop_offsets"] = rearrange(crop_offsets_ctx, "(b v) d   -> b v d",   b=b_ctx, v=v_ctx)

        with torch.no_grad():
            gaussians_softmax, _ = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )
            output_softmax = self.decoder.forward(
                gaussians_softmax,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                render_near,
                render_far,
                (h, w),
                skew_params=batch["target"].get("skew_params"),
                crop_offsets=batch["target"].get("crop_offsets"),
            )
        rgb_softmax = output_softmax.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        lpips = compute_lpips(rgb_gt, rgb_softmax).mean()
        ssim = compute_ssim(rgb_gt, rgb_softmax).mean()
        mse = ((rgb_gt.clamp(0, 1) - rgb_softmax.clamp(0, 1)) ** 2).mean()
        self.log("val/lpips_val", lpips, sync_dist=True)
        self.log("val/ssim_val", ssim, sync_dist=True)
        self.log("val/mse_val", mse, sync_dist=True)

        # Validation losses (same loss functions as training, for direct comparison).
        val_total = 0
        for loss_fn in self.losses:
            val_loss = loss_fn.forward(output_softmax, batch, gaussians_softmax, self.global_step)
            self.log(f"val/{loss_fn.name}", val_loss, sync_dist=True)
            val_total = val_total + val_loss
        self.log("val/total", val_total, sync_dist=True)

        # Save rendered vs GT images and log to WandB (rank 0 only).
        if self.global_rank == 0:
            import torchvision

            vis_val_dir = self.output_dir / "vis_val"
            vis_val_dir.mkdir(parents=True, exist_ok=True)

            # Save side-by-side comparison for each target view
            for vi in range(rgb_softmax.shape[0]):
                comparison = torch.cat([rgb_gt[vi], rgb_softmax[vi]], dim=-1)
                img_path = vis_val_dir / f"step_{self.global_step:0>6}_v{vi}.png"
                torchvision.utils.save_image(comparison, img_path)

            # Log comparison image to WandB
            try:
                comparison_all = torch.cat([rgb_gt[0], rgb_softmax[0]], dim=-1)
                wandb.log(
                    {
                        "val/render_vs_gt": wandb.Image(
                            comparison_all.permute(1, 2, 0).clamp(0, 1).cpu().numpy(),
                            caption=f"step {self.global_step} | GT (left) Render (right)",
                        ),
                        "val/lpips_val": lpips.item(),
                        "val/ssim_val": ssim.item(),
                        "val/mse_val": mse.item(),
                    },
                    step=self.global_step,
                )
            except Exception:
                pass

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob, _ = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            valid_depths = result[result > 0][:16_000_000]
            if valid_depths.numel() == 0:
                # Fallback for empty depth maps
                near = torch.tensor(0.01, device=result.device).log()
                far = torch.tensor(100.0, device=result.device).log()
            else:
                near = valid_depths.quantile(0.01).log()
                far = result.view(-1)[:16_000_000].quantile(0.99).log()
            
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=30)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            # print(f"Cosine annealing LR enabled")
            

            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps ,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
