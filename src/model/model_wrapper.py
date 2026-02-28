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

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        # Ensure Target Extrinsics match the RPC coordinate system defined by Context
        if "rpc" in batch["context"] and "rpc" in batch["target"]:
            from ..geometry.rpc import RPC
            b_ctx, v_ctx, _ = batch["context"]["rpc"].shape
            b_tgt, v_tgt, _ = batch["target"]["rpc"].shape
            
            # 計算 context 第一視角圖像中心的真實經緯度作為 ENU 參考點
            rpc_first = RPC(batch["context"]["rpc"][:, 0, :])  # [B, 90]
            h_center_ref = batch["context"]["rpc"][:, 0, 8]   # HEIGHT_OFF 作為高度
            u_center = torch.full_like(h_center_ref, h / 2.0)
            v_center = torch.full_like(h_center_ref, w / 2.0)
            lat_ref_origin, lon_ref_origin = rpc_first.inverse(u_center, v_center, h_center_ref)  # [B]
            
            # Expand to matches Target (B * V_tgt)
            lat_ref_flat = lat_ref_origin.repeat_interleave(v_tgt)
            lon_ref_flat = lon_ref_origin.repeat_interleave(v_tgt)
            
            rpc_target_flat = RPC(rearrange(batch["target"]["rpc"], "b v c -> (b v) c"))
            
            # 從 RPC Jacobian 推導虛擬相機幾何（完全 Data-driven，無 hardcoded 參數）
            K_tgt, c2w_tgt, distance_tgt_flat = rpc_target_flat.compute_camera_geometry(h, w, lat_ref_flat, lon_ref_flat)

            # --- Hardcoded camera parameters for debugging ---
            K_tgt = torch.tensor([[
                [1187255.75952377,   0.0,                113722.91093382722],
                [0.0,                1178055.401262924,  -194568.0592035301],
                # [1187255.75952377, 0.0,              112954.91093382722], // 算過的，第0,3 個
                # [0.0,              1178055.401262924, -194568.0592035301],
                [0.0,              0.0,               1.0              ],
            ]], dtype=K_tgt.dtype, device=K_tgt.device)
            c2w_tgt = torch.linalg.inv(torch.tensor([[
                [0.8867532204124197,  -0.009486012188437915,  0.4621458013018875,  -37477.919892478676],
                [0.10741489563426863, -0.9681921197734719,   -0.22597800646175822,  65542.12029332563 ],
                [0.4495895531304993,   0.25002806798700034,  -0.8575285411778468,   394736.3683483885 ],
                [0.0,                  0.0,                   0.0,                  1.0               ],
            ]], dtype=c2w_tgt.dtype, device=c2w_tgt.device))
            # Since RPC and Image are both (h, w), no further scaling of K is needed for projection.
            # Just ensure the batch is updated with these consistent matrices.
            batch["target"]["extrinsics"] = rearrange(c2w_tgt, "(b v) i j -> b v i j", b=b_tgt, v=v_tgt)
            batch["target"]["intrinsics"] = rearrange(K_tgt, "(b v) i j -> b v i j", b=b_tgt, v=v_tgt)
            
            # 保存 distance 供後續計算 near/far 使用（避免重複呼叫 compute_camera_geometry）
            distance_tgt = rearrange(distance_tgt_flat, "(b v) -> b v", b=b_tgt, v=v_tgt)
            
        gaussians, vis_dump = self.encoder(batch["context"], self.global_step, False, scene_names=batch["scene"])
        
        # ============================================================
        # [RENDER DEBUG] 詳細輸出相機和 Gaussian 參數
        # ============================================================
        if False:
            print("\n" + "=" * 70)
            print("🎥 [RENDER DEBUG] 渲染參數詳細分析")
            print("=" * 70)
            
            # 1. Target Camera 參數
            tgt_ext = batch["target"]["extrinsics"][0, 0]  # 第一個 batch, 第一個 target view
            tgt_int = batch["target"]["intrinsics"][0, 0]
            tgt_near = batch["target"]["near"][0, 0]
            tgt_far = batch["target"]["far"][0, 0]
            
            print("\n📷 Target Camera (用於渲染):")
            print(f"   位置 (c2w[:3,3]): [{tgt_ext[0,3].item():.2f}, {tgt_ext[1,3].item():.2f}, {tgt_ext[2,3].item():.2f}]")
            print(f"   焦距 (fx, fy): [{tgt_int[0,0].item():.2f}, {tgt_int[1,1].item():.2f}]")
            print(f"   主點 (cx, cy): [{tgt_int[0,2].item():.2f}, {tgt_int[1,2].item():.2f}]")
            print(f"   near/far: [{tgt_near.item():.2f}, {tgt_far.item():.2f}]")
            
            # 計算 FOV
            import math
            fx = tgt_int[0,0].item()
            fy = tgt_int[1,1].item()
            fov_x = 2 * math.atan(h / (2 * fx)) * 180 / math.pi
            fov_y = 2 * math.atan(w / (2 * fy)) * 180 / math.pi
            print(f"   FOV (度): [{fov_x:.2f}, {fov_y:.2f}]")
            
            # 2. Gaussian 參數
            g_means = gaussians.means  # [B, N, 3] after flatten
            # Flatten all views
            g_means_flat = g_means.view(-1, 3)
            print(f"\n🔵 Gaussians 位置:")
            print(f"   總數: {g_means_flat.shape[0]}")
            print(f"   X: min={g_means_flat[:,0].min().item():.2f}, max={g_means_flat[:,0].max().item():.2f}, mean={g_means_flat[:,0].mean().item():.2f}")
            print(f"   Y: min={g_means_flat[:,1].min().item():.2f}, max={g_means_flat[:,1].max().item():.2f}, mean={g_means_flat[:,1].mean().item():.2f}")
            print(f"   Z: min={g_means_flat[:,2].min().item():.2f}, max={g_means_flat[:,2].max().item():.2f}, mean={g_means_flat[:,2].mean().item():.2f}")
            
            # 3. 計算相機是否能看到 Gaussians
            cam_pos = tgt_ext[:3, 3]
            gs_center = g_means_flat.mean(dim=0)
            cam_to_gs = gs_center - cam_pos
            distance_cam_to_gs = torch.norm(cam_to_gs).item()
            
            print(f"\n📏 相機與 Gaussians 的關係:")
            print(f"   相機位置: [{cam_pos[0].item():.2f}, {cam_pos[1].item():.2f}, {cam_pos[2].item():.2f}]")
            print(f"   Gaussians 中心: [{gs_center[0].item():.2f}, {gs_center[1].item():.2f}, {gs_center[2].item():.2f}]")
            print(f"   距離: {distance_cam_to_gs:.2f} 米")
            
            # 相機朝向（假設 Z 軸是前方）
            cam_forward = tgt_ext[:3, 2]  # c2w 的第三列是相機的 Z 軸方向
            print(f"   相機朝向 (Z軸): [{cam_forward[0].item():.4f}, {cam_forward[1].item():.4f}, {cam_forward[2].item():.4f}]")
            
            # 檢查相機高度 vs Gaussian 高度
            cam_z = cam_pos[2].item()
            gs_z_min = g_means_flat[:,2].min().item()
            gs_z_max = g_means_flat[:,2].max().item()
            print(f"\n⚠️ 高度對齊檢查:")
            print(f"   相機高度 (Z): {cam_z:.2f}")
            print(f"   Gaussians 高度範圍: [{gs_z_min:.2f}, {gs_z_max:.2f}]")
            
            if cam_z < gs_z_min:
                print(f"   ❌ 相機在 Gaussians 下方！")
            elif cam_z > gs_z_max:
                print(f"   ℹ️ 相機在 Gaussians 上方 (這是預期的，因為是俯視)")
            else:
                print(f"   ⚠️ 相機在 Gaussians 範圍內")
            
            # 4. 計算 Gaussians 是否在視錐體內
            # 簡單檢查：在相機座標系中，Gaussians 應該在 near-far 範圍內
            w2c = torch.inverse(tgt_ext)  # world to camera
            gs_in_cam = (w2c[:3, :3] @ g_means_flat.T + w2c[:3, 3:4]).T  # [N, 3]
            gs_depths = gs_in_cam[:, 2]  # 在相機座標系中的 Z（深度）
            
            print(f"\n📊 Gaussians 在相機座標系中的深度:")
            print(f"   深度範圍: [{gs_depths.min().item():.2f}, {gs_depths.max().item():.2f}]")
            print(f"   near/far: [{tgt_near.item():.2f}, {tgt_far.item():.2f}]")
            
            in_frustum_depth = (gs_depths > tgt_near) & (gs_depths < tgt_far)
            print(f"   在 near-far 內的 Gaussians: {in_frustum_depth.sum().item()}/{len(gs_depths)} ({in_frustum_depth.sum().item()/len(gs_depths)*100:.1f}%)")
            
            # 檢查負深度（在相機後面）
            behind_camera = gs_depths < 0
            print(f"   ❌ 在相機後面的 Gaussians: {behind_camera.sum().item()}/{len(gs_depths)} ({behind_camera.sum().item()/len(gs_depths)*100:.1f}%)")
            
            print("=" * 70 + "\n")
            # exit()
        # ============================================================
        # [CRITICAL FIX] 動態計算渲染器需要的 near/far
        # 使用上方已計算的 distance_tgt（來自 Block 1 的 compute_camera_geometry）
        # ============================================================
        if "rpc" in batch["context"] and "rpc" in batch["target"]:
            # 直接使用 Block 1 已計算的 distance_tgt，無需重複呼叫 compute_camera_geometry
            h_off_tgt = batch["target"]["rpc"][:, :, 8]    # HEIGHT_OFF [B, V_target]
            scene_range = 20.0  # ±20m 場景範圍（與 encoder_costvolume 一致）
            scene_z_min = h_off_tgt - scene_range
            scene_z_max = h_off_tgt + scene_range
            
            # 相機高度（MSL 絕對高度）
            camera_z_tgt = h_off_tgt + distance_tgt
            
            # Near/Far 定義（相機到場景的距離）
            render_near = (camera_z_tgt - scene_z_max).clamp(min=1.0)
            render_far = (camera_z_tgt - scene_z_min).clamp(min=render_near + 1.0)
            
        else:
            render_near = batch["target"]["near"]
            render_far = batch["target"]["far"]
        
        render_near = torch.full_like(render_near, 1.0, dtype=torch.float32)
        render_far = torch.full_like(render_far, 10000.0, dtype=torch.float32)
        # 手動建立一個 3x3 內參矩陣
        # 手動建立一個 3x3 內參矩陣
        # device = batch["target"]["intrinsics"].device
        # dtype = batch["target"]["intrinsics"].dtype
        # new_K = torch.tensor([
        #     [12, 0.0,        128.0],
        #     [0.0,        12, 128.0],
        #     [0.0,        0.0,        1.0  ]
        # ], device=device, dtype=dtype)
        # new_K_batched = new_K.unsqueeze(0).unsqueeze(0).expand(1, 3, 3, 3)
        # print(f"render_near , render_far  before decoder forward : {render_near}, {render_far}")
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            # new_K,
            render_near,
            render_far,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
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

            # 4. Log Detailed Stats
            with open(self.output_dir / "training_debug.log", "a") as f:
                f.write(f"\n--- STEP {self.global_step} ---\n")
                if "rpc" in batch["context"]:
                    rpc_data = batch["context"]["rpc"][0, 0].detach().cpu().numpy()
                    f.write(f"RPC Coefficients (First Batch, First View):\n")
                    f.write(f"{rpc_data.tolist()}\n")
                
                means_flat = gaussians.means.detach().reshape(-1, 3)
                f.write(f"  Means range: min={means_flat.min().item():.2f}, max={means_flat.max().item():.2f}, std={means_flat.std(dim=0).cpu().numpy()}\n")
                f.write(f"  Opacities: mean={gaussians.opacities.mean().item():.4f}, max={gaussians.opacities.max().item():.4f}\n")
                
                # SH 顏色統計
                sh = gaussians.harmonics.detach()  # [B, N, 3, d_sh]
                dc = sh[..., 0]  # DC component: [B, N, 3]
                f.write(f"  SH DC (color): R={dc[..., 0].mean().item():.4f}, G={dc[..., 1].mean().item():.4f}, B={dc[..., 2].mean().item():.4f}\n")
                f.write(f"  SH DC std: R={dc[..., 0].std().item():.4f}, G={dc[..., 1].std().item():.4f}, B={dc[..., 2].std().item():.4f}\n")
                
                f.write(f"  Target Extrinsics Trans: {batch['target']['extrinsics'][0, 0, :3, 3].cpu().numpy()}\n")
                f.write(f"  Near/Far: {batch['target']['near'][0, 0].item():.2f} / {batch['target']['far'][0, 0].item():.2f}\n")
                f.write(f"  Depth Range in Dump: {d_min.item():.2f} ~ {d_max.item():.2f}\n")

        # Compute metrics.
        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        
        # Opacity regularization: 防止所有 Gaussian 變透明（opacity → 0）
        # 確保平均 opacity 保持在合理範圍（至少 0.3）
        opacity_mean = gaussians.opacities.mean()
        opacity_reg_weight = 0.1
        opacity_target = 0.3
        opacity_reg = opacity_reg_weight * torch.relu(opacity_target - opacity_mean)
        total_loss = total_loss + opacity_reg
        self.log("loss/opacity_reg", opacity_reg)
        
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"loss = {total_loss:.6f}"
            )
            
            # Debug: Check Gradients and Flow
            # 注意：這個檢查是在 backward 之前執行的，所以它看到的是「上一個 step」的 gradient。
            # 在 validation interval 之後，gradient 可能被清空，這是正常的。
            if self.global_step > 0 and self.global_step % 10 == 0:
                print(f"--- Gradient Check (Step {self.global_step}, from previous backward) ---")
                grad_stats = {}
                for name, param in self.encoder.named_parameters():
                    if param.grad is not None:
                        g_max = param.grad.abs().max().item()
                        if g_max > 1e-12:
                            layer_name = name.split('.')[0] if '.' in name else name
                            if layer_name not in grad_stats:
                                grad_stats[layer_name] = {'count': 0, 'max': 0}
                            grad_stats[layer_name]['count'] += 1
                            grad_stats[layer_name]['max'] = max(grad_stats[layer_name]['max'], g_max)
                
                if grad_stats:
                    print(" Gradient Flow by Layer:")
                    for layer, stats in grad_stats.items():
                        print(f"   {layer}: {stats['count']} params, max_grad={stats['max']:.2e}")
                else:
                    # 在 validation interval 後 gradient 被清空是正常的
                    print(" INFO: No gradients from previous step (normal after validation or optimizer.step)")

            # --- Mandatory 3DGS PLY Export for Debugging ---
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
                print(f" [SAVE] Saved 3DGS PLY to: {ply_path}")
                
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
                print(f" [SAVE] Saved full 3DGS data to: {pt_path}")

        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        # exit()
        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Fix Target Camera (Same as Training)
        if "rpc" in batch["context"] and "rpc" in batch["target"]:
            from ..geometry.rpc import RPC
            b_ctx, v_ctx, _ = batch["context"]["rpc"].shape
            b_tgt, v_tgt, _ = batch["target"]["rpc"].shape
            
            # 計算 context 第一視角圖像中心的真實經緯度作為 ENU 參考點
            rpc_first = RPC(batch["context"]["rpc"][:, 0, :])  # [B, 90]
            h_center_ref = batch["context"]["rpc"][:, 0, 8]   # HEIGHT_OFF 作為高度
            u_center = torch.full_like(h_center_ref, h / 2.0)
            v_center = torch.full_like(h_center_ref, w / 2.0)
            lat_ref_origin, lon_ref_origin = rpc_first.inverse(u_center, v_center, h_center_ref)  # [B]
            
            lat_ref_flat = lat_ref_origin.repeat_interleave(v_tgt)
            lon_ref_flat = lon_ref_origin.repeat_interleave(v_tgt)
            
            rpc_target_flat = RPC(rearrange(batch["target"]["rpc"], "b v c -> (b v) c"))
            # 直接使用實際影像尺寸 (h, w)，完全由 RPC 數據驅動（height_scale × GSD_rpc）
            K_tgt, c2w_tgt, _ = rpc_target_flat.compute_camera_geometry(h, w, lat_ref_flat, lon_ref_flat)
            
            batch["target"]["extrinsics"] = rearrange(c2w_tgt, "(b v) i j -> b v i j", b=b_tgt, v=v_tgt)
            batch["target"]["intrinsics"] = rearrange(K_tgt, "(b v) i j -> b v i j", b=b_tgt, v=v_tgt)

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
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
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

    @rank_zero_only
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
        gaussians_softmax, _ = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_softmax = output_softmax.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # # Construct comparison image.
        # comparison = hcat(
        #     add_label(vcat(*batch["context"]["image"][0]), "Context"),
        #     add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
        #     add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        # )
        # self.logger.log_image(
        #     "comparison",
        #     [prep_image(add_border(comparison))],
        #     step=self.global_step,
        #     caption=batch["scene"],
        # )

        # # Render projections and construct projection image.
        # projections = hcat(*render_projections(
        #                         gaussians_softmax,
        #                         256,
        #                         extra_label="(Softmax)",
        #                     )[0])
        # self.logger.log_image(
        #     "projection",
        #     [prep_image(add_border(projections))],
        #     step=self.global_step,
        # )

        # # Draw cameras.
        # cameras = hcat(*render_cameras(batch, 256))
        # self.logger.log_image(
        #     "cameras", [prep_image(add_border(cameras))], step=self.global_step
        # )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

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
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
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
