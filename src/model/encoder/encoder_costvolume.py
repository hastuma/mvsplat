from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView, EncoderOutput
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
from ...geometry.rpc import RPC
from einops import repeat

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool


class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            self.epipolar_sampler = EpipolarSampler(
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
            no_cross_attn=cfg.wo_backbone_cross_attn,
            use_epipolar_trans=cfg.use_epipolar_trans,
        )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not cfg.wo_backbone_cross_attn
                self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> tuple[Gaussians, dict]:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        
        if visualization_dump is None:
            visualization_dump = {}

        # Encode the context images.
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        trans_features, cnn_features = self.backbone(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )
        if visualization_dump is not None:
            visualization_dump["cnn_features"] = cnn_features

        # Sample depths from the resulting features.
        in_feats = trans_features
        gpp = self.cfg.gaussians_per_pixel
        
        # --- CRITICAL: RPC Camera Pre-computation ---
        # We must calculate the virtual pinhole cameras BEFORE the depth_predictor
        # so that the cost volume matching uses physically correct intrinsics/poses.
        if "rpc" in context:
            rpc_flat = RPC(rearrange(context["rpc"], "b v c -> (b v) c"))
            
            # 計算第一視角圖像中心的真實經緯度作為 ENU 參考點
            # 這比使用 RPC 的 LAT_OFF/LONG_OFF 更準確，因為那只是 RPC 正規化參數
            rpc_first = RPC(context["rpc"][:, 0, :])  # [B, 90]
            h_center_ref = context["rpc"][:, 0, 8]   # HEIGHT_OFF 作為高度
            u_center = torch.full_like(h_center_ref, h / 2.0)
            v_center = torch.full_like(h_center_ref, w / 2.0)
            lat_ref_b, lon_ref_b = rpc_first.inverse(u_center, v_center, h_center_ref)  # [B]
            print(f"lat_ref_b,lon_ref_b", lat_ref_b, lon_ref_b)
            # kgorepgw = input("RPC camera pre-computation ")
            # 存儲用於後續的 Gaussian 位置計算
            context["_lat_ref"] = lat_ref_b
            context["_lon_ref"] = lon_ref_b
            
            # 從 RPC Jacobian 近似出虛擬相機幾何（完全 Data-driven，由 height_scale × GSD_rpc 自動推導）
            K_approx, c2w, distance_flat = rpc_flat.compute_camera_geometry(h, w, lat_ref_b, lon_ref_b)
            # distance_flat: [B*V]，reshape 為 [B, V] 供後續 per-view 使用
            distance_bv = rearrange(distance_flat, "(b v) -> b v", b=b, v=v)
            w2c = torch.linalg.inv(c2w)
            # Update Context: Now depth_predictor will see these correct cameras!
            context["extrinsics"] = rearrange(c2w, "(b v) i j -> b v i j", b=b, v=v)
            context["intrinsics"] = rearrange(K_approx, "(b v) i j -> b v i j", b=b, v=v)
            
            # 存儲 per-view distance [B, V] 供後續使用
            context["_virtual_camera_distance"] = distance_bv

        # ============================================================
        # 動態計算 near/far（相機座標系的深度範圍）
        # ============================================================
        if "rpc" in context:
            # 從 RPC 參數取得高度偏移量 (Height Offset)
            h_off = context["rpc"][:, :, 8]    # HEIGHT_OFF [B, V]
            scene_range = 20.0
            scene_z_min = h_off - scene_range  # 最低海拔 (MSL)
            scene_z_max = h_off + scene_range  # 最高海拔 (MSL)
            
            # 3. 相機高度決定（使用 RPC 推導的 per-view distance [B, V]）
            distance_bv = context.get("_virtual_camera_distance", None)
            if distance_bv is None:
                # fallback: 若無 RPC，使用 h_off 形狀的預設值
                distance_bv = torch.full_like(h_off, 44.8)
            camera_z = h_off + distance_bv
            
            # 4. Near/Far 定義 (這是"距離" Distance, 而非絕對海拔)
            # near = 相機到最高海拔平面的垂直距離
            # far = 相機到最低海拔平面的垂直距離
            distance_near = (camera_z - scene_z_max).clamp(min=1.0)
            distance_far = (camera_z - scene_z_min ).clamp(min=distance_near + 1.0)
            
            # if global_step % 50 == 0:
            print(f"\n[GEOMETRY CONFIG] Encoder")
            print(f"  HEIGHT_OFF: {h_off[0,0].item():.1f}m")
            print(f"  Derived H (from RPC GSD×w/2): {distance_bv[0,0].item():.2f}m  "
                  f"focal≈{(distance_bv[0,0]/0.35).item():.1f}px")
            print(f"  Search Range (MSL Alt): [{scene_z_min[0,0].item():.1f}, {scene_z_max[0,0].item():.1f}]")
            print(f"  Camera Altitude (MSL): {camera_z[0,0].item():.1f}")
            print(f"  Distance near/far (to Cam): [{distance_near[0,0].item():.1f}, {distance_far[0,0].item():.1f}]")
            # hjiotehte = input()

        # Prepare extra_info for depth_predictor
        extra_info = {}
        extra_info["rpcs"] = context["rpc"]
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        # 傳遞 per-view 推導相機高度 [B, V]，讓 depth_predictor 做正確的 altitude 轉換
        if "_virtual_camera_distance" in context:
            extra_info["virtual_camera_distance"] = context["_virtual_camera_distance"]

        encoder_output = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            distance_near,
            distance_far,
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
            visualization_dump=visualization_dump,
        )
        depths = encoder_output.depths #應該是距離相機的距離，印出來的值大概都是100 
        densities = encoder_output.densities
        raw_gaussians = encoder_output.raw_gaussians
        # print("depths", depths)
        # print("densities", densities.shape)
        # print("raw_gaussians", raw_gaussians[0, 0, :5])
        # gergeg = input("take a look ")
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        # Predict Gaussian parameters.
        if "rpc" in context:
            # RPC branch: Compute ENU means using the already updated cameras
            b, v, r, srf, _ = depths.shape
            # distance_bv: [B, V]，每個視角各自的虛擬相機高度（由 RPC Jacobian 推導）
            distance_bv = context["_virtual_camera_distance"]  # [B, V]
            
            u_all = rearrange(xy_ray[..., 1] * h, "b v r srf -> (b v r srf)")
            v_all = rearrange(xy_ray[..., 0] * w, "b v r srf -> (b v r srf)")
            
            # Match Distance-First scheme: depths are distances from camera.
            dist_all = rearrange(depths, "b v r srf () -> (b v r srf)")
            h_off_all_v = rearrange(repeat(context["rpc"][:, :, 8], "b v -> b v r srf", r=r, srf=srf), "b v r srf -> (b v r srf)")
            # 展開 per-view distance 到 [B*V*R*SRF]
            distance_all = rearrange(repeat(distance_bv, "b v -> b v r srf", r=r, srf=srf), "b v r srf -> (b v r srf)")
            
            # Altitude = Cam_Altitude - Distance
            h_all = (h_off_all_v + distance_all) - dist_all
            
            # ============================================================
            # [DEBUG] 儲存 3 個視角的預測高度圖 (Height Maps)
            # ============================================================
            if global_step % 10 == 0:
                import os
                from PIL import Image
                import numpy as np
                
                # 取得當前輸出的根目錄，增加多重保險
                output_dir = None
                try:
                    import hydra
                    output_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
                except Exception:
                    pass
                
                if output_dir is None:
                    output_dir = "outputs_debug"
                
                save_dir = os.path.join(output_dir, "debug_height_maps")
                os.makedirs(save_dir, exist_ok=True)
                
                # h_all shape: [B*V*R*SRF] -> 我們需要轉回 [B, V, H, W]
                # 這裡的 h, w 是影像解析度
                try:
                    h_maps = rearrange(h_all, "(b v h w) -> b v h w", b=b, v=v, h=h, w=w)
                    
                    for v_idx in range(v):
                        # 取出 Batch 0, View v_idx 的高度圖
                        h_map_v = h_maps[0, v_idx].detach().cpu().numpy()
                        
                        # 正規化到 0~255 (以該視角的 min/max 為基準)
                        h_min, h_max = h_map_v.min(), h_map_v.max()
                        if h_max > h_min:
                            norm_map = (h_map_v - h_min) / (h_max - h_min)
                        else:
                            norm_map = np.zeros_like(h_map_v)
                            
                        # 轉換為 uint8 影像
                        img_array = (norm_map * 255.0).astype(np.uint8)
                        
                        # 存檔
                        save_path = os.path.join(save_dir, f"step_{global_step:06d}_view_{v_idx}_height.png")
                        Image.fromarray(img_array).save(save_path)
                    print(f"[VISUALIZATION] Saved {v} Height Maps to {save_dir}")
                except Exception as e:
                    print(f"\n[VISUALIZATION ERROR] Failed to save height maps: {e}")

            if global_step % 10 == 0:
                print(f"\n[GEOMETRY LOG] Encoder_CostVolume (Output)")
                print(f"  > Mean Predicted Distance: {dist_all.mean().item():.2f}m")
                print(f"  > Mean Target Altitude: {h_all.mean().item():.2f}m MSL")

            # For GaussianAdapter, we need the "distance from camera" to scale the 3DGS correctly.
            rel_depths_adapter = rearrange(dist_all, "(b v r srf) -> b v r srf ()", b=b, v=v, r=r, srf=srf)
            
            rpc_all = RPC(repeat(context["rpc"], "b v c -> (b v r srf) c", r=r, srf=srf))
            lat_a, lon_a = rpc_all.inverse(u_all, v_all, h_all)
            
            # ============================================================
            # [DEBUG] 驗證 3 個視角 RPC Inverse 結果是否重疊
            # ============================================================
            """
            with torch.no_grad():
                lat_per_view = rearrange(lat_a, "(b v r srf) -> b v (r srf)", b=b, v=v, r=r, srf=srf)
                lon_per_view = rearrange(lon_a, "(b v r srf) -> b v (r srf)", b=b, v=v, r=r, srf=srf)
                
                print(f"\n{'='*70}")
                print(f"[RPC INVERSE DEBUG] 3 視角經緯度比較 (Batch 0)")
                print(f"{'='*70}")
                
                for vi in range(v):
                    lat_v = lat_per_view[0, vi]
                    lon_v = lon_per_view[0, vi]
                    # 各視角的 RPC 原始 LAT_OFF / LONG_OFF
                    rpc_lat_off = context["rpc"][0, vi, 4].item()
                    rpc_lon_off = context["rpc"][0, vi, 6].item()
                    print(f"\n  View {vi}:")
                    print(f"    RPC params:  LAT_OFF={rpc_lat_off:.8f}, LON_OFF={rpc_lon_off:.8f}")
                    print(f"    Inverse結果: lat=[{lat_v.min().item():.8f}, {lat_v.max().item():.8f}], mean={lat_v.mean().item():.8f}")
                    print(f"    Inverse結果: lon=[{lon_v.min().item():.8f}, {lon_v.max().item():.8f}], mean={lon_v.mean().item():.8f}")
                
                # 計算 view 間的重疊度
                rad_c = 3.141592653589793 / 180.0
                r_e = 6378137.0
                cos_lat_ref = torch.cos(lat_per_view[0, 0].mean() * rad_c)
                
                print(f"\n  --- 視角間重疊分析 (轉換為公尺) ---")
                for vi in range(v):
                    for vj in range(vi + 1, v):
                        lat_i, lon_i = lat_per_view[0, vi], lon_per_view[0, vi]
                        lat_j, lon_j = lat_per_view[0, vj], lon_per_view[0, vj]
                        
                        # 中心點偏移 (公尺)
                        dx = (lon_i.mean() - lon_j.mean()) * rad_c * r_e * cos_lat_ref
                        dy = (lat_i.mean() - lat_j.mean()) * rad_c * r_e
                        
                        # 各自的 spatial extent (公尺)
                        ext_i_x = (lon_i.max() - lon_i.min()) * rad_c * r_e * cos_lat_ref
                        ext_j_x = (lon_j.max() - lon_j.min()) * rad_c * r_e * cos_lat_ref
                        ext_i_y = (lat_i.max() - lat_i.min()) * rad_c * r_e
                        ext_j_y = (lat_j.max() - lat_j.min()) * rad_c * r_e
                        
                        print(f"\n  View {vi} vs View {vj}:")
                        print(f"    中心偏移: dx={dx.item():.2f}m, dy={dy.item():.2f}m")
                        print(f"    View {vi} 覆蓋範圍: {ext_i_x.item():.2f}m × {ext_i_y.item():.2f}m")
                        print(f"    View {vj} 覆蓋範圍: {ext_j_x.item():.2f}m × {ext_j_y.item():.2f}m")
                        
                        # 判斷重疊
                        avg_ext = (ext_i_x + ext_j_x).item() / 2
                        shift = (dx.abs() + dy.abs()).item()
                        if shift < avg_ext * 0.1:
                            print(f"    ✅ 高度重疊 (偏移僅 {shift:.2f}m vs 範圍 {avg_ext:.2f}m)")
                        elif shift < avg_ext * 0.5:
                            print(f"    ⚠️ 部分重疊 (偏移 {shift:.2f}m vs 範圍 {avg_ext:.2f}m)")
                        else:
                            print(f"    ❌ 幾乎不重疊！(偏移 {shift:.2f}m >> 範圍 {avg_ext:.2f}m)")
                
                print(f"\n{'='*70}\n")
                # ehoprwtjoptw = input()"""
            # ENU Origin: 使用先前計算並存儲的圖像中心經緯度
            lat_ref_b = context["_lat_ref"]  # [B]
            lon_ref_b = context["_lon_ref"]  # [B]
            lat_ref_a = repeat(lat_ref_b, "b -> (b v r srf)", v=v, r=r, srf=srf)
            lon_ref_a = repeat(lon_ref_b, "b -> (b v r srf)", v=v, r=r, srf=srf)
            
            rad = 3.141592653589793 / 180.0
            r_earth = 6378137.0
            cos_lat_a = torch.cos(lat_ref_a * rad)
            
            x_a = (lon_a - lon_ref_a) * rad * r_earth * cos_lat_a
            y_a = (lat_a - lat_ref_a) * rad * r_earth
            z_a = h_all
            means = torch.stack([x_a, y_a, z_a], dim=-1)

            # 3. Gaussian Adapter: Use 1.0 as depth because virtual camera is 1m from ground
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                rel_depths_adapter, 
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
                (h, w),
            )
            
            if global_step % 10 == 0:
                g_mean = gaussians.means[0, 0].detach().mean(dim=0) 
                cam_pos = context["extrinsics"][0, 0, :3, 3].detach()
                print(f"Gaussian Center (Mean): {g_mean.cpu().numpy()}")
                print(f"Camera Position       : {cam_pos.cpu().numpy()}")
                print(f"Distance (Cam <-> GS) : {torch.norm(g_mean - cam_pos).item():.2f} meters")
                print(f"  Opacities: min={gaussians.opacities.min().item():.2e}, max={gaussians.opacities.max().item():.2e}")
                print(f"  Scales: min={gaussians.scales.min().item():.2e}, max={gaussians.scales.max().item():.2e}")
                print(f"  Scales mean: {gaussians.scales.mean().item():.2e}")
                print("-" * 30 + "\n")
            
            # 覆蓋 means 為 RPC 計算的 ENU 座標
            gaussians.means = rearrange(means, "(b v r srf) xyz -> b v r srf () xyz", b=b, v=v, r=r, srf=srf).float()
            
            # ============================================================
            # [DEBUG] 將中間 25% 面積的 Gaussians 染成紅色 (驗證中心投影)
            # ============================================================
            DEBUG_CENTER_COLOR = False
            if DEBUG_CENTER_COLOR:
                # 取得所有 Gaussians 的 X, Y 座標
                means_flat = gaussians.means.view(-1, 3)
                x_coords, y_coords = means_flat[:, 0], means_flat[:, 1]
                
                # 計算範圍
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                x_range, y_range = x_max - x_min, y_max - y_min
                
                # 中間 25% 面積意味著 X, Y 各取中間 50% (0.5 * 0.5 = 0.25)
                # 也就是兩邊各留 25% 的邊際
                margin = 0.1
                is_center = (
                    (x_coords > x_min + x_range * margin) & 
                    (x_coords < x_max - x_range * margin) &
                    (y_coords > y_min + y_range * margin) &
                    (y_coords < y_max - y_range * margin)
                )
                
                # 修改 SH (只保留 0 階，且設為紅色)
                original_shape = gaussians.harmonics.shape
                harmonics_flat = gaussians.harmonics.view(-1, 3, original_shape[-1])
                
                # 設定所有 Gaussians 的高階 SH 均為 0 (只留 0 階 DC)
                if original_shape[-1] > 1:
                    harmonics_flat[:, :, 1:] = 0.0
                
                # 將中心區域設為純紅
                harmonics_flat[is_center, 0, 0] = 1.0  # R = 1
                harmonics_flat[is_center, 1, 0] = 0.0  # G = 0
                harmonics_flat[is_center, 2, 0] = 0.0  # B = 0
                
                gaussians.harmonics = harmonics_flat.view(original_shape)
                
                # 增加中心區域的 Opacity 確保看得到
                opacity_flat = gaussians.opacities.view(-1)
                opacity_flat[is_center] = 1.0
                gaussians.opacities = opacity_flat.view_as(gaussians.opacities)
                
                if global_step % 10 == 0:
                    center_count = is_center.sum().item()
                    total_count = is_center.numel()
                    print(f"\n🎯 [CENTER DEBUG] Step {global_step}")
                    print(f"   中心 Gaussians (紅): {center_count}/{total_count} ({center_count/total_count*100:.1f}%)")
                    print(f"   X Center Range: [{x_min + x_range*margin:.2f}, {x_max - x_range*margin:.2f}]")
                    print(f"   Y Center Range: [{y_min + y_range*margin:.2f}, {y_max - y_range*margin:.2f}]")
            
            
        else:
            # Original Pinhole branch (Unchanged)
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
                (h, w),
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )


        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        # The values are already in visualization_dump from the depth_predictor call
        visualization_dump["depth_lowres"] = visualization_dump.get("depth_lowres_raw")
        visualization_dump["depth_highres"] = visualization_dump.get("depth_highres_raw")

        gaussians_obj = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            rearrange(
                gaussians.scales,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.rotations,
                "b v r srf spp xyzw -> b (v r srf spp) xyzw",
            ),
        )
        
        # === [INFO] 印出最終 Gaussians 物件資訊 ===
        # total_gaussians = gaussians_obj.means.shape[1]
        # print(f"\n{'='*60}")
        # print(f"[FINAL GAUSSIANS OBJECT INFO - Encoder CostVolume]")
        # print(f"  Batch size: {gaussians_obj.means.shape[0]}")
        # print(f"  Total Gaussians per batch: {total_gaussians:,}")
        # print(f"{'='*60}")
        
        # --- Position (Means) 統計 ---
        means = gaussians_obj.means.detach()
        print(f"\n[POSITION / MEANS]")
        print(f"  Shape: {means.shape}")
        print(f"  X: min={means[..., 0].min().item():.4f}, max={means[..., 0].max().item():.4f}, mean={means[..., 0].mean().item():.4f}, std={means[..., 0].std().item():.4f}")
        print(f"  Y: min={means[..., 1].min().item():.4f}, max={means[..., 1].max().item():.4f}, mean={means[..., 1].mean().item():.4f}, std={means[..., 1].std().item():.4f}")
        print(f"  Z: min={means[..., 2].min().item():.4f}, max={means[..., 2].max().item():.4f}, mean={means[..., 2].mean().item():.4f}, std={means[..., 2].std().item():.4f}")
        
        # --- Scale 統計 ---
        scales = gaussians_obj.scales.detach()
        print(f"\n[SCALES]")
        print(f"  Shape: {scales.shape}")
        print(f"  Overall: min={scales.min().item():.6f}, max={scales.max().item():.6f}, mean={scales.mean().item():.6f}, std={scales.std().item():.6f}")
        print(f"  Scale X: min={scales[..., 0].min().item():.6f}, max={scales[..., 0].max().item():.6f}, mean={scales[..., 0].mean().item():.6f}")
        print(f"  Scale Y: min={scales[..., 1].min().item():.6f}, max={scales[..., 1].max().item():.6f}, mean={scales[..., 1].mean().item():.6f}")
        print(f"  Scale Z: min={scales[..., 2].min().item():.6f}, max={scales[..., 2].max().item():.6f}, mean={scales[..., 2].mean().item():.6f}")
        
        # --- Opacity 統計 ---
        opacities = gaussians_obj.opacities.detach()
        print(f"\n[OPACITIES]")
        print(f"  Shape: {opacities.shape}")
        print(f"  min={opacities.min().item():.6f}, max={opacities.max().item():.6f}, mean={opacities.mean().item():.6f}, std={opacities.std().item():.6f}")
        # 計算透明度分佈
        low_opacity = (opacities < 0.1).sum().item()
        mid_opacity = ((opacities >= 0.1) & (opacities < 0.5)).sum().item()
        high_opacity = (opacities >= 0.5).sum().item()
        print(f"  Distribution: <0.1: {low_opacity:,} ({100*low_opacity/opacities.numel():.1f}%), 0.1~0.5: {mid_opacity:,} ({100*mid_opacity/opacities.numel():.1f}%), >=0.5: {high_opacity:,} ({100*high_opacity/opacities.numel():.1f}%)")
        
        # --- Color (Spherical Harmonics) 統計 ---
        harmonics = gaussians_obj.harmonics.detach()
        print(f"\n[COLORS / SPHERICAL HARMONICS]")
        print(f"  Shape: {harmonics.shape}")
        # DC component (index 0) 代表基本顏色
        dc_component = harmonics[..., 0]  # [B, N, 3]
        print(f"  DC Component (Base Color):")
        print(f"    R: min={dc_component[..., 0].min().item():.4f}, max={dc_component[..., 0].max().item():.4f}, mean={dc_component[..., 0].mean().item():.4f}")
        print(f"    G: min={dc_component[..., 1].min().item():.4f}, max={dc_component[..., 1].max().item():.4f}, mean={dc_component[..., 1].mean().item():.4f}")
        print(f"    B: min={dc_component[..., 2].min().item():.4f}, max={dc_component[..., 2].max().item():.4f}, mean={dc_component[..., 2].mean().item():.4f}")
        
        # --- Rotation 統計 ---
        rotations = gaussians_obj.rotations.detach()
        print(f"\n[ROTATIONS (Quaternion)]")
        print(f"  Shape: {rotations.shape}")
        print(f"  Norm: min={rotations.norm(dim=-1).min().item():.6f}, max={rotations.norm(dim=-1).max().item():.6f}, mean={rotations.norm(dim=-1).mean().item():.6f}")
        
        # --- Covariance 統計 ---
        covariances = gaussians_obj.covariances.detach()
        print(f"\n[COVARIANCES]")
        print(f"  Shape: {covariances.shape}")
        print(f"  Overall: min={covariances.min().item():.6f}, max={covariances.max().item():.6f}, mean={covariances.mean().item():.6f}")
        
        print(f"\n{'='*60}\n")
        
        return gaussians_obj, visualization_dump

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
