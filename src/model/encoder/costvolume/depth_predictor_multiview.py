import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import os
from PIL import Image
import numpy as np

from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel

from dataclasses import dataclass
from torch import Tensor

@dataclass
class EncoderOutput:
    raw_gaussians: Tensor
    densities: Tensor
    depths: Tensor

# --- [工具函數] 保持不變，僅加入 Log ---
def warp_with_pose_depth_candidates(feature1, intrinsics, pose, depth, warp_padding_mode="zeros"):
    # (保持原樣，這用於 Perspective 模式)
    B, D, H, W = depth.shape
    C = feature1.shape[1]
    device = feature1.device
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    x, y = x.float() + 0.5, y.float() + 0.5
    fx, fy = intrinsics[:, 0, 0].view(B, 1, 1), intrinsics[:, 1, 1].view(B, 1, 1)
    cx, cy = intrinsics[:, 0, 2].view(B, 1, 1), intrinsics[:, 1, 2].view(B, 1, 1)
    x_norm, y_norm = (x[None, ...] - cx) / fx, (y[None, ...] - cy) / fy
    x_norm, y_norm = x_norm.unsqueeze(1).expand(-1, D, -1, -1), y_norm.unsqueeze(1).expand(-1, D, -1, -1)
    X_ref, Y_ref, Z_ref = x_norm * depth, y_norm * depth, depth
    P_ref_homo = torch.stack([X_ref, Y_ref, Z_ref, torch.ones_like(Z_ref)], dim=-1)
    P_src_homo = torch.matmul(P_ref_homo, pose.transpose(1, 2).unsqueeze(1).unsqueeze(1))
    X_src, Y_src, Z_src = P_src_homo[..., 0], P_src_homo[..., 1], P_src_homo[..., 2]
    Z_src_safe = Z_src.clamp(min=1e-7)
    x_src_pix = (X_src / Z_src_safe) * fx.unsqueeze(1) + cx.unsqueeze(1)
    y_src_pix = (Y_src / Z_src_safe) * fy.unsqueeze(1) + cy.unsqueeze(1)
    H_src, W_src = feature1.shape[-2:]
    grid = torch.stack([2 * x_src_pix / (W_src - 1) - 1, 2 * y_src_pix / (H_src - 1) - 1], dim=-1)
    return F.grid_sample(feature1, grid.view(B, D*H, W, 2), mode="bilinear", padding_mode=warp_padding_mode, align_corners=True).view(B, C, D, H, W)

# def warp_with_rpc(feature1, rpc1, rpc0, depth, warp_padding_mode="zeros", scale_factor=4.0):

def warp_with_rpc(
    feature1,
    rpc1,        # source RPC (Src)
    rpc0,        # ref RPC (Ref)
    depth,       # [B, D, H, W] - 這裡傳入的是絕對海拔高度 (Altitude)
    warp_padding_mode="zeros",
    scale_factor=4.0, # 假設 TIF 是 256px, 特徵圖是 64px, 則 factor=4
):
    """
    使用 RPC 將 Source 視角特徵變換至 Reference 視角。
    包含自動化幾何檢查與參數驗證。
    """
    from src.geometry.rpc import RPC
    
    b, d, h, w = depth.size()
    c = feature1.size(1)
    device = feature1.device
    
    # --- [DEBUG] 參數驗證與幾何健康檢查 ---
    torch.set_printoptions(precision=8, sci_mode=False)

    # --- 1. 建立 Ref 圖像坐標網格 ---
    # 建立特徵圖尺度的網格 (0 ~ 63)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32), 
        torch.arange(w, device=device, dtype=torch.float32), 
        indexing="ij"
    )
    # 轉換到 RPC 定義域 (例如 0 ~ 255)
    # 尋找像素中心 (+0.5)
    u_ref = (x_grid + 0.5) * scale_factor
    v_ref = (y_grid + 0.5) * scale_factor
    
    # 擴展維度
    u_ref = u_ref.unsqueeze(0).unsqueeze(0).expand(b, d, h, w)
    v_ref = v_ref.unsqueeze(0).unsqueeze(0).expand(b, d, h, w)
    print("\n" + "="*30 + " [RPC WARP CHECK] " + "="*30)
    print(f"1. Scale Info: Feature Map {w}x{h} | Scale Factor: {scale_factor}")
    
    # 檢驗 Ref RPC 參數
    # Index 0: LINE_OFF (應該有大幅度位移，可能是負數)
    # Index 4: LAT_OFF (應該是全域值，例如 30.360...)
    print(f"2. Ref RPC Params (First 10):")
    print(f"   LINE_OFF (idx0): {rpc0[0, 0].item():.4f} <--- Check if shifted")
    print(f"   LAT_OFF  (idx4): {rpc0[0, 4].item():.10f} <--- Check global precision")
    
    # 檢驗 Height
    print(f"3. Depth/Height Plane (First 3): {depth[0, :3, 0, 0].flatten().cpu().numpy()}")
    print(f"3. Depth/Height Plane (last 3): {depth[0, -3:, 0, 0].flatten().cpu().numpy()}")
    # exit()
    # print(f"x_grid sample: {x_grid[0, :5]} | y_grid sample: {y_grid[:5, 0]}")
    # print(f"u_ref sample: {u_ref[0, 0, :5, 0]} | v_ref sample: {v_ref[0, 0, :5, 0]}")
    
    # hehe= input("warp_with_rpc function ")

    
    # --- 2. 執行幾何投影運算 ---
    rpc_ref_obj = RPC(rpc0)
    rpc_src_obj = RPC(rpc1)
    
    # [反投影] Ref (u,v,z) -> World (lat,lon)
    # 使用較高的迭代次數確保精度
    lat, lon = rpc_ref_obj.inverse(v_ref, u_ref, depth, iterations=50)

    # [正投影] World (lat,lon) -> Src (u,v)
    v_src_raw, u_src_raw = rpc_src_obj.forward(lat, lon, depth)

    # --- [DEBUG] 自我投影檢查 (Self-Projection Check) ---
    # 理論上：Ref -> World -> Ref 應該要變回原本的坐標
    # 這是檢驗 RPC 參數是否崩壞的最好方法
    with torch.no_grad():
        v_check, u_check = rpc_ref_obj.forward(lat, lon, depth)
        diff_u = torch.abs(u_check - u_ref)
        diff_v = torch.abs(v_check - v_ref)
        max_diff = torch.max(diff_u + diff_v).item()
        
        print(f"4. Self-Projection Error: {max_diff:.6f} pixels")
        if max_diff > 1.0:
            print("   [WARNING] 自我投影誤差過大！請檢查 RPC 參數或高度範圍。")
        else:
            print("   [OK] 自我投影幾何閉環成功。")

    # --- 3. 座標還原與歸一化 ---
    # 從 256px 空間轉回 64px 特徵空間
    u_src_feat = (u_src_raw / scale_factor) - 0.5
    v_src_feat = (v_src_raw / scale_factor) - 0.5

    # 歸一化至 [-1, 1] 供 grid_sample 使用
    # align_corners=True: -1 指向 index 0, +1 指向 index w-1
    u_norm = 2 * u_src_feat / (w - 1) - 1
    v_norm = 2 * v_src_feat / (h - 1) - 1
    
    # --- [DEBUG] 投影範圍檢查 ---
    if True:
        min_u, max_u = u_norm.min().item(), u_norm.max().item()
        print(f"5. Cross-Projection Result (u_norm):")
        print(f"   Range: [{min_u:.4f}, {max_u:.4f}]")
        if min_u < -2.0 or max_u > 2.0:
            print("   [WARNING] 投影範圍異常大！可能沒有對齊 (Misalignment)。")
        elif -1.5 <= min_u and max_u <= 1.5:
             print("   [OK] 投影範圍合理，幾何對齊正常。")
        print("="*60 + "\n")

    grid = torch.stack([u_norm, v_norm], dim=-1) # [B, D, H, W, 2]
    # 【關鍵修正】將 grid 轉回與 feature1 相同的型態 (通常是 float32)
    grid = grid.to(feature1.dtype) 

    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(b, c, d, h, w)

    return warped_feature


def prepare_feat_proj_data_lists(
    features, intrinsics, extrinsics, near, far, num_samples, rpcs=None # near far 這邊指的是distance 
):  

    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    rpc_curr_lists = [] # List of RPCs relative to the Ref view
    
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)
        # calculate reference pose
        cur_ref_pose_to_v0_list = []
        cur_rpc_pair_list = []
        
        for v0, v1 in zip(init_view_order, cur_view_order):
            cur_ref_pose_to_v0_list.append(
                extrinsics[:, v1].clone().detach().inverse()
                @ extrinsics[:, v0].clone().detach()
            )
            if rpcs is not None:
                # Store (RefRPC, SrcRPC)
                # rpcs is [B, V, 90]
                # We need [B, 90] for each
                ref_rpc = rpcs[:, v0]
                src_rpc = rpcs[:, v1]
                cur_rpc_pair_list.append((ref_rpc, src_rpc))
                
        cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
        pose_curr_lists.append(cur_ref_pose_to_v0s)
        
        if rpcs is not None:
            # cat list of tuples? No, list of (RPC_Ref_Batch, RPC_Src_Batch)
            # Actually we just appended tuples. We need to stack them.
            # stack ref: [vxB, 90]
            refs = torch.cat([x[0] for x in cur_rpc_pair_list], dim=0)
            srcs = torch.cat([x[1] for x in cur_rpc_pair_list], dim=0)
            rpc_curr_lists.append((refs, srcs))
    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # Universal Distance Candidate Generation [B, V, D]
    # In both Standard and RPC mode, we now use POSITIVE DISTANCES.
    dist_candi = (
        near.unsqueeze(-1) 
        + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).unsqueeze(0).to(near.device)
        * (far - near).unsqueeze(-1)
    )
    depth_candi_curr = rearrange(dist_candi, "b v d -> (v b) d")

    # [LOG] Track depth-to-height mapping for RPC mode
    if rpcs is not None:
        h_off = rpcs[:, :, 8] 
        # near = H - scene_range → H = near + scene_range (scene_range=20 hardcoded in encoder)
        # 或直接用 near+far 中點估計，這裡僅供 debug print
        h_near_plane = h_off + 20 # 最高的平面
        h_far_plane = h_off - 20   # 最低的平面
        
        print(f"\n[GEOMETRY LOG] prepare_feat_proj_data_lists")
        print(f"  > Ground Level (approx): {h_off[0,0].item():.1f}m MSL")
        print(f"  > Search Range (Dist): {near.min().item():.1f}m to {far.max().item():.1f}m")
        print(f"  > Altitude Range (approx): top={h_near_plane.mean().item():.1f}m, bottom={h_far_plane.mean().item():.1f}m")

    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr, rpc_curr_lists


class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=64,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        wo_depth_refine=False,
        wo_cost_volume=False,
        wo_cost_volume_refine=False,
        **kwargs,
    ):
        super(DepthPredictorMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        # ablation settings
        # Table 3: base
        self.wo_depth_refine = wo_depth_refine
        # Table 3: w/o cost volume
        self.wo_cost_volume = wo_cost_volume
        # Table 3: w/o U-Net
        self.wo_cost_volume_refine = wo_cost_volume_refine

        # Cost volume refinement: 2D U-Net
        input_channels = feature_channels if wo_cost_volume else (num_depth_candidates + feature_channels)
        channels = self.regressor_feat_dim
        if wo_cost_volume_refine:
            self.corr_project = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            modules = [
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1,
                    attention_resolutions=costvolume_unet_attn_res,
                    channel_mult=costvolume_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
                nn.Conv2d(channels, num_depth_candidates, 3, 1, 1)
            ]
            self.corr_refine_net = nn.Sequential(*modules)
            # cost volume u-net skip connection
            self.regressor_residual = nn.Conv2d(
                input_channels, num_depth_candidates, 1, 1, 0
            )

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )

        # CNN-based feature upsampler
        proj_in_channels = feature_channels + feature_channels
        upsample_out_channels = feature_channels
        self.upsampler = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=upscale_factor,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(
            upsample_out_channels, depth_unet_feat_dim, 3, 1, 1
        )

        # Depth refinement: 2D U-Net
        input_channels = 3 + depth_unet_feat_dim + 1 + 1
        channels = depth_unet_feat_dim
        if wo_depth_refine:  # for ablations
            self.refine_unet = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            self.refine_unet = nn.Sequential(
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1, 
                    attention_resolutions=depth_unet_attn_res,
                    channel_mult=depth_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
            )

        # Gaussians prediction: covariance, color
        gau_in = depth_unet_feat_dim + 3 + feature_channels
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
        if not wo_depth_refine:
            channels = depth_unet_feat_dim
            disps_models = [
                nn.Conv2d(channels, channels * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
            ]
            self.to_disparity = nn.Sequential(*disps_models)

    def forward(
        self,
        features,
        intrinsics,
        extrinsics,
        distance_near,
        distance_far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
        visualization_dump=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""

        # format the input
        b, v, c, h, w = features.shape
        rpcs = extra_info.get("rpcs", None) if extra_info else None
        
        # format the input
        b, v, c, h, w = features.shape
        if visualization_dump is not None:
             visualization_dump["features"] = features
        # ejgroegore = input(f"distance near far {distance_near}, {distance_far}")
        feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr, rpc_curr_lists = (
            prepare_feat_proj_data_lists(
                features,
                intrinsics,
                extrinsics,
                distance_near,
                distance_far,
                num_samples=self.num_depth_candidates,
                rpcs=rpcs
            )
        )
        if cnn_features is not None:
            cnn_features = rearrange(cnn_features, "b v ... -> (v b) ...")

        # cost volume constructions
        feat01 = feat_comb_lists[0]


        # If RPCs exist, we use rpc_curr_lists instead of pose_curr_lists
        if rpcs is not None:

            # Use RPC warping + Variance-based Cost Volume (SkySplat)
            
            # 1. Prepare Reference features tiled to [B, C, D, H, W]
            # feat01 is [vB, C, H, W] (but here v=1 usually relative to pair?)
            # Actually prepare_feat_proj_data_lists returns flattened v*B.
            # But logic iterates neighbors.
            # feat_comb_lists[0] is Ref.
            # Actually logic is: for each view, treat as Ref, use others as Src.
            # feat01 is the "current reference view features" for the batch.
            
            # Expand Ref to D
            # feat01: [vB, C, H, W] -> [vB, C, D, H, W]
            d = self.num_depth_candidates
            feat_ref_expanded = feat01.unsqueeze(2).expand(-1, -1, d, -1, -1)
            
            # Collect all features [Ref, Src1_Warped, Src2_Warped...]
            # We need to accumulate them.
            # However, the loop structure `for feat10, rpc_pair ...` iterates over *single* source views (or groups?).
            # MVSPlat `prepare` returns lists of SHIFTED views.
            # If we have V views, `feat_comb_lists` has V items.
            # Item 0: Ref. Item 1: Neighbors (Src). 
            # So we can just collect them.
            
            all_warped_features = [feat_ref_expanded]
            
            # 準備 per-(vB) 相機推導高度，供各 view pair 的 altitude 轉換使用
            # virtual_cam_dist_vb: [v*B]，順序與 (v b) 排列一致
            virtual_cam_dist_raw = extra_info.get("virtual_camera_distance", None) if extra_info else None
            if virtual_cam_dist_raw is not None:
                # virtual_cam_dist_raw: [B, V] -> rearrange to [v*B]
                virtual_cam_dist_vb = rearrange(virtual_cam_dist_raw, "b v -> (v b)")  # [v*B]
            else:
                virtual_cam_dist_vb = None
            
            for feat10, rpc_pair in zip(feat_comb_lists[1:], rpc_curr_lists):
                ref_rpc_data, src_rpc_data = rpc_pair
                print(f"\n[WARP DEBUG] Processing View Pair:")
                print(f"  > Ref View LINE_SCALE: {ref_rpc_data[0, 1].item():.1f}")
                print(f"  > Src View LINE_SCALE: {src_rpc_data[0, 1].item():.1f}")
                h_off = ref_rpc_data[:, 8:9].unsqueeze(-1).unsqueeze(-1)  # [vB, 1, 1, 1]
                # 使用 RPC Jacobian 推導的虛擬相機高度（非 hardcode 100m）
                if virtual_cam_dist_vb is not None:
                    # ref_rpc_data 的 batch 維度大小 == v*B
                    vb_size = ref_rpc_data.shape[0]
                    H_derived = virtual_cam_dist_vb[:vb_size].reshape(-1, 1, 1, 1).to(h_off.device)
                else:
                    H_derived = torch.full_like(h_off, 44.8)  # fallback ≈ 128×0.35m
                cam_msl = h_off + H_derived
                print(f"  > cam_msl (center): {cam_msl[0,0,0,0].item():.2f}m  (H_derived={H_derived[0,0,0,0].item():.2f}m)")

                # Convert distance -> altitude for warping
                height_candi = cam_msl - disp_candi_curr  # [vB, D, 1, 1]
                
                feat_src_warped = warp_with_rpc(
                    feat10,
                    src_rpc_data,
                    ref_rpc_data,
                    height_candi.repeat([1, 1, *feat10.shape[-2:]]),
                    warp_padding_mode="zeros",
                    scale_factor=self.upscale_factor,
                ) # [vB, C, D, H, W]
                all_warped_features.append(feat_src_warped)
            
            # 2. Compute Variance across Views，拿feature 去算變異數，越低代表特徵差越小，代表越可能是對的高度
            # Stack: [NumViews, vB, C, D, H, W]
            stack_features = torch.stack(all_warped_features, dim=0)
            
            volume_var = torch.var(stack_features, dim=0, unbiased=False) # [vB, C, D, H, W]
            
            # 3. Reduce Channels
            # SkySplat architecture implies volume has D channels for 2D Refinement.
            # So we mean over C.
            raw_correlation_in = torch.mean(volume_var, dim=1) # [vB, D, H, W]
            
            # 1. 找出 Variance 最小的索引 (ArgMin) -> [B, H, W]
            best_depth_idx = torch.argmin(raw_correlation_in, dim=1)
            # 2. 數據統計
            mid_h, mid_w = best_depth_idx.shape[1]//2, best_depth_idx.shape[2]//2
            center_idx = best_depth_idx[0, mid_h, mid_w].item()
            # 簡單推算海拔 (僅供 Debug 參考)
            # 底下這個if 只是拿來visualize height map 
            if rpc_curr_lists and len(rpc_curr_lists) > 0:
                ref_rpc_debug = rpc_curr_lists[0][0] 
                h_off_debug = ref_rpc_debug[0, 8].item()
                # 使用 extra_info 中的推導高度（若可用），否則 fallback
                if virtual_cam_dist_vb is not None:
                    cam_msl_debug = h_off_debug + virtual_cam_dist_vb[0].item()
                else:
                    cam_msl_debug = h_off_debug + 44.8  # fallback
                # 取得對應距離
                dists_debug = disp_candi_curr[0].detach().cpu()
                best_dist = dists_debug[center_idx].item()
                best_alt = cam_msl_debug - best_dist
                
                # ============================================================
                # [DEBUG] 儲存預測出的 Altitude (海拔高度)
                # ============================================================
                # 1. 取得整張圖的距離矩陣 (根據 best_depth_idx 查表)
                # best_depth_idx shape: [vB, H, W] -> 取第 0 個 view: [H, W]
                best_idx_map = best_depth_idx[0].detach().cpu()
                
                # 建立一個與影像大小相同的距離矩陣
                dist_map = torch.zeros_like(best_idx_map, dtype=torch.float32)
                for i in range(self.num_depth_candidates):
                    dist_map[best_idx_map == i] = dists_debug[i]
                
                # 2. 轉換為海拔高度 (Altitude = Camera_MSL - Distance)
                alt_map = cam_msl_debug - dist_map
                alt_map_np = alt_map.numpy()
                
                # 3. 準備存檔路徑
                save_dir = "/project/winston/mvsplat/outputs/debug_height_maps"
                os.makedirs(save_dir, exist_ok=True)
                
                # 4. 儲存為 .txt 檔案 (記錄原始數值，包含負數)
                txt_path = os.path.join(save_dir, "coarse_altitude.txt")
                np.savetxt(txt_path, alt_map_np, fmt="%.4f")
                
                # 5. 儲存為影像格式 (為了能存成影像，需要做簡單的 Min-Max 正規化到 0-255)
                # 雖然你說不需要操作，但影像格式 (PNG/JPG) 不支援負數浮點數
                # 所以這裡將其線性映射到 0-255 僅供視覺化，真實數值請看 .txt
                alt_min, alt_max = alt_map_np.min(), alt_map_np.max()
                if alt_max > alt_min:
                    norm_map = (alt_map_np - alt_min) / (alt_max - alt_min)
                else:
                    norm_map = np.zeros_like(alt_map_np)
                
                img_array = (norm_map * 255.0).astype(np.uint8)
                img_path = os.path.join(save_dir, "coarse_altitude.png")
                Image.fromarray(img_array).save(img_path)
                
                print(f"\n[DEBUG] Saved Coarse Altitude Map to {save_dir}")
                print(f"  > Range: [{alt_min:.2f}m, {alt_max:.2f}m]")
                # ============================================================
                # exit()

            # 把cost volume 跟 Ref feature 結合，讓 U-Net 可以同時看到兩者資訊
            # print("raw_correlation_in shape[before concat]:", raw_correlation_in.shape)
            # print("feat01 shape:", feat01.shape)
            raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)

        # refine cost volume via 2D u-net
        raw_correlation = self.corr_refine_net(raw_correlation_in)  # (vb d h w)
        # apply skip connection
        # print("raw_correlation_in shape:", raw_correlation_in.shape)
        # print("raw_correlation shape:", raw_correlation.shape)
        # exit()
        raw_correlation += self.regressor_residual(raw_correlation_in)

        # softmax to get coarse depth and density
        pdf = F.softmax(self.depth_head_lowres(raw_correlation), dim=1)  # [2xB, D, H, W]
        coarse_disps = (disp_candi_curr * pdf).sum(dim=1, keepdim=True)  # (vb, 1, h, w)
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax
        pdf_max = F.interpolate(pdf_max, scale_factor=self.upscale_factor)
        fullres_disps = F.interpolate(
            coarse_disps,
            scale_factor=self.upscale_factor,
            mode="bilinear",
            align_corners=True,
        )

        if visualization_dump is not None:
            visualization_dump["depth_lowres_raw"] = coarse_disps
            
        # depth refinement
        proj_feat_in_fullres = self.upsampler(torch.cat((feat01, cnn_features), dim=1))
        proj_feature = self.proj_feature(proj_feat_in_fullres)
        refine_out = self.refine_unet(torch.cat((extra_info["images"], proj_feature, fullres_disps, pdf_max), dim=1).float())

        # gaussians head
        raw_gaussians_in = [refine_out, extra_info["images"], proj_feat_in_fullres]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1).float()
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        raw_gaussians = rearrange(raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b)

        if self.wo_depth_refine:
            densities = repeat(pdf_max, "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1)
            # 【修正】: 這裡不應該取倒數，因為 fullres_disps 是 Distance
            depths = fullres_disps 
            depths = repeat(depths, "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1)
        else:
            delta_disps_density = self.to_disparity(refine_out)
            delta_disps, raw_densities = delta_disps_density.split(gaussians_per_pixel, dim=1)
            densities = repeat(F.sigmoid(raw_densities), "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1)
            # Refine Distance
            fine_disps = (fullres_disps + delta_disps).clamp(
                rearrange(distance_near, "b v -> (v b) () () ()"),
                rearrange(distance_far, "b v -> (v b) () () ()"),
            )
            # 【嚴重修正】: 移除 1.0 / fine_disps
            # fine_disps 已經是 Metric Distance (公尺)，直接當作 depth 使用
            depths = fine_disps 
            depths = repeat(depths, "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1)

        if visualization_dump is not None:
            # Reshape depth for visualization: [b, v, (h w), srf, dpt] -> [b, v, h, w, srf, dpt]
            # Use depths (high-res)
            visualization_dump["depth_highres_raw"] = fine_disps
            visualization_dump["depth"] = rearrange(
                depths, 
                "b v (h w) srf dpt -> b v h w srf dpt", 
                h=h*self.upscale_factor, 
                w=w*self.upscale_factor
            )
        return EncoderOutput(
            raw_gaussians,
            densities,
            depths,
        )

