import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel






def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W] Source feature
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4] T_ref_to_src
    depth: [B, D, H, W] Depth in Ref
    """
    B, D, H, W = depth.shape
    C = feature1.shape[1]
    device = feature1.device

    # 1. Back-project Ref to 3D CameraRef
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    x = x.float() + 0.5
    y = y.float() + 0.5
    
    # intrinsics: fx, fy, cx, cy
    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)
    
    x_norm = (x[None, ...] - cx) / fx # [B, H, W]
    y_norm = (y[None, ...] - cy) / fy # [B, H, W]
    
    # Expand to D
    x_norm = x_norm.unsqueeze(1).expand(-1, D, -1, -1) # [B, D, H, W]
    y_norm = y_norm.unsqueeze(1).expand(-1, D, -1, -1)
    
    # P_cam = Z * [x_norm, y_norm, 1]
    X_ref = x_norm * depth
    Y_ref = y_norm * depth
    Z_ref = depth
    
    # Homogeneous
    ones = torch.ones_like(Z_ref)
    P_ref_homo = torch.stack([X_ref, Y_ref, Z_ref, ones], dim=-1) # [B, D, H, W, 4]
    
    # 2. Transform to Src
    # P_src = P_ref @ pose.T
    P_src_homo = torch.matmul(P_ref_homo, pose.transpose(1, 2).unsqueeze(1).unsqueeze(1)) 
    
    X_src = P_src_homo[..., 0]
    Y_src = P_src_homo[..., 1]
    Z_src = P_src_homo[..., 2]
    
    # 3. Project to Src Pixels
    eps = 1e-7
    Z_src_safe = Z_src.clamp(min=eps) 
    
    x_src_pix = (X_src / Z_src_safe) * fx.unsqueeze(1) + cx.unsqueeze(1)
    y_src_pix = (Y_src / Z_src_safe) * fy.unsqueeze(1) + cy.unsqueeze(1)
    
    # 4. Sample
    H_src, W_src = feature1.shape[-2:]
    
    x_norm = 2 * x_src_pix / (W_src - 1) - 1
    y_norm = 2 * y_src_pix / (H_src - 1) - 1
    
    grid = torch.stack([x_norm, y_norm], dim=-1) # [B, D, H, W, 2]
    
    warped = F.grid_sample(
        feature1,
        grid.view(B, D*H, W, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True
    )
    
    return warped.view(B, C, D, H, W)


def warp_with_rpc(
    feature1,
    rpc1, # source RPC
    rpc0, # ref RPC
    depth, # [B, D, H, W] - actually we use height planes for RPC
    warp_padding_mode="zeros",
):
    """
    Warp feature1 (source) to feature0 (ref) viewpoint using RPCs.
    depth here represents absolute height (altitude) planes.
    """
    from src.geometry.rpc import RPC
    
    b, d, h, w = depth.size()
    c = feature1.size(1)
    
    # 1. Create grid for Ref image (0)
    # We want to find (lat, lon) for every pixel (u, v) in Ref at height H
    
    # Grid (u, v) for Ref
    device = feature1.device
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32), 
        torch.arange(w, device=device, dtype=torch.float32), 
        indexing="ij"
    ) # [H, W]
    
    # Flatten
    u_ref = x_grid.unsqueeze(0).unsqueeze(0).expand(b, d, h, w) # [B, D, H, W] Col
    v_ref = y_grid.unsqueeze(0).unsqueeze(0).expand(b, d, h, w) # [B, D, H, W] Row
    
    # Height planes
    # depth tensor passed in is expected to be height values (check prepare_feat_proj_data_lists)
    H_planes = depth 
    
    # 2. Inverse RPC on Ref: (u, v, h) -> (lat, lon)
    # We need to flatten to call RPC methods efficiently or modify RPC to handle B,D,H,W
    # RPC class handles [..., 90]. 
    
    # Instantiate RPC objects
    rpc_ref_obj = RPC(rpc0)
    rpc_src_obj = RPC(rpc1)
    
    # Inverse Projection (Ref View)
    # This is expensive. For cost volume, we usually do this once per batch/view pair?
    # Or simplified.
    lat, lon = rpc_ref_obj.inverse(v_ref, u_ref, H_planes, iterations=3)

    # 3. Forward RPC on Src: (lat, lon, h) -> (u_src, v_src)
    v_src, u_src = rpc_src_obj.forward(lat, lon, H_planes)
    
    # 4. Sample
    # Normalize (u_src, v_src) to [-1, 1] for grid_sample
    # Note: feature1 is [B, C, H, W], so size is (W, H)
    
    u_norm = 2 * u_src / (w - 1) - 1
    v_norm = 2 * v_src / (h - 1) - 1
    
    grid = torch.stack([u_norm, v_norm], dim=-1) # [B, D, H, W, 2]
    
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(b, c, d, h, w)
    
    return warped_feature


def prepare_feat_proj_data_lists(
    features, intrinsics, extrinsics, near, far, num_samples, rpcs=None
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    rpc_curr_lists = [] # List of RPCs relative to the Ref view
    
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    
    # Prepare RPC lists in (v b) format matching the features
    # Ref view is always the first one in the pair logic below
    
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        if rpcs is not None:
             # Pairing: Ref is view 'v0', Src is view 'v1'
             # We want to pair RPCs. 
             # Output needs to be aligned with (v * b)
             # Structure here is iterating through neighbor shifts.
             # For each shift 'idx', we have a list of pairs (v0, v1).
             pass

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
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

    
    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper
    if v == 2:
        pose_ref = extrinsics[:, 0].clone().detach()
        pose_tgt = extrinsics[:, 1].clone().detach()
        pose = pose_tgt.inverse() @ pose_ref
        pose_curr_lists = [torch.cat((pose, pose.inverse()), dim=0),]
        
        if rpcs is not None:
            # Pair 1: Ref=0, Src=1
            r0 = rpcs[:, 0]
            r1 = rpcs[:, 1]
            
            # Pair 2: Ref=1, Src=0
            # We concat them along batch dimension to match (v*b)
            refs = torch.cat((r0, r1), dim=0)
            srcs = torch.cat((r1, r0), dim=0)
            rpc_curr_lists = [(refs, srcs)]

    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth) [v*b, d]
    if rpcs is None:
        # Standard MVS: Inverse Depth
        min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
        max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
        depth_candi_curr = (
            min_depth
            + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
            * (max_depth - min_depth)
        ).type_as(features)
    else:
        # RPC Mode: Height Planes (meters) - Linear spacing usually
        # near/far in dataset should be acting as min_height/max_height
        # We assume dataset provided near/far as min/max height in meters.
        min_height = rearrange(near.clone().detach(), "b v -> (v b) 1") # Height min
        max_height = rearrange(far.clone().detach(), "b v -> (v b) 1")  # Height max
        
        depth_candi_curr = (
            min_height
            + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_height.device)
            * (max_height - min_height)
        ).type_as(features)

    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr, rpc_curr_lists


class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
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
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""

        # format the input
        b, v, c, h, w = features.shape
        rpcs = extra_info.get("rpcs", None) if extra_info else None
        
        # format the input
        b, v, c, h, w = features.shape
        feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr, rpc_curr_lists = (
            prepare_feat_proj_data_lists(
                features,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=self.num_depth_candidates,
                rpcs=rpcs
            )
        )
        if cnn_features is not None:
            cnn_features = rearrange(cnn_features, "b v ... -> (v b) ...")

        # cost volume constructions
        feat01 = feat_comb_lists[0]
        if self.wo_cost_volume:
            raw_correlation_in = feat01
        else:
            
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
                
                for feat10, rpc_pair in zip(feat_comb_lists[1:], rpc_curr_lists):
                    ref_rpc, src_rpc = rpc_pair
                    # feat10 is Src features [vB, C, H, W]
                    
                    feat_src_warped = warp_with_rpc(
                        feat10,
                        src_rpc,
                        ref_rpc,
                        disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                        warp_padding_mode="zeros"
                    ) # [vB, C, D, H, W]
                    
                    all_warped_features.append(feat_src_warped)
                
                # 2. Compute Variance across Views
                # Stack: [NumViews, vB, C, D, H, W]
                stack_features = torch.stack(all_warped_features, dim=0)
                
                # Var = Mean(X^2) - Mean(X)^2
                # shape [vB, C, D, H, W]
                # We compute var over dim 0 (views)
                # var = torch.var(stack_features, dim=0, unbiased=False) 
                # (unbiased=False matches simple Mean Square Diff for 2 views)
                
                volume_var = torch.var(stack_features, dim=0, unbiased=False) # [vB, C, D, H, W]
                
                # 3. Reduce Channels
                # SkySplat architecture implies volume has D channels for 2D Refinement.
                # So we mean over C.
                raw_correlation_in = torch.mean(volume_var, dim=1) # [vB, D, H, W]
                
                # MVSPlat `raw_correlation_in` is usually [vB, D, H, W] (correlation score).
                # This matches.
                    
            else:
                # Use Standard Perspective Warping
                raw_correlation_in_lists = []
                for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
                    # sample feat01 from feat10 via camera projection
                    feat01_warped = warp_with_pose_depth_candidates(
                        feat10,
                        intr_curr,
                        pose_curr,
                        1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                        warp_padding_mode="zeros",
                    )  # [B, C, D, H, W]
                    # calculate similarity
                    raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(
                        1
                    ) / (
                        c**0.5
                    )  # [vB, D, H, W]
                    raw_correlation_in_lists.append(raw_correlation_in)
                    
                # average all cost volumes
                # average all cost volumes
                raw_correlation_in = torch.mean(
                    torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
                )  # [vxb d, h, w]
            
            raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)

        # refine cost volume via 2D u-net
        if self.wo_cost_volume_refine:
            raw_correlation = self.corr_project(raw_correlation_in)
        else:
            raw_correlation = self.corr_refine_net(raw_correlation_in)  # (vb d h w)
            # apply skip connection
            raw_correlation = raw_correlation + self.regressor_residual(
                raw_correlation_in
            )

        # softmax to get coarse depth and density
        pdf = F.softmax(
            self.depth_head_lowres(raw_correlation), dim=1
        )  # [2xB, D, H, W]
        coarse_disps = (disp_candi_curr * pdf).sum(
            dim=1, keepdim=True
        )  # (vb, 1, h, w)
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax
        pdf_max = F.interpolate(pdf_max, scale_factor=self.upscale_factor)
        fullres_disps = F.interpolate(
            coarse_disps,
            scale_factor=self.upscale_factor,
            mode="bilinear",
            align_corners=True,
        )

        # depth refinement
        proj_feat_in_fullres = self.upsampler(torch.cat((feat01, cnn_features), dim=1))
        proj_feature = self.proj_feature(proj_feat_in_fullres)
        refine_out = self.refine_unet(torch.cat(
            (extra_info["images"], proj_feature, fullres_disps, pdf_max), dim=1
        ))

        # gaussians head
        raw_gaussians_in = [refine_out,
                            extra_info["images"], proj_feat_in_fullres]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        raw_gaussians = rearrange(
            raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
        )

        if self.wo_depth_refine:
            densities = repeat(
                pdf_max,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
            depths = 1.0 / fullres_disps
            depths = repeat(
                depths,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
        else:
            # delta fine depth and density
            delta_disps_density = self.to_disparity(refine_out)
            delta_disps, raw_densities = delta_disps_density.split(
                gaussians_per_pixel, dim=1
            )

            # combine coarse and fine info and match shape
            densities = repeat(
                F.sigmoid(raw_densities),
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )

            fine_disps = (fullres_disps + delta_disps).clamp(
                1.0 / rearrange(far, "b v -> (v b) () () ()"),
                1.0 / rearrange(near, "b v -> (v b) () () ()"),
            )
            depths = 1.0 / fine_disps
            depths = repeat(
                depths,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )

        return depths, densities, raw_gaussians
