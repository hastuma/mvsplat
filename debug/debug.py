import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from jaxtyping import install_import_hook
from PIL import Image
import numpy as np
import torchvision.utils as vutils

# Configure jaxtyping
# Note: we need to wrap the imports that use beartype/jaxtyping

# python /project/winston/mvsplat/debug/debug.py +experiment=dfc2019 data_loader.train.batch_size=1

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.model.encoder.costvolume.depth_predictor_multiview import warp_with_rpc, prepare_feat_proj_data_lists
    from src.misc.step_tracker import StepTracker



def save_tensor_as_image(tensor, path, title=""):
    """Helper to save a tensor as an image using torchvision/PIL."""
    # tensor: [C, H, W] or [H, W] or [B, C, H, W]
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    if tensor.ndim == 2:
        # If [H, W], add a channel dimension for vutils
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[0] > 3:
        # If [C, H, W] and C > 3 (e.g. features), average over channels
        tensor = tensor.mean(dim=0, keepdim=True)
    
    # Normalize to [0, 1] for visualization
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max > t_min:
        tensor = (tensor - t_min) / (t_max - t_min)
    
    vutils.save_image(tensor, path)
    print(f"Saved visualization to {path}")

@hydra.main(version_base=None, config_path="../config", config_name="main")
def debug_cost_volume(cfg_dict: DictConfig):
    # 1. Setup Config
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Setup Data
    print("Step 2: Loading Data...")
    step_tracker = StepTracker()
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker)
    data_module.setup("train")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    # Push batch to device
    def to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: to_device(v) for k, v in obj.items()}
        return obj
    
    batch = to_device(batch)
    print(f"Loaded batch for scene: {batch['scene']}")

    # 3. Setup Model
    print("Step 3: Setup Model...")
    encoder, _ = get_encoder(cfg.model.encoder)
    encoder = encoder.to(device)
    encoder.eval()
    
    # 4. Extract Features
    print("Step 4: Running Backbone Feature Extraction...")
    with torch.no_grad():
        trans_features, cnn_features = encoder.backbone(
            batch["context"]["image"],
            attn_splits=cfg.model.encoder.multiview_trans_attn_split,
            return_cnn_features=True,
        )
    print(f"Features shape: {trans_features.shape}") # [B, V, C, H_feat, W_feat]

    # 5. Prepare for Warping
    print("Step 5: Preparing for Warping...")
    depth_predictor = encoder.depth_predictor
    rpcs = batch["context"].get("rpc", None)
    
    # prepare_feat_proj_data_lists handles coordinate transformations
    feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr, rpc_curr_lists = (
        prepare_feat_proj_data_lists(
            trans_features,
            batch["context"]["intrinsics"],
            batch["context"]["extrinsics"],
            batch["context"]["near"],
            batch["context"]["far"],
            num_samples=cfg.model.encoder.num_depth_candidates,
            rpcs=rpcs
        )
    )

    # 6. Debug Warping (Manual check for first view pair)
    print("Step 6: Debugging Warping...")
    feat_ref = feat_comb_lists[0] # [vB, C, H, W]
    feat_src = feat_comb_lists[1]
    ref_rpc, src_rpc = rpc_curr_lists[0]
    
    # Let's take a specific depth plane (e.g., middle one)
    mid_idx = cfg.model.encoder.num_depth_candidates // 2
    # disp_candi_curr is already [vB, D, 1, 1] from prepare_feat_proj_data_lists
    h_planes = disp_candi_curr[:, mid_idx:mid_idx+1].repeat([1, 1, *feat_ref.shape[-2:]])
    
    print(f"Warping at height plane {mid_idx} (Height: {disp_candi_curr[0, mid_idx].item()})...")
    with torch.no_grad():
        warped_feat = warp_with_rpc(
            feat_src,
            src_rpc,
            ref_rpc,
            h_planes,
            scale_factor=depth_predictor.upscale_factor
        )
    
    # Visualization
    debug_dir = Path("debug/outputs")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    save_tensor_as_image(feat_ref[0], debug_dir / "feat_ref.png")
    save_tensor_as_image(feat_src[0], debug_dir / "feat_src.png")
    save_tensor_as_image(warped_feat[0, :, 0], debug_dir / "feat_warped.png")
    
    # Correlation/Difference
    diff = torch.abs(feat_ref[0] - warped_feat[0, :, 0])
    save_tensor_as_image(diff, debug_dir / "feat_diff.png")

    # 7. Run Full Variance Volume
    print("Step 7: Building Full Variance Volume...")
    d = cfg.model.encoder.num_depth_candidates
    feat_ref_expanded = feat_ref.unsqueeze(2).expand(-1, -1, d, -1, -1)
    all_warped = [feat_ref_expanded]
    
    # Prepare depth tensor for all channels
    depth_grids = disp_candi_curr.repeat([1, 1, *feat_ref.shape[-2:]])
    
    for feat10, rpc_pair in zip(feat_comb_lists[1:], rpc_curr_lists):
        ref_rpc_i, src_rpc_i = rpc_pair
        with torch.no_grad():
            w_feat = warp_with_rpc(
                feat10,
                src_rpc_i,
                ref_rpc_i,
                depth_grids,
                scale_factor=depth_predictor.upscale_factor
            )
            all_warped.append(w_feat)
    
    stack_features = torch.stack(all_warped, dim=0)
    volume_var = torch.var(stack_features, dim=0, unbiased=False)
    cost_volume = torch.mean(volume_var, dim=1) # [vB, D, H, W]
    print(f"Cost Volume Shape: {cost_volume.shape}")
    
    # Save a slice of cost volume
    save_tensor_as_image(cost_volume[0, mid_idx], debug_dir / "cost_volume_slice.png")

    # 8. Height Map Prediction
    print("Step 8: Predicting Height Map...")
    # Matches DepthPredictorMultiView logic
    raw_correlation_in = torch.cat((cost_volume, feat_ref), dim=1)
    with torch.no_grad():
        if depth_predictor.wo_cost_volume_refine:
            raw_correlation = depth_predictor.corr_project(raw_correlation_in)
        else:
            raw_correlation = depth_predictor.corr_refine_net(raw_correlation_in)
            raw_correlation = raw_correlation + depth_predictor.regressor_residual(raw_correlation_in)
            
        # Passing through depth_head_lowres
        logits = depth_predictor.depth_head_lowres(raw_correlation)
        pdf = torch.nn.functional.softmax(logits, dim=1)
        
        # disp_candi_curr is [vB, D, 1, 1]
        prediction = (disp_candi_curr * pdf).sum(dim=1, keepdim=True)
    
    print(f"Predicted Height Map Shape: {prediction.shape}")
    save_tensor_as_image(prediction[0, 0], debug_dir / "predicted_height_map.png")

    # 9. Visualizing RGB Warping (The User Request)
    print("Step 9: Visualizing RGB Warping with Predicted Geometry...")
    # Get original RGB images [B, V, 3, H, W]
    imgs = batch["context"]["image"] # [B, V, C, H, W]
    
    # Slice to Batch Size 1 for visualization
    # We take the first sample in the batch
    ref_img_rgb = imgs[0:1, 0] # [1, C, H, W]
    src_img_rgb = imgs[0:1, 1] # [1, C, H, W]
    
    # Slice RPCs as well! ref_rpc was [B*Pairs, 90] or similar.
    # We just want the RPC for the first sample's first pair.
    ref_rpc_viz = ref_rpc[0:1]
    src_rpc_viz = src_rpc[0:1]
    
    # Resize prediction to Image Resolution
    # prediction was [vB, 1, H, W]. Take first one.
    pred_up = torch.nn.functional.interpolate(
        prediction[0:1], 
        size=ref_img_rgb.shape[-2:], 
        mode='bilinear', 
        align_corners=False
    ) # [1, 1, H_img, W_img]
    
    depth_rgb = pred_up # [1, 1, H, W]
    
    with torch.no_grad():
        warped_rgb = warp_with_rpc(
            src_img_rgb, 
            src_rpc_viz,     
            ref_rpc_viz,     
            depth_rgb,   
            scale_factor=1.0 
        ) # Output: [B, C, 1, H, W]
        
    warped_rgb = warped_rgb.squeeze(2) # [1, C, H, W]
    
    save_tensor_as_image(ref_img_rgb[0], debug_dir / "rgb_ref.png")
    save_tensor_as_image(warped_rgb[0], debug_dir / "rgb_src_warped.png")
    
    # Stack them for easy comparison: Ref | Warped | Diff
    # Diff needs to be 3 channels
    diff_rgb = torch.abs(ref_img_rgb - warped_rgb)
    
    combined = torch.cat([ref_img_rgb, warped_rgb, diff_rgb], dim=3) # Concat along Width
    save_tensor_as_image(combined[0], debug_dir / "rgb_warp_comparison.png")
    print(f"Saved RGB comparison to {debug_dir / 'rgb_warp_comparison.png'}")
    
    print("\nDebug Script Completed Successfully!")
    print(f"Check results in: {debug_dir.absolute()}")

if __name__ == "__main__":
    debug_cost_volume()
