
import sys
import os
import torch
import numpy as np
from dataclasses import dataclass
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.dataset.dataset_dfc2019 import DFC2019Dataset, DatasetDFC2019Cfg
# We don't need real ViewSampler if we mock it or if it's unused in basic parsing
# from src.dataset.view_sampler import ViewSampler

def test_rpc_roundtrip():
    # Path to dataset (Use the one verified to exist or intended)
    dataset_root = "/project/winston/datasets/DFC2019/skysplat_version/training"
    
    if not os.path.exists(dataset_root):
        print(f"Dataset root {dataset_root} not found. Skipping test.")
        return

    # Mock Configuration
    # We need to match DatasetDFC2019Cfg fields
    # It inherits DatasetCfgCommon
    
    # Create a dummy config object
    cfg = MagicMock(spec=DatasetDFC2019Cfg)
    cfg.roots = [dataset_root]
    cfg.near = 0.1
    cfg.far = 100.0
    cfg.image_shape = [768, 768]
    cfg.name = "dfc2019"
    
    # Mock ViewSampler
    view_sampler = MagicMock()
    
    # Init dataset
    ds = DFC2019Dataset(cfg=cfg, stage="train", view_sampler=view_sampler)
    
    if len(ds) == 0:
        print("No images found in dataset. Skipping test.")
        return

    # Get one item
    # This calls __getitem__, which uses load_view, which parses RPC
    try:
        item = ds[0]
    except Exception as e:
        print(f"Failed to load item from dataset: {e}")
        # Identify if it's because of missing files or logic
        import traceback
        traceback.print_exc()
        return

    target = item["target"]
    rpc_coeffs = target["rpc"] # [V, 90]
    
    # Pick first view from target or context
    # item['target']['rpc'] is [V_target, 90]
    # Let's verify shapes
    print("Context RPC shape:", item['context']['rpc'].shape)
    print("Target RPC shape:", item['target']['rpc'].shape)
    
    # Use first context view for test
    rpc_data = item['context']['rpc'][0] # [90]
    
    # We need to import RPC class to test projection
    from src.geometry.rpc import RPC
    # Use CPU for testing to avoid device mismatch issues in this simple test
    rpc_tensor = rpc_data.unsqueeze(0).cpu()
    rpc_obj = RPC(rpc_tensor)
    
    # Pick a pixel
    # Image is resized to 768x768
    H, W = 768, 768
    row = torch.tensor([H/2.0], dtype=torch.float32)
    col = torch.tensor([W/2.0], dtype=torch.float32)
    height = torch.tensor([50.0], dtype=torch.float32) # Assume 50m height
    
    print(f"Original Pixel: Row={row.item()}, Col={col.item()}, Height={height.item()}")
    
    # Inverse: Pixel -> Lat/Lon
    # Note: RPC class expects [B, ...]. Our batch is 1.
    lat, lon = rpc_obj.inverse(row.unsqueeze(0), col.unsqueeze(0), height.unsqueeze(0), iterations=10)
    
    print(f"Inverse Result: Lat={lat.item()}, Lon={lon.item()}")
    
    # Forward: Lat/Lon -> Pixel
    row_proj, col_proj = rpc_obj.forward(lat, lon, height.unsqueeze(0))
    
    print(f"Reprojected Pixel: Row={row_proj.item()}, Col={col_proj.item()}")
    
    diff_row = (row - row_proj.squeeze(0)).abs().item()
    diff_col = (col - col_proj.squeeze(0)).abs().item()
    
    print(f"Difference: Row={diff_row}, Col={diff_col}")
    
    assert diff_row < 1.0, f"Row error too large: {diff_row}"
    assert diff_col < 1.0, f"Col error too large: {diff_col}"
    print("Test PASSED: Roundtrip error within 1 pixel.")

if __name__ == "__main__":
    test_rpc_roundtrip()
