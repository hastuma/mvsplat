
import sys
import torch
import numpy as np
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from dataset.dataset_dfc2019 import DFC2019Dataset
from geometry.rpc import RPC

def test_rpc_roundtrip():
    # Path to dataset
    dataset_root = "/project/winston/datasets/DFC2019/Track3-RGB-1/"
    
    # Init dataset
    ds = DFC2019Dataset(dataset_root, num_views=2)
    
    if len(ds) == 0:
        print("No images found. Skipping test.")
        return

    # Get one item
    item = ds[0]
    target = item["target"]
    rpc_coeffs = target["rpc"] # [V, 90]
    
    # Pick first view
    rpc_data = rpc_coeffs[0] # [90]
    rpc_obj = RPC(rpc_data.unsqueeze(0)) # Add batch dim -> [1, 90]
    
    # Pick a pixel
    # Image size 2048x2048 usually
    # Center pixel
    H, W = 2048, 2048 
    row = torch.tensor([1024.0], dtype=torch.float32)
    col = torch.tensor([1024.0], dtype=torch.float32)
    height = torch.tensor([50.0], dtype=torch.float32) # Assume 50m height
    
    print(f"Original Pixel: Row={row.item()}, Col={col.item()}, Height={height.item()}")
    
    # Inverse: Pixel -> Lat/Lon
    lat, lon = rpc_obj.inverse(row, col, height, iterations=10)
    
    print(f"Inverse Result: Lat={lat.item()}, Lon={lon.item()}")
    
    # Forward: Lat/Lon -> Pixel
    row_proj, col_proj = rpc_obj.forward(lat, lon, height)
    
    print(f"Reprojected Pixel: Row={row_proj.item()}, Col={col_proj.item()}")
    
    diff_row = (row - row_proj).abs().item()
    diff_col = (col - col_proj).abs().item()
    
    print(f"Difference: Row={diff_row}, Col={diff_col}")
    
    assert diff_row < 1.0, f"Row error too large: {diff_row}"
    assert diff_col < 1.0, f"Col error too large: {diff_col}"
    print("Test PASSED: Roundtrip error within 1 pixel.")

if __name__ == "__main__":
    test_rpc_roundtrip()
