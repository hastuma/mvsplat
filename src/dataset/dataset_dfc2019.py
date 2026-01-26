
import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from osgeo import gdal
from typing import Literal, Any
from dataclasses import dataclass

from .types import UnbatchedExample, UnbatchedViews
from .view_sampler import ViewSampler
from .dataset import DatasetCfgCommon

@dataclass
class DatasetDFC2019Cfg(DatasetCfgCommon):
    name: Literal["dfc2019"]
    roots: list[str]
    near: float
    far: float

class DFC2019Dataset(Dataset):
    def __init__(
        self,
        cfg: DatasetDFC2019Cfg,
        stage: Literal["train", "val", "test"],
        view_sampler: ViewSampler,
    ):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        
        # Use roots from config
        self.root_dir = Path(cfg.roots[0]) # For now handle single root
        
        # Map stage to folder names
        stage_map = {
            "train": "training",
            "val": "validation",
            "test": "validation" # Fallback
        }
        folder_name = stage_map.get(stage, "training")
        self.stage_dir = self.root_dir / folder_name
        
        if not self.stage_dir.exists():
             # Fallback or error
             print(f"Warning: {self.stage_dir} does not exist. trying root directly")
             self.stage_dir = self.root_dir

        # Identify all scene directories
        self.scenes = sorted([d for d in self.stage_dir.iterdir() if d.is_dir()])
        
        if len(self.scenes) == 0:
             print(f"Warning: No scenes found in {self.stage_dir}")

    def __len__(self):
        # One epoch = visiting each scene once (random sampling within scene)
        return len(self.scenes)
    
    def _parse_rpc(self, ds) -> np.ndarray:
        # Get RPC metadata
        md = ds.GetMetadata('RPC')
        if not md:
            # Fallback or zeros?
            return np.zeros(90, dtype=np.float32)

        # Keys we expect
        keys = [
            'LINE_OFF', 'LINE_SCALE', 'SAMP_OFF', 'SAMP_SCALE',
            'LAT_OFF', 'LAT_SCALE', 'LONG_OFF', 'LONG_SCALE',
            'HEIGHT_OFF', 'HEIGHT_SCALE'
        ]
        coeffs = []
        for k in keys:
            coeffs.append(float(md.get(k, 0)))
            
        # Vectors
        for prefix in ['LINE_NUM_COEFF', 'LINE_DEN_COEFF', 'SAMP_NUM_COEFF', 'SAMP_DEN_COEFF']:
            # The value is a string with space separated numbers
            val_str = md.get(prefix, "")
            vals = [float(x) for x in val_str.split()]
            # Ensure it is 20
            if len(vals) != 20:
                vals = vals + [0]*(20-len(vals))
            coeffs.extend(vals)
            
        return np.array(coeffs, dtype=np.float32)

    def __getitem__(self, index):
        scene_dir = self.scenes[index]
        files = sorted(list(scene_dir.glob("*.tif")))
        
        if len(files) < 4:
            # Need at least 3 context + 1 target
            # Fallback to random other scene or error?
            # recursive retry
            return self.__getitem__((index + 1) % len(self))
            
        # SkySplat Sparse Input Condition: 3 input views
        num_context = 3
        num_target = 1 # For training loss, usually 1 target view per step? Or more?
                       # User said "render image... then calc loss with same path other photos".
                       # MVSplat usually targets 1 view for memory reasons.
        
        # Randomly select context and target
        # For validation, we might want fixed split, but here we do random for training
        if self.stage == 'train':
            perm = torch.randperm(len(files))
            context_indices = perm[:num_context]
            target_indices = perm[num_context:num_context+num_target]
        else:
            # Deterministic for val
            # Just take first 3 as context, next 1 as target
            context_indices = torch.arange(num_context)
            target_indices = torch.arange(num_context, num_context+num_target)

        # Helper to load
        def load_view(idx):
            fpath = files[idx]
            # Load Image
            ds = gdal.Open(str(fpath))
            if ds is None:
                raise IOError(f"Could not open {fpath}")
            
            width = ds.RasterXSize
            height = ds.RasterYSize
            
            # Read bands (assuming RGB)
            # DFC2019 might be uint8 or uint16.
            # MVSplat expects float [0,1] tensors
            arr = ds.ReadAsArray() # [C, H, W]
            if arr.shape[0] > 3:
                arr = arr[:3] # RGB
            
            # Normalize
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            elif arr.dtype == np.uint16:
                arr = arr.astype(np.float32) / 65535.0 # Or specific max
            
            tensor_img = torch.from_numpy(arr)
            
            # Parse RPC
            rpc_coeff = self._parse_rpc(ds)
            rpc_tensor = torch.from_numpy(rpc_coeff)
            
            return tensor_img, rpc_tensor

        context_images, context_rpcs = [], []
        target_images, target_rpcs = [], []
        
        for idx in context_indices:
            img, rpc = load_view(idx.item())
            context_images.append(img)
            context_rpcs.append(rpc)
            
        for idx in target_indices:
            img, rpc = load_view(idx.item())
            target_images.append(img)
            target_rpcs.append(rpc)
            
        images = context_images + target_images
        rpcs = context_rpcs + target_rpcs
        
        images = torch.stack(images) # [V, C, H, W]
        rpcs = torch.stack(rpcs)     # [V, 90]
        
        # Resize to 768x768 (SkySplat)
        target_h, target_w = 768, 768
        orig_h, orig_w = images.shape[-2], images.shape[-1]
        
        if orig_h != target_h or orig_w != target_w:
            import torch.nn.functional as F
            images = F.interpolate(images, size=(target_h, target_w), mode='area')
            
            # Adjust RPCs
            # Scale factors
            scale_h = target_h / float(orig_h)
            scale_w = target_w / float(orig_w)
            
            # rpcs is [V, 90]
            # 0: LINE_OFF, 1: LINE_SCALE
            rpcs[:, 0] *= scale_h
            rpcs[:, 1] *= scale_h
            
            # 2: SAMP_OFF, 3: SAMP_SCALE
            rpcs[:, 2] *= scale_w
            rpcs[:, 3] *= scale_w
            
        # Dummy extrinsics/intrinsics/near/far
        V, C, H, W = images.shape
        extrinsics = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(V, 1, 1)
        near = torch.ones(V) * -50.0  
        far = torch.ones(V) * 300.0   
        
        indices = torch.arange(V) # Dummy indices relative to this batch

        # Construct views
        # Split back to context and target
        # Indices in 'images' stack: 0..2 are context, 3 is target
        
        context_views = []
        for i in range(num_context):
            context_views.append({
                "extrinsics": extrinsics[i],
                "intrinsics": intrinsics[i],
                "image": images[i],
                "near": near[i],
                "far": far[i],
                "index": indices[i],
                "rpc": rpcs[i]
            })
            
        target_views = []
        for i in range(num_context, V):
            target_views.append({
                "extrinsics": extrinsics[i],
                "intrinsics": intrinsics[i],
                "image": images[i],
                "near": near[i],
                "far": far[i],
                "index": indices[i],
                "rpc": rpcs[i]
            })

        return {
            "context": {
                "extrinsics": torch.stack([v["extrinsics"] for v in context_views]),
                "intrinsics": torch.stack([v["intrinsics"] for v in context_views]),
                "image": torch.stack([v["image"] for v in context_views]),
                "near": torch.stack([v["near"] for v in context_views]),
                "far": torch.stack([v["far"] for v in context_views]),
                "index": torch.stack([v["index"] for v in context_views]),
                "rpc": torch.stack([v["rpc"] for v in context_views]),
            },
            "target": {
                "extrinsics": torch.stack([v["extrinsics"] for v in target_views]),
                "intrinsics": torch.stack([v["intrinsics"] for v in target_views]),
                "image": torch.stack([v["image"] for v in target_views]),
                "near": torch.stack([v["near"] for v in target_views]),
                "far": torch.stack([v["far"] for v in target_views]),
                "index": torch.stack([v["index"] for v in target_views]),
                "rpc": torch.stack([v["rpc"] for v in target_views]),
            },
            "scene": scene_dir.name,
        }
