
import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from osgeo import gdal
from typing import Literal, Any
from dataclasses import dataclass
import json
from .types import UnbatchedExample, UnbatchedViews
from .view_sampler import ViewSampler
from .dataset import DatasetCfgCommon
# 請確保路徑跟你訓練程式碼裡寫的一模一樣

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
        self.enu_origin_dir = self.root_dir / "enu_origin" 
        if not self.stage_dir.exists():
             self.stage_dir = self.root_dir

        # Identify all scene directories
        self.scenes = sorted([d for d in self.stage_dir.iterdir() if d.is_dir()])
        
        if len(self.scenes) == 0:
             print(f"Warning: No scenes found in {self.stage_dir}")

    def __len__(self):
        # One epoch = visiting each scene once (random sampling within scene)
        return len(self.scenes)
    def _parse_rpc(self, ds) -> np.ndarray:
        # 使用 float64 以保證精度
        keys = ['LINE_OFF', 'LINE_SCALE', 'SAMP_OFF', 'SAMP_SCALE',
                'LAT_OFF', 'LAT_SCALE', 'LONG_OFF', 'LONG_SCALE',
                'HEIGHT_OFF', 'HEIGHT_SCALE']
        coeffs = []
        for k in keys:
            # 強制從 Metadata 讀取我們寫入的數值
            val = ds.GetMetadataItem(k, 'RPC') 
            if val is None: val = ds.GetMetadata('RPC').get(k, 0)
            coeffs.append(float(val))
        for prefix in ['LINE_NUM_COEFF', 'LINE_DEN_COEFF', 'SAMP_NUM_COEFF', 'SAMP_DEN_COEFF']:
            val_str = ds.GetMetadataItem(prefix, 'RPC')
            if val_str is None: val_str = ds.GetMetadata('RPC').get(prefix, "")
            vals = [float(x) for x in val_str.split()]
            coeffs.extend(vals if len(vals) == 20 else vals + [0.0]*(20-len(vals)))
            
        return np.array(coeffs, dtype=np.float64)

    def __getitem__(self, index):
        scene_dir = self.scenes[index]
        files = sorted(list(scene_dir.glob("*.tif")))#4張會拿前三張當context，最後一張當target
        if len(files) < 4:
            # Need at least 3 context + 1 target
            return self.__getitem__((index + 1) % len(self))
            
        # SkySplat Sparse Input Condition: 3 input views
        num_context = 3
        num_target = 1 # For training loss, usually 1 target view per step? Or more?
                       # User said "render image... then calc loss with same path other photos".
                       # MVSplat usually targets 1 view for memory reasons.
        
        # Randomly select context and target
        # For validation, we might want fixed split, but here we do random for training
        if self.stage == 'train':
            # perm = torch.randperm(len(files))
            perm = torch.arange(len(files))
            context_indices = perm[:num_context]
            target_indices = perm[num_context:num_context+num_target]
        else:
            # Deterministic for val
            # Just take first 3 as context, next 1 as target
            context_indices = torch.arange(num_context)
            target_indices = torch.arange(num_context, num_context+num_target)

        # 只讀取一次 enu_origin (同一 scene 的所有 view 都相同)
        scene = "_".join(files[0].name.split("_")[:2])
        enu_origin_path = self.enu_origin_dir / f"{scene}.json"
        with open(enu_origin_path, "r") as f:
            enu_data = json.load(f) #[30.357821655028108, -81.70643392476823, -47.180099999999996]

        # 讀取每張圖的真實裁切起點（由 geographic_cropping.py 生成）
        crop_offsets_path = scene_dir / "crop_offsets.json"
        if crop_offsets_path.exists():
            with open(crop_offsets_path, "r") as f:
                crop_offsets = json.load(f)
        else:
            crop_offsets = {}
        
        enu_origin_tensor = torch.tensor(
            [enu_data[0], enu_data[1], enu_data[2]],
            dtype=torch.float64
       )
        print(f"Loaded ENU origin for scene {scene}: {enu_origin_tensor}")
        # Helper to load
        def load_view(idx):
            fpath = files[idx]
            ds = gdal.Open(str(fpath))
            orig_w, orig_h = ds.RasterXSize, ds.RasterYSize
            target_h, target_w = self.cfg.image_shape
            
            # 讀取影像並轉為 float32 (影像不需要 float64)
            arr = ds.ReadAsArray(buf_xsize=target_w, buf_ysize=target_h)
            if arr.shape[0] > 3: arr = arr[:3]
            
            if arr.dtype == np.uint8:
                tensor_img = torch.from_numpy(arr).float() / 255.0
            else:
                tensor_img = torch.from_numpy(arr).float() / 65535.0
            
            # 解析 RPC (精確度關鍵)
            rpc_coeff = self._parse_rpc(ds)
            
            # 轉回 Tensor (必須保持 float64 以避免精度丟失)
            rpc_tensor = torch.from_numpy(rpc_coeff).double()
            return tensor_img, rpc_tensor

        context_images, context_rpcs = [], []
        target_images, target_rpcs = [], []
        
        for idx in context_indices:
            img, rpc = load_view(idx.item())
            context_images.append(img)
            context_rpcs.append(rpc)
        
        for idx in target_indices:
            # 這邊的scene_dir 會是像”/project/winston/datasets/DFC2019/geo_cropped/training/JAX_004_012_p0406“
            # 所以讀file時要先讀主圖，也就是JAX_004_012 , 
            print(f"Processing target view: {files[idx]}")
            img, rpc = load_view(idx.item())
            target_images.append(img)
            target_rpcs.append(rpc)
        
        images = context_images + target_images
        rpcs = context_rpcs + target_rpcs
        
        images = torch.stack(images) # [V, C, H, W]
        rpcs = torch.stack(rpcs)     # [V, 90]
        
        # 複製 enu_origin 給所有 view (同一 scene 的所有 view 都相同)
        V, C, H, W = images.shape
        enu_origins = enu_origin_tensor.unsqueeze(0).repeat(V, 1)  # [V, 3]
        
        # Images are already resized in load_view
        
        # Define dimensions and placeholders
        extrinsics = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(V, 1, 1)
        
        # Use RPC Heights for building range: offset -20 to +50 meters
        ref_h_off = rpcs[0, 8].item() # HEIGHT_OFF is at index 8 of DFC2019 RPC
        near = torch.ones(V) * (ref_h_off - 20.0)
        far = torch.ones(V) * (ref_h_off + 50.0)
        indices = torch.arange(V)
        # Construct views
        # Split back to context and target
        # Indices in 'images' stack: 0..2 are context, 3 is target
        
        context_views = []

        # 提取 patch 座標（所有 view 都在同一個 scene_dir，px/py 相同）
        _folder = os.path.basename(scene_dir)
        _patch = _folder.split("_")[-1]  # e.g. "p0406"
        _patch_py = int(_patch[1:3])
        _patch_px = int(_patch[3:5])

        for i in range(num_context):
            ctx_image_name = os.path.splitext(os.path.basename(files[context_indices[i].item()]))[0]
            ctx_offsets = crop_offsets.get(ctx_image_name, {})
            context_views.append({
                "extrinsics": extrinsics[i],
                "intrinsics": intrinsics[i],
                "image": images[i],
                "near": near[i],
                "far": far[i],
                "index": indices[i],
                "rpc": rpcs[i],
                "enu_origin": enu_origins[i],
                "image_name": ctx_image_name,
                "px": _patch_px,
                "py": _patch_py,
                "col_start": ctx_offsets.get("col_start", _patch_px * 256),
                "row_start": ctx_offsets.get("row_start", _patch_py * 256),
            })
            print(f"Context view {i}: image_name={ctx_image_name}\ncol_start={ctx_offsets.get('col_start', _patch_px * 256)}, row_start={ctx_offsets.get('row_start', _patch_py * 256)}")
        target_views = []
        for i in range(num_context, V):
            image_name = os.path.splitext(os.path.basename(files[i]))[0]
            folder = os.path.basename(os.path.dirname(files[i]))# folder = JAX_004_012_p0406 或 JAX_004_012_p0406_topleft
            # 提取 patch 字符串，處理可能存在的 _topleft 或 _center 後綴
            folder_parts = folder.split("_")
            if folder_parts[-1] in ["topleft", "center"]:
                patch = folder_parts[-2]
            else:
                patch = folder_parts[-1]
            py = int(patch[1:3])  # 04 -> 4
            px = int(patch[3:5])  # 06 -> 6
            tgt_offsets = crop_offsets.get(image_name, {})
            target_views.append({
                "extrinsics": extrinsics[i],
                "intrinsics": intrinsics[i],
                "image": images[i],
                "near": near[i],
                "far": far[i],
                "index": indices[i],
                "rpc": rpcs[i],
                "enu_origin": enu_origins[i],
                "image_name": image_name,
                "px": px,
                "py": py,
                "col_start": tgt_offsets.get("col_start", px * 256),
                "row_start": tgt_offsets.get("row_start", py * 256),
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
                "enu_origin": torch.stack([v["enu_origin"] for v in context_views]),
                "image_name": [v["image_name"] for v in context_views],
                "px": [v["px"] for v in context_views],
                "py": [v["py"] for v in context_views],
                "col_start": [v["col_start"] for v in context_views],
                "row_start": [v["row_start"] for v in context_views],
            },
            "target": {
                "extrinsics": torch.stack([v["extrinsics"] for v in target_views]),
                "intrinsics": torch.stack([v["intrinsics"] for v in target_views]),
                "image": torch.stack([v["image"] for v in target_views]),
                "near": torch.stack([v["near"] for v in target_views]),
                "far": torch.stack([v["far"] for v in target_views]),
                "index": torch.stack([v["index"] for v in target_views]),
                "rpc": torch.stack([v["rpc"] for v in target_views]),
                "enu_origin": torch.stack([v["enu_origin"] for v in target_views]),
                "image_name": [v["image_name"] for v in target_views],
                "px": [v["px"] for v in target_views],
                "py": [v["py"] for v in target_views],
                "col_start": [v["col_start"] for v in target_views],
                "row_start": [v["row_start"] for v in target_views],
            },
            "scene": scene_dir.name,
        }
