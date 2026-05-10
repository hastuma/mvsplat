
import os
import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from osgeo import gdal
from typing import Literal, Optional
from dataclasses import dataclass, field
import json
from .types import UnbatchedExample, UnbatchedViews
from .view_sampler import ViewSampler
from .dataset import DatasetCfgCommon


# ---------------------------------------------------------------------------
# Numpy-based RPC helpers (CPU, used during DataLoader preprocessing)
# Mirror the math in datasets/DFC2019/geographic_cropping.py
# ---------------------------------------------------------------------------

def _poly(c, x, y, z):
    return (c[0] + c[1]*x + c[2]*y + c[3]*z
            + c[4]*x*y + c[5]*x*z + c[6]*y*z
            + c[7]*x*x + c[8]*y*y + c[9]*z*z
            + c[10]*x*y*z
            + c[11]*x**3 + c[12]*x*y**2 + c[13]*x*z**2
            + c[14]*x**2*y + c[15]*y**3 + c[16]*y*z**2
            + c[17]*x**2*z + c[18]*y**2*z + c[19]*z**3)


def _rpc_forward(rpc, lat, lon, height):
    """RPC forward: (lat, lon, height) → (row, col) in the original image."""
    p = (lat    - rpc[4]) / rpc[5]
    l = (lon    - rpc[6]) / rpc[7]
    h = (height - rpc[8]) / rpc[9]
    row = _poly(rpc[10:30], l, p, h) / _poly(rpc[30:50], l, p, h) * rpc[1] + rpc[0]
    col = _poly(rpc[50:70], l, p, h) / _poly(rpc[70:90], l, p, h) * rpc[3] + rpc[2]
    return row, col


def _rpc_inverse(rpc, row, col, height, iters=50):
    """RPC inverse: (row, col, height) → (lat, lon) via Newton iteration."""
    lat, lon = rpc[4], rpc[6]
    for _ in range(iters):
        r_est, c_est = _rpc_forward(rpc, lat, lon, height)
        delta = 1e-5
        r_dlat, c_dlat = _rpc_forward(rpc, lat + delta, lon, height)
        r_dlon, c_dlon = _rpc_forward(rpc, lat, lon + delta, height)
        dr_dlat = (r_dlat - r_est) / delta
        dc_dlat = (c_dlat - c_est) / delta
        dr_dlon = (r_dlon - r_est) / delta
        dc_dlon = (c_dlon - c_est) / delta
        det = dr_dlat * dc_dlon - dr_dlon * dc_dlat
        if abs(det) < 1e-15:
            break
        dlat = (dc_dlon * (row - r_est) - dr_dlon * (col - c_est)) / det
        dlon = (dr_dlat * (col - c_est) - dc_dlat * (row - r_est)) / det
        lat += dlat
        lon += dlon
    return lat, lon


def _read_patch(ds, col_start, row_start, size=256):
    """Read a size×size crop from a GDAL dataset, zero-padding if out-of-bounds."""
    W, H = ds.RasterXSize, ds.RasterYSize
    x1, y1 = int(col_start), int(row_start)
    x2, y2 = x1 + size, y1 + size
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(W, x2), min(H, y2)
    rw, rh = rx2 - rx1, ry2 - ry1

    gdal_type = ds.GetRasterBand(1).DataType
    np_type = np.uint8 if gdal_type == gdal.GDT_Byte else np.float32
    out = np.zeros((ds.RasterCount, size, size), dtype=np_type)

    if rw > 0 and rh > 0:
        data = ds.ReadAsArray(rx1, ry1, rw, rh)
        if data.ndim == 2:
            data = data[None]
        out_x1, out_y1 = rx1 - x1, ry1 - y1
        out[:, out_y1:out_y1+rh, out_x1:out_x1+rw] = data
    return out


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

@dataclass
class DatasetDFC2019Cfg(DatasetCfgCommon):
    name: Literal["dfc2019"]
    roots: list[str]
    near: float
    far: float
    # Path to full-resolution images for online random crop (train only).
    # When non-empty, training uses online geo-aligned random crops instead of
    # pre-cropped patches. Val/test always use pre-cropped patches from roots[0].
    raw_scenes_dir: str = ""
    # Number of random crops per scene per epoch in online crop mode.
    online_crops_per_scene: int = 50


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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

        self.root_dir = Path(cfg.roots[0])
        self.enu_origin_dir = self.root_dir / "enu_origin"

        # ----------------------------------------------------------------
        # Online crop mode: active only when raw_scenes_dir is set AND training
        # ----------------------------------------------------------------
        self.use_online_crop = bool(cfg.raw_scenes_dir) and stage == "train"

        if self.use_online_crop:
            self._init_online(Path(cfg.raw_scenes_dir))
        else:
            self._init_precropped()

    # ------------------------------------------------------------------
    # Init: pre-cropped mode (val / test / train without raw_scenes_dir)
    # ------------------------------------------------------------------

    def _init_precropped(self):
        stage_map = {"train": "training", "val": "validation", "test": "validation"}
        stage_dir = self.root_dir / stage_map.get(self.stage, "training")
        if not stage_dir.exists():
            stage_dir = self.root_dir
        self.scenes = sorted(
            [stage_dir / e.name for e in os.scandir(stage_dir) if e.is_dir(follow_symlinks=False)]
        )
        if not self.scenes:
            print(f"Warning: No scenes found in {stage_dir}")

    # ------------------------------------------------------------------
    # Init: online crop mode
    # ------------------------------------------------------------------

    def _init_online(self, raw_dir: Path, col_axis_angle_thresh: float = 45.0):
        # Group images by scene prefix (e.g., "JAX_068")
        scene_groups: dict[str, list[Path]] = {}
        n_filtered = 0
        for f in sorted(raw_dir.glob("*.tif")):
            parts = f.stem.split("_")
            if len(parts) < 2:
                continue
            scene_id = f"{parts[0]}_{parts[1]}"   # e.g. "JAX_068"

            # Filter out heavily rotated images (column axis > 45° from East).
            # Such images (e.g. JAX_068_001, column axis ~+87° ≈ pointing North)
            # have an orthogonal pixel grid relative to normal east-up images,
            # which breaks cost-volume feature correlation when mixed in the same
            # training batch.
            angle = self._col_axis_angle_from_east(f)
            if abs(angle) > col_axis_angle_thresh:
                n_filtered += 1
                continue

            scene_groups.setdefault(scene_id, []).append(f)

        if n_filtered:
            print(f"[OnlineCrop] Filtered {n_filtered} rotated images (|col_axis| > {col_axis_angle_thresh}°)")

        # Keep scenes with ≥4 images
        self.scene_groups = {k: v for k, v in scene_groups.items() if len(v) >= 4}
        self.scene_ids = sorted(self.scene_groups.keys())

        # Respect overfit_to_scene: restrict to a single scene if specified.
        # Set dataset.overfit_to_scene: JAX_068 in config to enable.
        # Set to null (or omit) to use all scenes.
        overfit = getattr(self.cfg, "overfit_to_scene", None)
        if overfit:
            if overfit not in self.scene_groups:
                raise RuntimeError(
                    f"[OnlineCrop] overfit_to_scene='{overfit}' not found. "
                    f"Available: {self.scene_ids}"
                )
            self.scene_groups = {overfit: self.scene_groups[overfit]}
            self.scene_ids = [overfit]

        # Pre-load ENU origins; auto-compute from image center for missing ones
        self.enu_origins: dict[str, list[float]] = {}
        for scene_id in self.scene_ids:
            enu_path = self.enu_origin_dir / f"{scene_id}.json"
            if enu_path.exists():
                with open(enu_path) as f:
                    self.enu_origins[scene_id] = json.load(f)
            else:
                self.enu_origins[scene_id] = self._compute_enu_origin(
                    self.scene_groups[scene_id][0]
                )

        if not self.scene_ids:
            raise RuntimeError(f"No valid scenes (≥4 images) found in {raw_dir}")
        print(
            f"[OnlineCrop] {len(self.scene_ids)} scenes, "
            f"{sum(len(v) for v in self.scene_groups.values())} total images"
        )

    def _col_axis_angle_from_east(self, img_path: Path) -> float:
        """Return the angle (degrees) of the image column axis w.r.t. East.

        Normal (north-up) images → ~0°.  Rotated-90° images → ~±90°.
        Uses the RPC Jacobian at the image center to measure orientation.
        """
        ds = gdal.Open(str(img_path))
        rpc = self._parse_rpc(ds)
        lat0, lon0, h0 = rpc[4], rpc[6], rpc[8]

        d = 1e-5
        r0,  c0  = _rpc_forward(rpc, lat0,     lon0,     h0)
        r_dlat, c_dlat = _rpc_forward(rpc, lat0 + d, lon0,     h0)
        r_dlon, c_dlon = _rpc_forward(rpc, lat0,     lon0 + d, h0)

        dr_dlat = (r_dlat - r0) / d;  dc_dlat = (c_dlat - c0) / d
        dr_dlon = (r_dlon - r0) / d;  dc_dlon = (c_dlon - c0) / d

        # Invert 2×2 Jacobian to find (dlat, dlon) per column step
        det = dr_dlat * dc_dlon - dr_dlon * dc_dlat
        if abs(det) < 1e-12:
            return 0.0
        dlat_per_col = -dr_dlon / det
        dlon_per_col =  dr_dlat / det
        return float(np.degrees(np.arctan2(dlat_per_col, dlon_per_col)))

    def _compute_enu_origin(self, img_path: Path) -> list[float]:
        """Derive ENU origin from the center pixel of an image via RPC inverse."""
        ds = gdal.Open(str(img_path))
        rpc = self._parse_rpc(ds)
        cx, cy = ds.RasterXSize / 2.0, ds.RasterYSize / 2.0
        h_ref = rpc[8]  # HEIGHT_OFF
        lat, lon = _rpc_inverse(rpc, cy, cx, h_ref)
        return [float(lat), float(lon), float(h_ref)]

    # ------------------------------------------------------------------
    # RPC parsing (shared)
    # ------------------------------------------------------------------

    def _parse_rpc(self, ds) -> np.ndarray:
        keys = [
            "LINE_OFF", "LINE_SCALE", "SAMP_OFF", "SAMP_SCALE",
            "LAT_OFF", "LAT_SCALE", "LONG_OFF", "LONG_SCALE",
            "HEIGHT_OFF", "HEIGHT_SCALE",
        ]
        coeffs = []
        for k in keys:
            val = ds.GetMetadataItem(k, "RPC")
            if val is None:
                val = ds.GetMetadata("RPC").get(k, 0)
            coeffs.append(float(val))

        for prefix in ["LINE_NUM_COEFF", "LINE_DEN_COEFF", "SAMP_NUM_COEFF", "SAMP_DEN_COEFF"]:
            val_str = ds.GetMetadataItem(prefix, "RPC")
            if val_str is None:
                val_str = ds.GetMetadata("RPC").get(prefix, "")
            vals = [float(x) for x in val_str.split()]
            coeffs.extend(vals if len(vals) == 20 else vals + [0.0] * (20 - len(vals)))

        return np.array(coeffs, dtype=np.float64)

    # ------------------------------------------------------------------
    # __len__ and __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        if self.use_online_crop:
            return len(self.scene_ids) * self.cfg.online_crops_per_scene
        return len(self.scenes)

    def __getitem__(self, index):
        if self.use_online_crop:
            return self._getitem_online(index)
        return self._getitem_precropped(index)

    # ------------------------------------------------------------------
    # Online crop __getitem__
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_overlap_ratio(col_s: int, row_s: int, img_w: int, img_h: int, tile: int) -> float:
        """Fraction of the tile [col_s, col_s+tile) × [row_s, row_s+tile) that
        falls inside the image [0, img_w) × [0, img_h).  Returns 0.0–1.0."""
        overlap_x = max(0, min(col_s + tile, img_w) - max(col_s, 0))
        overlap_y = max(0, min(row_s + tile, img_h) - max(row_s, 0))
        return (overlap_x * overlap_y) / (tile * tile)

    def _getitem_online(self, index, _depth=0):
        tile_size = self.cfg.image_shape[0]  # e.g. 256
        min_overlap = 0.8   # slave crop must cover ≥80% of tile area
        max_retries = 10    # re-sample master crop position if slaves fall OOB

        # Pick scene (round-robin, then random within scene)
        scene_id = self.scene_ids[index % len(self.scene_ids)]
        images = self.scene_groups[scene_id]

        # Random 4-image selection: 1 master + 3 slaves
        perm = random.sample(range(len(images)), min(4, len(images)))
        selected = [images[i] for i in perm]
        master_path = selected[0]

        # Open master image
        ds_m = gdal.Open(str(master_path))
        W, H = ds_m.RasterXSize, ds_m.RasterYSize
        rpc_m = self._parse_rpc(ds_m)
        h_ground = rpc_m[8]  # HEIGHT_OFF as reference altitude

        if W < tile_size or H < tile_size:
            return self._getitem_online((index + 1) % len(self))

        # Pre-open slave datasets once (reused across retries)
        slave_ds   = [gdal.Open(str(p)) for p in selected[1:]]
        slave_rpcs = [self._parse_rpc(ds) for ds in slave_ds]

        # ---- Retry loop: resample master crop if any slave is mostly OOB ----
        for attempt in range(max_retries):
            x = random.randint(0, W - tile_size)   # guaranteed: x + tile_size ≤ W
            y = random.randint(0, H - tile_size)   # guaranteed: y + tile_size ≤ H

            lat_c, lon_c = _rpc_inverse(rpc_m, float(y), float(x), h_ground)

            # Check all slave crops before doing any heavy reads
            slave_positions = []
            all_valid = True
            for ds_s, rpc_s in zip(slave_ds, slave_rpcs):
                row_f, col_f = _rpc_forward(rpc_s, lat_c, lon_c, h_ground)
                col_s = int(round(col_f))
                row_s = int(round(row_f))
                ratio = self._crop_overlap_ratio(
                    col_s, row_s, ds_s.RasterXSize, ds_s.RasterYSize, tile_size
                )
                if ratio < min_overlap:
                    all_valid = False
                    break
                slave_positions.append((col_s, row_s))

            if all_valid:
                break
        else:
            # All retries exhausted — pick a completely different index
            if _depth < 5:
                return self._getitem_online(
                    random.randint(0, len(self) - 1), _depth=_depth + 1
                )
            # Last resort: fall back to pre-cropped mode for this index
            return self._getitem_precropped(index % max(len(self.scenes), 1))

        enu_data = self.enu_origins[scene_id]

        # ---- Crop each image (master first, then slaves) ----
        view_images, view_rpcs, col_starts, row_starts, image_names = [], [], [], [], []

        # Master
        arr = _read_patch(ds_m, x, y, size=tile_size)
        if arr.shape[0] > 3:
            arr = arr[:3]
        img_tensor = (torch.from_numpy(arr).float() / 255.0
                      if arr.dtype == np.uint8
                      else torch.from_numpy(arr).float() / 65535.0)
        rpc_new = rpc_m.copy()
        rpc_new[0] -= y   # LINE_OFF
        rpc_new[2] -= x   # SAMP_OFF
        view_images.append(img_tensor)
        view_rpcs.append(torch.from_numpy(rpc_new).double())
        col_starts.append(x)
        row_starts.append(y)
        image_names.append(master_path.stem)

        # Slaves
        for ds_s, rpc_s, (col_s, row_s) in zip(slave_ds, slave_rpcs, slave_positions):
            arr = _read_patch(ds_s, col_s, row_s, size=tile_size)
            if arr.shape[0] > 3:
                arr = arr[:3]
            img_tensor = (torch.from_numpy(arr).float() / 255.0
                          if arr.dtype == np.uint8
                          else torch.from_numpy(arr).float() / 65535.0)
            rpc_new = rpc_s.copy()
            rpc_new[0] -= row_s   # LINE_OFF
            rpc_new[2] -= col_s   # SAMP_OFF
            view_images.append(img_tensor)
            view_rpcs.append(torch.from_numpy(rpc_new).double())
            col_starts.append(col_s)
            row_starts.append(row_s)
            image_names.append(selected[len(view_images) - 1].stem)

        # ---- Assemble output ----
        num_context = 3
        num_target = 1
        V = num_context + num_target

        images_t = torch.stack(view_images)      # [V, 3, H, W]
        rpcs_t   = torch.stack(view_rpcs)        # [V, 90]

        enu_origin_tensor = torch.tensor(enu_data, dtype=torch.float64)
        enu_origins_t = enu_origin_tensor.unsqueeze(0).expand(V, -1)  # [V, 3]

        ref_h_off = float(rpcs_t[0, 8].item())
        near_t = torch.full((V,), ref_h_off - 20.0)
        far_t  = torch.full((V,), ref_h_off + 50.0)
        extrinsics = torch.eye(4).unsqueeze(0).expand(V, -1, -1)
        intrinsics = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
        indices    = torch.arange(V)

        def _make_views(start, count):
            views = []
            for i in range(start, start + count):
                cs, rs = col_starts[i], row_starts[i]
                views.append({
                    "extrinsics": extrinsics[i],
                    "intrinsics": intrinsics[i],
                    "image":      images_t[i],
                    "near":       near_t[i],
                    "far":        far_t[i],
                    "index":      indices[i],
                    "rpc":        rpcs_t[i],
                    "enu_origin": enu_origins_t[i],
                    "image_name": image_names[i],
                    "px":         cs // tile_size,
                    "py":         rs // tile_size,
                    "col_start":  cs,
                    "row_start":  rs,
                })
            return views

        ctx_views = _make_views(0, num_context)
        tgt_views = _make_views(num_context, num_target)

        def _stack(views, key):
            return torch.stack([v[key] for v in views])

        def _list(views, key):
            return [v[key] for v in views]

        return {
            "context": {
                "extrinsics":  _stack(ctx_views, "extrinsics"),
                "intrinsics":  _stack(ctx_views, "intrinsics"),
                "image":       _stack(ctx_views, "image"),
                "near":        _stack(ctx_views, "near"),
                "far":         _stack(ctx_views, "far"),
                "index":       _stack(ctx_views, "index"),
                "rpc":         _stack(ctx_views, "rpc"),
                "enu_origin":  _stack(ctx_views, "enu_origin"),
                "image_name":  _list(ctx_views, "image_name"),
                "px":          _list(ctx_views, "px"),
                "py":          _list(ctx_views, "py"),
                "col_start":   _list(ctx_views, "col_start"),
                "row_start":   _list(ctx_views, "row_start"),
            },
            "target": {
                "extrinsics":  _stack(tgt_views, "extrinsics"),
                "intrinsics":  _stack(tgt_views, "intrinsics"),
                "image":       _stack(tgt_views, "image"),
                "near":        _stack(tgt_views, "near"),
                "far":         _stack(tgt_views, "far"),
                "index":       _stack(tgt_views, "index"),
                "rpc":         _stack(tgt_views, "rpc"),
                "enu_origin":  _stack(tgt_views, "enu_origin"),
                "image_name":  _list(tgt_views, "image_name"),
                "px":          _list(tgt_views, "px"),
                "py":          _list(tgt_views, "py"),
                "col_start":   _list(tgt_views, "col_start"),
                "row_start":   _list(tgt_views, "row_start"),
            },
            "scene": scene_id,
        }

    # ------------------------------------------------------------------
    # Pre-cropped __getitem__ (unchanged from original)
    # ------------------------------------------------------------------

    def _getitem_precropped(self, index):
        scene_dir = self.scenes[index]
        files = sorted([
            scene_dir / e.name
            for e in os.scandir(scene_dir)
            if e.name.endswith(".tif") and e.is_file(follow_symlinks=False)
        ])
        if len(files) < 4:
            return self._getitem_precropped((index + 1) % len(self.scenes))

        num_context = 3
        num_target = 1

        if self.stage == "train":
            perm = torch.arange(len(files))
            context_indices = perm[:num_context]
            target_indices  = perm[num_context:num_context + num_target]
        else:
            context_indices = torch.arange(num_context)
            target_indices  = torch.arange(num_context, num_context + num_target)

        scene = "_".join(files[0].name.split("_")[:2])
        enu_origin_path = self.enu_origin_dir / f"{scene}.json"
        with open(enu_origin_path) as f:
            enu_data = json.load(f)

        crop_offsets_path = scene_dir / "crop_offsets.json"
        crop_offsets = {}
        if crop_offsets_path.exists():
            with open(crop_offsets_path) as f:
                crop_offsets = json.load(f)

        enu_origin_tensor = torch.tensor(
            [enu_data[0], enu_data[1], enu_data[2]], dtype=torch.float64
        )

        def load_view(idx):
            fpath = files[idx]
            ds = gdal.Open(str(fpath))
            target_h, target_w = self.cfg.image_shape
            arr = ds.ReadAsArray(buf_xsize=target_w, buf_ysize=target_h)
            if arr.shape[0] > 3:
                arr = arr[:3]
            if arr.dtype == np.uint8:
                tensor_img = torch.from_numpy(arr).float() / 255.0
            else:
                tensor_img = torch.from_numpy(arr).float() / 65535.0
            rpc_coeff = self._parse_rpc(ds)
            rpc_tensor = torch.from_numpy(rpc_coeff).double()
            return tensor_img, rpc_tensor

        context_images, context_rpcs = [], []
        target_images,  target_rpcs  = [], []

        for idx in context_indices:
            img, rpc = load_view(idx.item())
            context_images.append(img)
            context_rpcs.append(rpc)
        for idx in target_indices:
            img, rpc = load_view(idx.item())
            target_images.append(img)
            target_rpcs.append(rpc)

        images = torch.stack(context_images + target_images)   # [V, C, H, W]
        rpcs   = torch.stack(context_rpcs   + target_rpcs)     # [V, 90]

        V = num_context + num_target
        enu_origins = enu_origin_tensor.unsqueeze(0).expand(V, -1)
        ref_h_off   = rpcs[0, 8].item()
        near = torch.full((V,), ref_h_off - 20.0)
        far  = torch.full((V,), ref_h_off + 50.0)
        extrinsics = torch.eye(4).unsqueeze(0).expand(V, -1, -1)
        intrinsics = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
        indices    = torch.arange(V)

        _folder = os.path.basename(scene_dir)
        _patch  = _folder.split("_")[-1]
        _patch_py = int(_patch[1:3])
        _patch_px = int(_patch[3:5])

        context_views = []
        for i in range(num_context):
            ctx_image_name = os.path.splitext(os.path.basename(files[context_indices[i].item()]))[0]
            ctx_offsets = crop_offsets.get(ctx_image_name, {})
            context_views.append({
                "extrinsics": extrinsics[i],
                "intrinsics": intrinsics[i],
                "image":      images[i],
                "near":       near[i],
                "far":        far[i],
                "index":      indices[i],
                "rpc":        rpcs[i],
                "enu_origin": enu_origins[i],
                "image_name": ctx_image_name,
                "px":         _patch_px,
                "py":         _patch_py,
                "col_start":  ctx_offsets.get("col_start", _patch_px * 256),
                "row_start":  ctx_offsets.get("row_start", _patch_py * 256),
            })

        target_views = []
        for i in range(num_context, V):
            image_name  = os.path.splitext(os.path.basename(files[i]))[0]
            folder      = os.path.basename(os.path.dirname(files[i]))
            folder_parts = folder.split("_")
            if folder_parts[-1] in ["topleft", "center"]:
                patch = folder_parts[-2]
            else:
                patch = folder_parts[-1]
            py = int(patch[1:3])
            px = int(patch[3:5])
            tgt_offsets = crop_offsets.get(image_name, {})
            target_views.append({
                "extrinsics": extrinsics[i],
                "intrinsics": intrinsics[i],
                "image":      images[i],
                "near":       near[i],
                "far":        far[i],
                "index":      indices[i],
                "rpc":        rpcs[i],
                "enu_origin": enu_origins[i],
                "image_name": image_name,
                "px":         px,
                "py":         py,
                "col_start":  tgt_offsets.get("col_start", px * 256),
                "row_start":  tgt_offsets.get("row_start", py * 256),
            })

        def _stack(views, key):
            return torch.stack([v[key] for v in views])

        def _list(views, key):
            return [v[key] for v in views]

        return {
            "context": {
                "extrinsics":  _stack(context_views, "extrinsics"),
                "intrinsics":  _stack(context_views, "intrinsics"),
                "image":       _stack(context_views, "image"),
                "near":        _stack(context_views, "near"),
                "far":         _stack(context_views, "far"),
                "index":       _stack(context_views, "index"),
                "rpc":         _stack(context_views, "rpc"),
                "enu_origin":  _stack(context_views, "enu_origin"),
                "image_name":  _list(context_views, "image_name"),
                "px":          _list(context_views, "px"),
                "py":          _list(context_views, "py"),
                "col_start":   _list(context_views, "col_start"),
                "row_start":   _list(context_views, "row_start"),
            },
            "target": {
                "extrinsics":  _stack(target_views, "extrinsics"),
                "intrinsics":  _stack(target_views, "intrinsics"),
                "image":       _stack(target_views, "image"),
                "near":        _stack(target_views, "near"),
                "far":         _stack(target_views, "far"),
                "index":       _stack(target_views, "index"),
                "rpc":         _stack(target_views, "rpc"),
                "enu_origin":  _stack(target_views, "enu_origin"),
                "image_name":  _list(target_views, "image_name"),
                "px":          _list(target_views, "px"),
                "py":          _list(target_views, "py"),
                "col_start":   _list(target_views, "col_start"),
                "row_start":   _list(target_views, "row_start"),
            },
            "scene": scene_dir.name,
        }
