#!/usr/bin/env python3
"""
論文方法實現：從 OSM 提取建築物 → Instance Segmentation → 3D Volume 構建 → SuperSplat 導出
"""

import sys

try:
    import osmnx as ox
except ModuleNotFoundError:
    print("✗ osmnx 未安裝，請執行: pip install osmnx")
    raise SystemExit(1)

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    print("✗ opencv-python 未安裝，請執行: pip install opencv-python")
    raise SystemExit(1)

try:
    from shapely.geometry import box, mapping
    from shapely.ops import unary_union
except ModuleNotFoundError:
    print("✗ shapely 未安裝，請執行: pip install shapely")
    raise SystemExit(1)
import json
import pickle
from pathlib import Path

# ================== 設定參數 ==================
POINT = (24.786944654672414, 120.9979260167771)  # 交大經緯度
# 其他測試座標（避免語法錯誤；需要時可改成 POINT = (...)）
# POINT = (30.2996, -81.6403)
DIST = 200  # 搜尋半徑（公尺）
GRID_RESOLUTION = 1.0  # 柵格解析度（公尺/像素）
MAX_HEIGHT = 100  # 最大高度（公尺）

# 地板（地面）可視化：在 z=0 放一個稀疏取樣的平面點雲
GROUND_STRIDE_METERS = 2.0  # 越大越稀疏、點越少；1.0 表示每個像素都放一點
GROUND_COLOR_RGB = (0.58, 0.68, 0.58)  # 淡綠，與建築灰階略做區別
OUTPUT_DIR = Path('./output')
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("論文方法實現：3D Volume 構建")
print("=" * 60)

# ================== Step 1: 從 OSM 提取建築物數據 ==================
print("\n[1/5] 從 OpenStreetMap 提取建築物數據...")
try:
    gdf = ox.features_from_point(POINT, tags={'building': True}, dist=DIST)
    print(f"✓ 成功提取 {len(gdf)} 棟建築物")
    
    # 保存原始數據的統計
    print(f"  - 計算邊界長方體...")
    gdf_valid = gdf[gdf.geometry.is_valid].copy()
    bounds = gdf_valid.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    print(f"  - 經度範圍: [{min_lon:.6f}, {max_lon:.6f}]")
    print(f"  - 緯度範圍: [{min_lat:.6f}, {max_lat:.6f}]")
    
except ImportError:
    print("✗ osmnx 未安裝，請執行: pip install osmnx")
    exit(1)

# ================== Step 2: 座標轉換與柵格化 ==================
print("\n[2/5] 進行座標轉換與柵格化...")

# 簡化投影：使用平面座標（經度為X，緯度為Y，單位：度）
# 實際應用應使用 UTM 或其他投影
scale_x = 111320 * np.cos(np.radians(POINT[0]))  # 1度經度對應的公尺數
scale_y = 111320  # 1度緯度對應的公尺數

# 計算網格大小
img_width = int((max_lon - min_lon) * scale_x / GRID_RESOLUTION) + 10
img_height = int((max_lat - min_lat) * scale_y / GRID_RESOLUTION) + 10
img_depth = int(MAX_HEIGHT / GRID_RESOLUTION)

print(f"  - 3D Volume 大小: {img_height} × {img_width} × {img_depth}")
print(f"  - 柵格解析度: {GRID_RESOLUTION} m/pixel")

# ================== Step 3: 生成 Instance Segmentation Map ==================
print("\n[3/5] 生成 Instance Segmentation Map...")

# 建立 instance 和 semantic 標籤
instance_map = np.zeros((img_height, img_width), dtype=np.int32)
semantic_map = np.zeros((img_height, img_width), dtype=np.int32)
height_map = np.zeros((img_height, img_width), dtype=np.float32)

# 語義類別映射
semantic_classes = {
    'residential': 1,
    'commercial': 2,
    'industrial': 3,
    'public': 4,
    'other': 5
}

instance_id = 1
building_info = []

for idx, (geom_id, row) in enumerate(gdf_valid.iterrows()):
    try:
        geom = row.geometry
        
        # 獲取高度
        height = 15.0  # 預設高度
        if 'height' in row and row['height'] is not None:
            try:
                height = float(str(row['height']).split()[0])
            except:
                pass
        if 'building:levels' in row and row['building:levels'] is not None:
            try:
                height = float(row['building:levels']) * 3.5  # ~3.5m per level
            except:
                pass
        
        # 獲取語義類別
        building_type = row.get('building', 'other')
        semantic_id = semantic_classes.get(building_type, 5)
        
        # 將多邊形轉換為像素座標
        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
        elif geom.geom_type == 'MultiPolygon':
            continue  # 簡化：跳過多邊形
        else:
            continue
        
        # 轉換為像素座標
        pixel_coords = []
        for lon, lat in coords:
            px = int((lon - min_lon) * scale_x / GRID_RESOLUTION)
            py = int((max_lat - lat) * scale_y / GRID_RESOLUTION)
            
            if 0 <= px < img_width and 0 <= py < img_height:
                pixel_coords.append([px, py])
        
        if len(pixel_coords) >= 3:
            pixel_coords = np.array(pixel_coords, dtype=np.int32)
            
            # 在 instance map 上繪製多邊形
            cv2.fillPoly(instance_map, [pixel_coords], instance_id)
            cv2.fillPoly(semantic_map, [pixel_coords], semantic_id)
            
            # 填充高度圖
            mask = instance_map == instance_id
            height_map[mask] = height
            
            # 記錄建築物信息
            building_info.append({
                'instance_id': instance_id,
                'semantic_id': semantic_id,
                'semantic_name': building_type,
                'height': height,
                'centroid': [np.mean([c[0] for c in pixel_coords]), 
                            np.mean([c[1] for c in pixel_coords])]
            })
            
            instance_id += 1
    
    except Exception as e:
        continue

print(f"✓ 成功處理 {instance_id - 1} 棟唯一建築物")

# 保存 segmentation 地圖
seg_vis = (instance_map % 256).astype(np.uint8)
cv2.imwrite(str(OUTPUT_DIR / 'instance_segmentation.png'), seg_vis)
print(f"  - Instance map 已保存: {OUTPUT_DIR / 'instance_segmentation.png'}")

seg_vis_color = cv2.applyColorMap(seg_vis, cv2.COLORMAP_JET)
cv2.imwrite(str(OUTPUT_DIR / 'instance_segmentation_color.png'), seg_vis_color)
print(f"  - 彩色 Instance map 已保存: {OUTPUT_DIR / 'instance_segmentation_color.png'}")

# 保存高度圖可視化
height_map_clean = np.nan_to_num(height_map, nan=0.0)
height_vis = (np.clip(height_map_clean / MAX_HEIGHT * 255, 0, 255)).astype(np.uint8)
height_vis_color = cv2.applyColorMap(height_vis, cv2.COLORMAP_TURBO)
cv2.imwrite(str(OUTPUT_DIR / 'height_map.png'), height_vis_color)
print(f"  - 高度圖已保存: {OUTPUT_DIR / 'height_map.png'}")

# ================== Step 4: 構建 3D Volume ==================
print("\n[4/5] 構建 3D Volume...")

# Volume 結構：V[y, x, z] = (instance_id, semantic_id)
volume_instance = np.zeros((img_height, img_width, img_depth), dtype=np.int32)
volume_semantic = np.zeros((img_height, img_width, img_depth), dtype=np.int32)

for i in range(instance_id - 1):
    # 獲取該 instance 的掩碼
    mask_2d = instance_map == (i + 1)
    if not mask_2d.any():
        continue
    
    height = height_map[mask_2d].max()
    if np.isnan(height) or height <= 0:
        height = 15.0  # 預設高度
    
    # 獲取該 instance 的語義類別
    semantic_id = semantic_map[mask_2d].max() if mask_2d.any() else 0
    
    # 在 z 軸方向填充
    z_levels = int(height / GRID_RESOLUTION)
    for z in range(min(z_levels, img_depth)):
        volume_instance[mask_2d, z] = i + 1
        volume_semantic[mask_2d, z] = semantic_id

print(f"✓ 3D Volume 構建完成")
print(f"  - 非零 voxel 數量: {np.count_nonzero(volume_instance)}")

# ================== Step 5: 導出為 3D Gaussian Splatting PLY（SuperSplat 相容） ==================
print("\n[5/5] 導出為 3D Gaussian Splatting PLY...")
import struct

# 向量化提取非零 voxel 座標（比三層循環快數百倍）
ys, xs, zs = np.where(volume_instance > 0)
N = len(ys)
print(f"  - 非零 voxel 數量: {N}")

# 偵測邊緣 voxel（只標記建築物表面，不標記內部細節）
print(f"  - 偵測建築物表面邊緣...")
edge_mask = np.zeros_like(volume_instance, dtype=bool)
roof_mask = np.zeros_like(volume_instance, dtype=bool)  # 新增：屋頂標記

# 檢查 6 個相鄰方向
neighbors = [
    (-1, 0, 0),  # 上
    (1, 0, 0),   # 下
    (0, -1, 0),  # 左
    (0, 1, 0),   # 右
    (0, 0, -1),  # 前（-z）
    (0, 0, 1),   # 後（+z）
]

for dy, dx, dz in neighbors:
    # 移位後的 volume
    shifted = np.roll(volume_instance, (dy, dx, dz), axis=(0, 1, 2))
    
    # 邊界處理：邊界外視為空（0）
    if dy == -1:
        shifted[0, :, :] = 0
    elif dy == 1:
        shifted[-1, :, :] = 0
    if dx == -1:
        shifted[:, 0, :] = 0
    elif dx == 1:
        shifted[:, -1, :] = 0
    if dz == -1:
        shifted[:, :, 0] = 0
    elif dz == 1:
        shifted[:, :, -1] = 0
    
    # **關鍵改進**：只標記相鄰是**背景(0)**的 voxel
    # 這樣就只會標記建築物的外表面，不會標記內部細節（如窗戶、多層樓）
    edge_mask |= (volume_instance > 0) & (shifted == 0)

# 新增：區分屋頂和側面
# 屋頂 = 上方是背景(0)的體素
shifted_above = np.roll(volume_instance, (-1, 0, 0), axis=(0, 1, 2))
shifted_above[0, :, :] = 0  # 邊界處理
roof_mask = (volume_instance > 0) & (shifted_above == 0)

# 側面 = 表面但不是屋頂
side_mask = edge_mask & ~roof_mask

edge_voxel_count = np.sum(edge_mask)
roof_voxel_count = np.sum(roof_mask)
side_voxel_count = np.sum(side_mask)
interior_voxel_count = np.sum(volume_instance > 0) - edge_voxel_count
print(f"  - 屋頂 voxel: {roof_voxel_count:,} ({100*roof_voxel_count/N:.1f}%)")
print(f"  - 側面 voxel: {side_voxel_count:,} ({100*side_voxel_count/N:.1f}%)")
print(f"  - 內部 voxel: {interior_voxel_count:,} ({100*interior_voxel_count/N:.1f}%)")

# 使用局部公尺座標（以區域中心為原點，SuperSplat 才能正常顯示）
cx = img_width / 2.0
cy = img_height / 2.0
points_x = (xs.astype(np.float32) - cx) * GRID_RESOLUTION
points_y = (ys.astype(np.float32) - cy) * GRID_RESOLUTION
points_z = zs.astype(np.float32) * GRID_RESOLUTION

# 偵測每個點是否為屋頂或側面
is_roof = roof_mask[ys, xs, zs]
is_side = side_mask[ys, xs, zs]
is_interior = ~edge_mask[ys, xs, zs]

# === 3DGS 參數 ===
SH_C0 = 0.28209479177387814


def rgb_to_fdc(rgb):
    """Convert display RGB in [0,1] to 3DGS f_dc (per channel)."""
    r, g, b = rgb
    return (
        np.float32((r - 0.5) / SH_C0),
        np.float32((g - 0.5) / SH_C0),
        np.float32((b - 0.5) / SH_C0),
    )

# 白色（RGB 1.0）：f_dc = (1.0 - 0.5) / SH_C0 ≈ 1.7725
f_dc_white = np.float32((1.0 - 0.5) / SH_C0)

# 淺灰色（RGB 0.5）：f_dc = (0.5 - 0.5) / SH_C0 = 0
f_dc_light_gray = np.float32(0.0)

# 深灰色（RGB 0.2）：f_dc = (0.2 - 0.5) / SH_C0 ≈ -1.064
f_dc_dark_gray = np.float32((0.2 - 0.5) / SH_C0)

# 為屋頂、側面和內部分別分配顏色
f_dc_building_scalar = np.zeros(N, dtype=np.float32)
f_dc_building_scalar[is_roof] = f_dc_dark_gray    # 屋頂用深灰色
f_dc_building_scalar[is_side] = f_dc_light_gray   # 側面用淺灰色
f_dc_building_scalar[is_interior] = f_dc_white    # 內部用白色

# 建築物：灰階（3 通道相同）
f_dc0_building = f_dc_building_scalar.astype(np.float32)
f_dc1_building = f_dc_building_scalar.astype(np.float32)
f_dc2_building = f_dc_building_scalar.astype(np.float32)

# 地板：淡綠色（3 通道不同，跟建築灰階有區別）
ground_fdc0, ground_fdc1, ground_fdc2 = rgb_to_fdc(GROUND_COLOR_RGB)

# Scale: log(半個格子大小)，讓每個高斯覆蓋一個 voxel
scale_val = np.float32(np.log(GRID_RESOLUTION * 0.5))

# Rotation: 單位四元數 [1, 0, 0, 0]
# Opacity: logit(0.99) ≈ 4.595
opacity_val = np.float32(np.log(0.99 / 0.01))

# 寫入 binary PLY
ply_path = OUTPUT_DIR / 'buildings_3d_volume.ply'
print(f"  - 寫入 3DGS PLY (binary)...")

# ================== 新增：地板（地面）點雲 ==================
# 用局部公尺座標、z=0 的平面做稀疏取樣。
stride_px = max(1, int(round(GROUND_STRIDE_METERS / GRID_RESOLUTION)))
ground_ys = np.arange(0, img_height, stride_px, dtype=np.int32)
ground_xs = np.arange(0, img_width, stride_px, dtype=np.int32)
gy, gx = np.meshgrid(ground_ys, ground_xs, indexing='ij')
gy = gy.reshape(-1)
gx = gx.reshape(-1)
Ng = int(len(gy))

ground_points_x = (gx.astype(np.float32) - cx) * GRID_RESOLUTION
ground_points_y = (gy.astype(np.float32) - cy) * GRID_RESOLUTION
ground_points_z = np.zeros(Ng, dtype=np.float32)

# 合併：建築 voxel + 地板平面
points_x_all = np.concatenate([points_x.astype(np.float32), ground_points_x], axis=0)
points_y_all = np.concatenate([points_y.astype(np.float32), ground_points_y], axis=0)
points_z_all = np.concatenate([points_z.astype(np.float32), ground_points_z], axis=0)

f_dc0_all = np.concatenate([f_dc0_building, np.full(Ng, ground_fdc0, dtype=np.float32)], axis=0)
f_dc1_all = np.concatenate([f_dc1_building, np.full(Ng, ground_fdc1, dtype=np.float32)], axis=0)
f_dc2_all = np.concatenate([f_dc2_building, np.full(Ng, ground_fdc2, dtype=np.float32)], axis=0)

N_total = int(len(points_x_all))
print(f"  - 地板點數: {Ng:,} (stride={stride_px} px)")
print(f"  - 總點數（建築+地板）: {N_total:,}")

with open(ply_path, 'wb') as f:
    # Header（必須是 ASCII）
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment 3D Gaussian Splatting - Generated from OSM building data\n"
        f"element vertex {N_total}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )
    f.write(header.encode('ascii'))
    
    # 批次寫入二進制數據（每次寫 10 萬點）
    batch_size = 100000
    fmt = '<17f'  # 17 個 little-endian float
    
    for start in range(0, N_total, batch_size):
        end = min(start + batch_size, N_total)
        batch_bytes = bytearray()
        
        for i in range(start, end):
            batch_bytes.extend(struct.pack(
                fmt,
                points_x_all[i], points_y_all[i], points_z_all[i],  # x, y, z
                0.0, 0.0, 0.0,                           # nx, ny, nz
                f_dc0_all[i], f_dc1_all[i], f_dc2_all[i], # f_dc_0, f_dc_1, f_dc_2
                opacity_val,                              # opacity
                scale_val, scale_val, scale_val,          # scale_0, scale_1, scale_2
                1.0, 0.0, 0.0, 0.0                       # rot_0, rot_1, rot_2, rot_3
            ))
        
        f.write(batch_bytes)
        
        if end % 100000 == 0 or end == N_total:
            print(f"  - 已寫入 {end:,} / {N_total:,} 點...")

file_size_mb = ply_path.stat().st_size / (1024 * 1024)
print(f"✓ 3DGS PLY 已保存: {ply_path} ({file_size_mb:.1f} MB)")
print(f"  - 格式: binary_little_endian")
print(f"  - 著色: 屋頂深灰 | 側面淺灰 | 內部白色 | 地板淡綠（清楚區分地面與建築物）")
print(f"  - 可直接在 SuperSplat 中開啟")

# ================== 保存元數據 ==================
metadata = {
    'location': {'latitude': POINT[0], 'longitude': POINT[1]},
    'search_radius': DIST,
    'grid_resolution': GRID_RESOLUTION,
    'max_height': MAX_HEIGHT,
    'volume_shape': (img_height, img_width, img_depth),
    'num_buildings': instance_id - 1,
    'num_points_buildings': int(N),
    'num_points_ground': int(Ng),
    'num_points_total': int(N_total),
    'building_classes': semantic_classes,
    'building_info': building_info[:20]  # 保存前 20 棟建築物信息
}

metadata_path = OUTPUT_DIR / 'metadata.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✓ 元數據已保存: {metadata_path}")

# ================== 最終摘要 ==================
print("\n" + "=" * 60)
print("✓ 處理完成！")
print("=" * 60)
print(f"\n輸出文件位置: {OUTPUT_DIR.absolute()}")
print(f"\n生成的文件：")
print(f"  1. {OUTPUT_DIR / 'buildings_3d_volume.ply'} - SuperSplat 點雲")
print(f"  2. {OUTPUT_DIR / 'instance_segmentation.png'} - Instance 地圖")
print(f"  3. {OUTPUT_DIR / 'instance_segmentation_color.png'} - 彩色 Instance 地圖")
print(f"  4. {OUTPUT_DIR / 'height_map.png'} - 高度圖可視化")
print(f"  5. {OUTPUT_DIR / 'metadata.json'} - 元數據")
print(f"\n使用 SuperSplat 打開: {ply_path}")
print("="*60)