"""
診斷腳本：深入分析 RPC 座標計算問題

目標：確認為什麼 X/Y 座標範圍達到 13,000 米
"""
import numpy as np
from osgeo import gdal
import sys
sys.path.insert(0, '/project/winston/mvsplat')
from src.geometry.rpc import RPC
import torch

# 1. 讀取原始大圖的 RPC
print("=" * 70)
print("1. 讀取原始大圖的 RPC 參數")
print("=" * 70)

orig_tif_path = "/project/winston/datasets/DFC2019/Track3-RGB-1/JAX_004_006_RGB.tif"
ds_orig = gdal.Open(orig_tif_path)

if ds_orig is None:
    print(f"ERROR: Cannot open {orig_tif_path}")
    exit(1)

# 讀取原始 RPC
orig_w, orig_h = ds_orig.RasterXSize, ds_orig.RasterYSize
print(f"原始圖片尺寸: {orig_w} x {orig_h}")

rpc_meta = ds_orig.GetMetadata('RPC')
if not rpc_meta:
    print("ERROR: No RPC metadata found!")
    exit(1)

print(f"\n原始 RPC 參數:")
print(f"  LINE_OFF: {rpc_meta.get('LINE_OFF')}")
print(f"  LINE_SCALE: {rpc_meta.get('LINE_SCALE')}")
print(f"  SAMP_OFF: {rpc_meta.get('SAMP_OFF')}")
print(f"  SAMP_SCALE: {rpc_meta.get('SAMP_SCALE')}")
print(f"  LAT_OFF: {rpc_meta.get('LAT_OFF')}")
print(f"  LAT_SCALE: {rpc_meta.get('LAT_SCALE')}")
print(f"  LONG_OFF: {rpc_meta.get('LONG_OFF')}")
print(f"  LONG_SCALE: {rpc_meta.get('LONG_SCALE')}")
print(f"  HEIGHT_OFF: {rpc_meta.get('HEIGHT_OFF')}")
print(f"  HEIGHT_SCALE: {rpc_meta.get('HEIGHT_SCALE')}")

# 2. 讀取 overfit 小圖的 RPC
print("\n" + "=" * 70)
print("2. 讀取 overfit 小圖的 RPC 參數")
print("=" * 70)

crop_tif_path = "/project/winston/datasets/DFC2019/overfit/training/JAX_004_p0004/JAX_004_006_RGB.tif"
ds_crop = gdal.Open(crop_tif_path)

if ds_crop is None:
    print(f"ERROR: Cannot open {crop_tif_path}")
    exit(1)

crop_w, crop_h = ds_crop.RasterXSize, ds_crop.RasterYSize
print(f"小圖尺寸: {crop_w} x {crop_h}")

crop_rpc_meta = ds_crop.GetMetadata('RPC')
if not crop_rpc_meta:
    print("ERROR: No RPC metadata found in crop!")
    exit(1)

print(f"\n小圖 RPC 參數:")
print(f"  LINE_OFF: {crop_rpc_meta.get('LINE_OFF')} (原始: {rpc_meta.get('LINE_OFF')})")
print(f"  LINE_SCALE: {crop_rpc_meta.get('LINE_SCALE')}")
print(f"  SAMP_OFF: {crop_rpc_meta.get('SAMP_OFF')} (原始: {rpc_meta.get('SAMP_OFF')})")
print(f"  SAMP_SCALE: {crop_rpc_meta.get('SAMP_SCALE')}")
print(f"  LAT_OFF: {crop_rpc_meta.get('LAT_OFF')}")
print(f"  LAT_SCALE: {crop_rpc_meta.get('LAT_SCALE')}")

# 3. 計算座標範圍
print("\n" + "=" * 70)
print("3. 使用 RPC 計算地理座標範圍")
print("=" * 70)

# 解析 RPC 係數
def parse_rpc(ds):
    keys = ['LINE_OFF', 'LINE_SCALE', 'SAMP_OFF', 'SAMP_SCALE',
            'LAT_OFF', 'LAT_SCALE', 'LONG_OFF', 'LONG_SCALE',
            'HEIGHT_OFF', 'HEIGHT_SCALE']
    coeffs = []
    for k in keys:
        val = ds.GetMetadataItem(k, 'RPC') 
        if val is None: val = ds.GetMetadata('RPC').get(k, 0)
        coeffs.append(float(val))
    for prefix in ['LINE_NUM_COEFF', 'LINE_DEN_COEFF', 'SAMP_NUM_COEFF', 'SAMP_DEN_COEFF']:
        val_str = ds.GetMetadataItem(prefix, 'RPC')
        if val_str is None: val_str = ds.GetMetadata('RPC').get(prefix, "")
        vals = [float(x) for x in val_str.split()]
        coeffs.extend(vals if len(vals) == 20 else vals + [0.0]*(20-len(vals)))
    return np.array(coeffs, dtype=np.float64)

rpc_crop = parse_rpc(ds_crop)
rpc_tensor = torch.from_numpy(rpc_crop).double().unsqueeze(0)  # [1, 90]
rpc_obj = RPC(rpc_tensor)

# 使用高度 = 0 (地面)
h_ground = torch.tensor([0.0]).double()

# 計算四個角落的經緯度
print("\n小圖四角的經緯度 (使用你的 RPC inverse):")
corners = [
    (0, 0, "左上"),
    (0, 255, "右上"),
    (255, 0, "左下"),
    (255, 255, "右下")
]

lat_lon_results = []
for row, col, name in corners:
    row_t = torch.tensor([float(row)]).double()
    col_t = torch.tensor([float(col)]).double()
    lat, lon = rpc_obj.inverse(row_t, col_t, h_ground)
    print(f"  {name} (row={row}, col={col}): lat={lat.item():.8f}, lon={lon.item():.8f}")
    lat_lon_results.append((lat.item(), lon.item()))

# 4. 計算 ENU 範圍
print("\n" + "=" * 70)
print("4. 計算 ENU 座標範圍")
print("=" * 70)

lat_ref = lat_lon_results[0][0]  # 左上角作為參考
lon_ref = lat_lon_results[0][1]

rad = np.pi / 180.0
r_earth = 6378137.0

print(f"\nENU 參考點: lat={lat_ref:.8f}, lon={lon_ref:.8f}")
print("\nENU 座標 (相對於左上角):")
for (lat, lon), (row, col, name) in zip(lat_lon_results, corners):
    x = (lon - lon_ref) * rad * r_earth * np.cos(lat_ref * rad)
    y = (lat - lat_ref) * rad * r_earth
    print(f"  {name}: X={x:.2f}m, Y={y:.2f}m")

# 5. 計算預期的圖像範圍
print("\n" + "=" * 70)
print("5. 預期 vs 實際範圍分析")
print("=" * 70)

# 使用 GSD = 0.5m
gsd = 0.5
expected_range = crop_w * gsd
print(f"\n預期範圍 (GSD={gsd}m, 圖片尺寸={crop_w}x{crop_h}):")
print(f"  {expected_range:.1f}m x {expected_range:.1f}m")

# 計算實際範圍
all_x = []
all_y = []
for (lat, lon), _ in zip(lat_lon_results, corners):
    x = (lon - lon_ref) * rad * r_earth * np.cos(lat_ref * rad)
    y = (lat - lat_ref) * rad * r_earth
    all_x.append(x)
    all_y.append(y)

actual_x_range = max(all_x) - min(all_x)
actual_y_range = max(all_y) - min(all_y)
print(f"\n實際範圍 (從 RPC 計算):")
print(f"  X: {actual_x_range:.1f}m")
print(f"  Y: {actual_y_range:.1f}m")

if abs(actual_x_range - expected_range) > 10:
    print(f"\n⚠️ 警告: 實際範圍 ({actual_x_range:.1f}m) 與預期 ({expected_range:.1f}m) 差異較大!")
    print("  可能原因: RPC offset 沒有正確調整到小圖座標系")

# 6. 診斷訓練時的座標問題
print("\n" + "=" * 70)
print("6. 診斷訓練時的 X/Y 座標問題")
print("=" * 70)

# 從 log 中的數據
print("\n從訓練 log 中觀察到的數據:")
print("  Means range: min=-6303.27, max=6804.27 (X)")
print("  相機位置: X=-6257, Y=6755")
print("  X 範圍 = 6804.27 - (-6303.27) = 13107.54 米 ≈ 13 公里!")

print("\n❓ 問題分析:")
print("  如果小圖的 RPC offset 是相對於原始大圖的:")
print(f"    - 小圖 LINE_OFF: {crop_rpc_meta.get('LINE_OFF')}")
print(f"    - 小圖 SAMP_OFF: {crop_rpc_meta.get('SAMP_OFF')}")
print("  但你輸入的 row/col 是 0~255 (小圖座標)")
print("  這會導致 RPC inverse 計算出錯誤的經緯度!")

# 7. 計算正確的座標轉換
print("\n" + "=" * 70)
print("7. 正確的做法")
print("=" * 70)
print("""
如果你的 RPC 仍使用大圖座標系，但訓練時輸入的是小圖座標 (0~255):

選項 A: 在使用 RPC inverse 時，加上 offset
  實際 row = 小圖 row + 大圖中的 row_offset
  實際 col = 小圖 col + 大圖中的 col_offset

選項 B: Normalize 座標到局部座標系
  既然 Z 座標是正確的，你可以：
  1. 計算圖像中心的經緯度作為參考點
  2. 或者直接使用相對於圖像中心的局部座標

建議: 使用選項 B - 將 X/Y 座標正規化到圖像範圍
""")

# 8. 檢查 p0004 的位置
print("\n" + "=" * 70)
print("8. 分析你的 crop 位置 (row=0, col=4)")
print("=" * 70)
# 假設原圖 2048x2048，分成 8x8 = 64 個 256x256 的 patch
# row=0, col=4 表示第一行第五個 patch
patch_row = 0
patch_col = 4
patch_size = 256

row_offset = patch_row * patch_size  # 0
col_offset = patch_col * patch_size  # 1024

print(f"Patch 位置: row={patch_row}, col={patch_col}")
print(f"在大圖中的 offset: row_start={row_offset}, col_start={col_offset}")
print(f"像素範圍: row=[{row_offset}, {row_offset + patch_size - 1}], col=[{col_offset}, {col_offset + patch_size - 1}]")
