"""
診斷：檢查不同視角的 ENU 座標差異

目標：理解為什麼 X/Y 範圍達到 13km
"""
import numpy as np
from osgeo import gdal
import sys
sys.path.insert(0, '/project/winston/mvsplat')
from src.geometry.rpc import RPC
import torch

# 讀取所有 4 個視角的圖像
files = [
    "/project/winston/datasets/DFC2019/overfit/training/JAX_004_p0004/JAX_004_006_RGB.tif",
    "/project/winston/datasets/DFC2019/overfit/training/JAX_004_p0004/JAX_004_009_RGB.tif",
    "/project/winston/datasets/DFC2019/overfit/training/JAX_004_p0004/JAX_004_012_RGB.tif",
    "/project/winston/datasets/DFC2019/overfit/training/JAX_004_p0004/JAX_004_014_RGB.tif",
]

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

print("=" * 70)
print("各視角的 RPC 參考點比較")
print("=" * 70)

view_data = []
for i, fpath in enumerate(files):
    ds = gdal.Open(fpath)
    rpc = parse_rpc(ds)
    
    lat_off = rpc[4]  # LAT_OFF
    lon_off = rpc[6]  # LONG_OFF
    height_off = rpc[8]  # HEIGHT_OFF
    line_off = rpc[0]  # LINE_OFF
    samp_off = rpc[2]  # SAMP_OFF
    
    view_data.append({
        'file': fpath.split('/')[-1],
        'lat_off': lat_off,
        'lon_off': lon_off,
        'height_off': height_off,
        'line_off': line_off,
        'samp_off': samp_off,
        'rpc': rpc
    })
    
    print(f"\n視角 {i}: {view_data[-1]['file']}")
    print(f"  LAT_OFF: {lat_off:.8f}")
    print(f"  LONG_OFF: {lon_off:.8f}")
    print(f"  HEIGHT_OFF: {height_off:.1f}")
    print(f"  LINE_OFF: {line_off:.2f}")
    print(f"  SAMP_OFF: {samp_off:.2f}")

# 計算第一視角圖像中心的經緯度作為參考點
print("\n" + "=" * 70)
print("計算各視角圖像中心的地理座標")
print("=" * 70)

rad = np.pi / 180.0
r_earth = 6378137.0

# 使用第一視角的 LAT_OFF, LONG_OFF 作為全局參考點
lat_ref_global = view_data[0]['lat_off']
lon_ref_global = view_data[0]['lon_off']
print(f"\n全局 ENU 參考點 (第一視角的 LAT/LONG OFF):")
print(f"  lat_ref = {lat_ref_global:.8f}")
print(f"  lon_ref = {lon_ref_global:.8f}")

# 計算各視角圖像**中心**的 ENU 座標
print("\n各視角圖像中心的 ENU 座標 (相對於全局參考點):")
h_ground = torch.tensor([0.0]).double()

for i, v in enumerate(view_data):
    rpc_tensor = torch.from_numpy(v['rpc']).double().unsqueeze(0)
    rpc_obj = RPC(rpc_tensor)
    
    # 圖像中心 (row=128, col=128 for 256x256)
    row_center = torch.tensor([128.0]).double()
    col_center = torch.tensor([128.0]).double()
    
    lat_center, lon_center = rpc_obj.inverse(row_center, col_center, h_ground)
    lat_center = lat_center.item()
    lon_center = lon_center.item()
    
    # ENU 座標
    x_enu = (lon_center - lon_ref_global) * rad * r_earth * np.cos(lat_ref_global * rad)
    y_enu = (lat_center - lat_ref_global) * rad * r_earth
    
    print(f"\n視角 {i} ({v['file']}):")
    print(f"  圖像中心經緯度: lat={lat_center:.8f}, lon={lon_center:.8f}")
    print(f"  ENU 座標: X={x_enu:.2f}m, Y={y_enu:.2f}m")
    
    v['x_enu_center'] = x_enu
    v['y_enu_center'] = y_enu

# 計算 ENU 範圍
all_x = [v['x_enu_center'] for v in view_data]
all_y = [v['y_enu_center'] for v in view_data]

print("\n" + "=" * 70)
print("ENU 範圍統計")
print("=" * 70)
print(f"X 範圍: {min(all_x):.2f} ~ {max(all_x):.2f} ({max(all_x) - min(all_x):.2f}m)")
print(f"Y 範圍: {min(all_y):.2f} ~ {max(all_y):.2f} ({max(all_y) - min(all_y):.2f}m)")

print("\n" + "=" * 70)
print("結論")
print("=" * 70)
print("""
如果 X/Y 範圲達到 13km，原因可能是：

1. 不同視角的衛星圖像來自**不同的軌道位置**
   → 它們的 RPC 參數 (LINE_OFF, SAMP_OFF) 差異很大
   → 相對於同一個 ENU 參考點，座標差異巨大

2. 你的訓練使用了多個視角的 Gaussians 一起計算
   → 每個視角有 65536 個 Gaussians
   → 3 個視角 = 196608 個 Gaussians
   → 這些 Gaussians 分散在不同的 ENU 位置

解決方案：
- 選項 A: 每個視角使用自己的局部座標系
- 選項 B: 讓 Gaussians 的 means 使用相機座標系（統一減去相機位置）
""")
