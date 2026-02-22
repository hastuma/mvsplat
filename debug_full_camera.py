"""
完整診斷：分析 Target Camera 和 Context Gaussians 的座標系對齊問題
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/project/winston/mvsplat')

from src.geometry.rpc import RPC
from osgeo import gdal

print("=" * 70)
print("🔍 完整診斷：相機與 Gaussians 座標系")
print("=" * 70)

# 加載測試數據
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

# 假設 context 是前 3 個，target 是第 4 個
context_files = files[:3]
target_file = files[3]

# 加載 RPC
rpcs = []
for f in files:
    ds = gdal.Open(f)
    rpc = parse_rpc(ds)
    rpcs.append(torch.from_numpy(rpc).double())

context_rpcs = torch.stack(rpcs[:3]).unsqueeze(0)  # [1, 3, 90]
target_rpc = rpcs[3].unsqueeze(0)  # [1, 90]

h, w = 256, 256
distance = 100.0
gsd = 0.5

print("\n1️⃣ 計算 ENU 參考點（Context 第一個視角的圖像中心）")
print("-" * 50)

# 使用 Context 第一視角圖像中心做 ENU 參考
rpc_first = RPC(context_rpcs[:, 0, :])  # [1, 90]
h_center_ref = context_rpcs[:, 0, 8]  # HEIGHT_OFF
u_center = torch.full_like(h_center_ref, h / 2.0)
v_center = torch.full_like(h_center_ref, w / 2.0)
lat_ref, lon_ref = rpc_first.inverse(u_center, v_center, h_center_ref)

print(f"   ENU 參考點 (Context 0 圖像中心):")
print(f"   lat_ref = {lat_ref.item():.8f}")
print(f"   lon_ref = {lon_ref.item():.8f}")

print("\n2️⃣ 計算 Context Cameras (encoder 使用的)")
print("-" * 50)

rpc_ctx_flat = RPC(context_rpcs.view(-1, 90))  # [3, 90]
lat_ref_ctx = lat_ref.repeat(3)
lon_ref_ctx = lon_ref.repeat(3)
K_ctx, c2w_ctx = rpc_ctx_flat.compute_camera_geometry(h, w, lat_ref_ctx, lon_ref_ctx, distance=distance, gsd=gsd)

for i in range(3):
    pos = c2w_ctx[i, :3, 3].numpy()
    print(f"   Context Camera {i}: position = [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

print("\n3️⃣ 計算 Target Camera (decoder 使用的)")
print("-" * 50)

rpc_tgt = RPC(target_rpc)
# Target 也使用相同的參考點
K_tgt, c2w_tgt = rpc_tgt.compute_camera_geometry(h, w, lat_ref, lon_ref, distance=distance, gsd=gsd)

tgt_pos = c2w_tgt[0, :3, 3].numpy()
print(f"   Target Camera: position = [{tgt_pos[0]:.2f}, {tgt_pos[1]:.2f}, {tgt_pos[2]:.2f}]")
print(f"   焦距: fx={K_tgt[0, 0, 0].item():.2f}, fy={K_tgt[0, 1, 1].item():.2f}")

# 檢查相機方向
tgt_z_axis = c2w_tgt[0, :3, 2].numpy()
print(f"   相機 Z 軸方向: [{tgt_z_axis[0]:.4f}, {tgt_z_axis[1]:.4f}, {tgt_z_axis[2]:.4f}]")

print("\n4️⃣ 模擬 Gaussians 位置（使用 Context 圖像）")
print("-" * 50)

# 模擬每個像素在高度 h_ground = HEIGHT_OFF 處的地理位置
# 簡化：只計算四個角落
h_ground = context_rpcs[0, 0, 8]  # HEIGHT_OFF of first view
corners = [(0, 0), (0, 255), (255, 0), (255, 255)]

all_positions = []
for i in range(3):
    rpc_v = RPC(context_rpcs[:, i, :])
    for row, col in corners:
        row_t = torch.tensor([float(row)]).double()
        col_t = torch.tensor([float(col)]).double()
        lat, lon = rpc_v.inverse(row_t, col_t, h_ground.unsqueeze(0))
        
        # ENU
        rad = np.pi / 180.0
        r_earth = 6378137.0
        x = (lon.item() - lon_ref.item()) * rad * r_earth * np.cos(lat_ref.item() * rad)
        y = (lat.item() - lat_ref.item()) * rad * r_earth
        z = h_ground.item()
        all_positions.append([x, y, z])

all_positions = np.array(all_positions)
print(f"   Gaussians 範圍 (12 個角點):")
print(f"   X: [{all_positions[:, 0].min():.2f}, {all_positions[:, 0].max():.2f}]")
print(f"   Y: [{all_positions[:, 1].min():.2f}, {all_positions[:, 1].max():.2f}]")
print(f"   Z: [{all_positions[:, 2].min():.2f}, {all_positions[:, 2].max():.2f}]")

gs_center = all_positions.mean(axis=0)
print(f"   Gaussians 中心: [{gs_center[0]:.2f}, {gs_center[1]:.2f}, {gs_center[2]:.2f}]")

print("\n5️⃣ 相機與 Gaussians 的關係")
print("-" * 50)

distance_to_gs = np.linalg.norm(gs_center - tgt_pos)
print(f"   Target Camera 到 Gaussians 中心的距離: {distance_to_gs:.2f} 米")

# 在相機座標系中的位置
w2c = np.linalg.inv(c2w_tgt[0].numpy())
gs_in_cam = (w2c[:3, :3] @ all_positions.T + w2c[:3, 3:4]).T
print(f"\n   Gaussians 在相機座標系中:")
print(f"   深度 (Z): [{gs_in_cam[:, 2].min():.2f}, {gs_in_cam[:, 2].max():.2f}]")

# near/far 範圍
near = 50.0  # 從 config
far = 200.0
print(f"\n   配置的 near/far: [{near}, {far}]")

# 有多少在 near-far 內？
in_range = (gs_in_cam[:, 2] > near) & (gs_in_cam[:, 2] < far)
print(f"   在 near-far 內: {in_range.sum()}/{len(in_range)}")

# 有多少在相機後面？
behind = gs_in_cam[:, 2] < 0
print(f"   在相機後面: {behind.sum()}/{len(behind)}")

print("\n" + "=" * 70)
print("6️⃣ 診斷結論")
print("=" * 70)

if behind.sum() > 0:
    print("\n❌ 問題：部分 Gaussians 在相機後面！")
    print("   這說明相機方向可能有問題（應該朝向 Gaussians）")
    
if gs_in_cam[:, 2].max() < near or gs_in_cam[:, 2].min() > far:
    print("\n❌ 問題：Gaussians 不在 near-far 範圍內！")
    print(f"   Gaussian 深度範圍: [{gs_in_cam[:, 2].min():.2f}, {gs_in_cam[:, 2].max():.2f}]")
    print(f"   配置的 near-far: [{near}, {far}]")

# 檢查相機高度
if tgt_pos[2] > gs_center[2]:
    print(f"\n✅ 相機在 Gaussians 上方 (高度差: {tgt_pos[2] - gs_center[2]:.2f}m)")
else:
    print(f"\n⚠️ 相機不在 Gaussians 上方！")

# 檢查相機 Z 軸
if tgt_z_axis[2] < -0.5:  # 應該大致朝下
    print("✅ 相機 Z 軸朝下（俯視）")
else:
    print(f"⚠️ 相機 Z 軸方向異常: {tgt_z_axis}")
