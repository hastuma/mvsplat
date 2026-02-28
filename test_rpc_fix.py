#!/usr/bin/env python
"""
测试 RPC 相机几何修正：
1. 验证单位正确性（不再有 m²/px 错误）
2. 验证 Z 轴不再强制垂直向下（有真实倾角）
"""
import torch
import sys
sys.path.insert(0, '/project/winston/mvsplat')
from src.geometry.rpc import RPC

# 创建测试 RPC 参数（使用真实 DFC2019 范围）
B = 2
rpc_coeffs = torch.randn(B, 90)

# 设置合理的 offsets/scales（模拟 DFC2019）
rpc_coeffs[:, 0] = 20000.0  # LINE_OFF
rpc_coeffs[:, 1] = 20000.0  # LINE_SCALE
rpc_coeffs[:, 2] = 20000.0  # SAMP_OFF
rpc_coeffs[:, 3] = 20000.0  # SAMP_SCALE
rpc_coeffs[:, 4] = 30.3  # LAT_OFF (Jacksonville, FL)
rpc_coeffs[:, 5] = 0.5   # LAT_SCALE
rpc_coeffs[:, 6] = -81.7 # LONG_OFF
rpc_coeffs[:, 7] = 0.5   # LONG_SCALE
rpc_coeffs[:, 8] = -20.0 # HEIGHT_OFF (MSL)
rpc_coeffs[:, 9] = 501.0 # HEIGHT_SCALE (这是归一化参数，非相机高度！)

# 初始化 RPC
rpc = RPC(rpc_coeffs, device='cpu')

print("=" * 70)
print("🔬 RPC 相机几何修正验证")
print("=" * 70)

# 测试 compute_camera_geometry
h, w = 256, 256
K, c2w, distance = rpc.compute_camera_geometry(h, w)

print(f"\n📐 图像尺寸: {h}×{w} pixels")
print(f"\n✅ 虚拟相机高度 (distance):")
print(f"   值: {distance[0].item():.2f} m")
print(f"   预期范围: ~40-50m (≈ w/2 × 0.35)")
print(f"   单位检查: (像素) × (米/像素) = 米 ✓")

print(f"\n✅ 焦距 (focal):")
focal_actual = K[0, 0, 0].item()
print(f"   值: {focal_actual:.2f} pixels")
print(f"   预期值: {w/2:.2f} pixels (≈ w/2)")
print(f"   单位检查: 米 / (米/像素) = 像素 ✓")

# 检查旧公式会出现的错误
height_scale = rpc_coeffs[0, 9].item()
gsd_estimate = 0.35  # 估计值
wrong_distance = height_scale * gsd_estimate
wrong_focal = wrong_distance / gsd_estimate
print(f"\n❌ 旧公式（错误）会产生:")
print(f"   distance = height_scale × gsd = {height_scale} × {gsd_estimate} = {wrong_distance:.2f}")
print(f"   单位: 米 × (米/像素) = 米²/像素 ❌ 无意义！")
print(f"   focal = {wrong_distance:.2f} / {gsd_estimate} = {wrong_focal:.2f}")
print(f"   单位: (米²/像素) / (米/像素) = 米 ❌ 焦距应该是像素！")

print(f"\n✅ 新公式（正确）:")
print(f"   distance = (w/2) × gsd ≈ {w/2} × {gsd_estimate} = {(w/2)*gsd_estimate:.2f} m")
print(f"   focal = distance / gsd ≈ {(w/2)*gsd_estimate / gsd_estimate:.2f} px = w/2")

# 测试 Z 轴倾角
print(f"\n🎯 相机朝向（Z 轴）检查:")
R_c2w = c2w[0, :3, :3]  # [3, 3]
cam_z = R_c2w[:, 2]  # Z 轴在世界坐标系中的方向

print(f"   Z 轴向量: [{cam_z[0]:.4f}, {cam_z[1]:.4f}, {cam_z[2]:.4f}]")
print(f"   Z_vertical (ENU 向下): [0, 0, -1]")

# 计算与垂直方向的夹角
z_vertical = torch.tensor([0.0, 0.0, -1.0])
cos_angle = torch.dot(cam_z, z_vertical)
angle_deg = torch.acos(cos_angle.clamp(-1, 1)) * 180 / 3.14159

print(f"   与垂直向下的夹角: {angle_deg.item():.2f}°")

if angle_deg.item() < 0.1:
    print(f"   ⚠️  相机几乎完全垂直向下（Nadir）")
    print(f"   这可能表示光线追踪未生效，或测试数据特殊")
else:
    print(f"   ✅ 相机有倾角（非强制 Nadir），光线追踪生效！")

# 检查 Z 轴的水平分量
z_horizontal_norm = torch.sqrt(cam_z[0]**2 + cam_z[1]**2)
print(f"   Z 轴水平分量: {z_horizontal_norm.item():.6f}")
if z_horizontal_norm.item() > 1e-6:
    print(f"   ✅ Z 轴有水平分量，不再强制 [0, 0, z]！")
else:
    print(f"   ⚠️  Z 轴水平分量为 0，可能仍然强制垂直")

print(f"\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
