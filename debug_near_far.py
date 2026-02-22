"""
診斷渲染器 near/far 問題
"""
import torch

# Dataset 的 near/far 是高度值
HEIGHT_OFF = -21.0  # 典型值
near_height = HEIGHT_OFF - 20.0  # -41
far_height = HEIGHT_OFF + 50.0   # +29

print("Dataset Near/Far (高度值):")
print(f"  near = {near_height} (HEIGHT_OFF - 20)")
print(f"  far = {far_height} (HEIGHT_OFF + 50)")

# 相機參數
camera_height = HEIGHT_OFF + 100.0  # 相機在地面以上 100m
print(f"\n相機高度: {camera_height}")

# Gaussians 的 Z 座標範圍（等於高度）
gs_z_min = near_height
gs_z_max = far_height
print(f"Gaussians Z 範圍: [{gs_z_min}, {gs_z_max}]")

# 相機到 Gaussians 的距離（深度）
depth_min = camera_height - gs_z_max  # 相機到最高 Gaussian 的距離
depth_max = camera_height - gs_z_min  # 相機到最低 Gaussian 的距離
print(f"\n相機到 Gaussians 的深度範圍:")
print(f"  depth_min (相機到最高Gaussian): {depth_min}")
print(f"  depth_max (相機到最低Gaussian): {depth_max}")

print("\n" + "=" * 50)
print("渲染器需要的 near/far (必須是正數):")
print(f"  render_near = {depth_min}")
print(f"  render_far = {depth_max}")

if near_height < 0:
    print("\n⚠️ 問題：Dataset 的 near 是負數！")
    print("   渲染器中 scale = 1 / near 會變成負數！")
    print("   這會導致座標被翻轉，Gaussians 可能被渲染到相機後面")
