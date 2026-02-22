"""
診斷腳本：分析相機和 Gaussian 的空間關係
"""
import numpy as np
import numpy as np

# 從 log 中提取的數據
camera_positions = [
    [-6257.4263, 6755.1123, 981.0],
    [-6333.809, 6432.9673, 979.0],
    [-6478.3867, 6588.1333, 981.0],
]

gaussian_stats = {
    "X": {"min": -6303.27, "max": 6804.27, "mean": -6257, "std": 25},
    "Y": {"min": -6524.23, "max": 6637.50, "mean": 6588, "std": 25},
    "Z": {"min": -3, "max": 3, "mean": 0, "std": 1.5},
}

print("=" * 60)
print("相機和場景空間分析")
print("=" * 60)

print("\n📍 相機位置 (ENU 座標):")
for i, pos in enumerate(camera_positions):
    print(f"  相機 {i+1}: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f} 米")

print(f"\n📊 Gaussian 分佈:")
print(f"  X 範圍: {gaussian_stats['X']['min']:.1f} ~ {gaussian_stats['X']['max']:.1f} 米")
print(f"        = {gaussian_stats['X']['max'] - gaussian_stats['X']['min']:.1f} 米總範圍")
print(f"  Y 範圍: {gaussian_stats['Y']['min']:.1f} ~ {gaussian_stats['Y']['max']:.1f} 米")
print(f"  Z 範圍: {gaussian_stats['Z']['min']:.1f} ~ {gaussian_stats['Z']['max']:.1f} 米 (地面高度)")

# 計算相機 FOV 覆蓋範圍
# 假設 256x256 圖像，焦距 = 128 * distance
distance = 1000.0
focal_length = 128 * distance  # 像素
image_size = 256  # 像素

# FOV 計算
fov_rad = 2 * np.arctan(image_size / (2 * focal_length))
fov_deg = np.degrees(fov_rad)

# 在 1000m 高度看下去能覆蓋的地面範圍
ground_coverage = 2 * distance * np.tan(fov_rad / 2)

print(f"\n📐 相機 FOV 分析:")
print(f"  焦距: {focal_length:.0f} 像素")
print(f"  FOV: {fov_deg:.2f} 度")
print(f"  在 {distance:.0f}m 高度能看到的地面範圍: {ground_coverage:.1f}m x {ground_coverage:.1f}m")

print(f"\n🚨 問題診斷:")
x_range = gaussian_stats['X']['max'] - gaussian_stats['X']['min']
print(f"  Gaussian X 範圍: {x_range:.1f} 米")
print(f"  相機可見範圍: {ground_coverage:.1f} 米")
print(f"  覆蓋比例: {ground_coverage / x_range * 100:.2f}%")

if ground_coverage < x_range * 0.1:
    print(f"\n  ⚠️ 相機只能看到場景的 {ground_coverage / x_range * 100:.2f}%!")
    print(f"  這解釋了為什麼渲染結果是灰色 - 大部分 Gaussians 不在視野內！")

print("\n💡 建議修復:")
print("  1. 使用局部座標系：讓 Gaussians 相對於相機中心")
print("  2. 降低焦距：增加 FOV")
print("  3. 確認 Gaussian 位置計算是否正確")
