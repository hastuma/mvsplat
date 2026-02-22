"""
驗證修正後的相機 FOV
"""
import numpy as np

print("=" * 60)
print("修正後的相機 FOV 分析")
print("=" * 60)

# 參數
image_size = 256  # 像素
distance = 100.0  # 相機高度（米）
gsd = 0.5         # Ground Sample Distance（米/像素）

# 新的焦距計算
focal = distance / gsd  # = 100 / 0.5 = 200 像素

print(f"\n📐 參數:")
print(f"  圖像尺寸: {image_size} x {image_size} 像素")
print(f"  相機高度 (distance): {distance} 米")
print(f"  GSD: {gsd} 米/像素")
print(f"  新焦距: {focal} 像素")

# FOV 計算
fov_rad = 2 * np.arctan(image_size / (2 * focal))
fov_deg = np.degrees(fov_rad)
print(f"  FOV: {fov_deg:.2f} 度")

# 可見地面範圍
ground_coverage = 2 * distance * np.tan(fov_rad / 2)
print(f"  可見地面範圍: {ground_coverage:.2f} x {ground_coverage:.2f} 米")

# 預期範圍（GSD）
expected_coverage = image_size * gsd
print(f"\n📊 比較:")
print(f"  預期範圍 (GSD={gsd}m): {expected_coverage} x {expected_coverage} 米")
print(f"  實際可見範圍: {ground_coverage:.2f} 米")
print(f"  覆蓋率: {ground_coverage / expected_coverage * 100:.2f}%")

if abs(ground_coverage - expected_coverage) < 1:
    print(f"\n✅ 相機 FOV 正確覆蓋整個場景！")
else:
    print(f"\n⚠️ 覆蓋範圍仍有差異")

print(f"""
修正摘要：
  之前: focal = base_focal * distance = 128 * 100 = 12,800
        → FOV = 1.15 度
        → 可見範圍 = 2m x 2m (只有 1.56%!)
  
  之後: focal = distance / gsd = 100 / 0.5 = 200
        → FOV = {fov_deg:.2f} 度
        → 可見範圍 = {ground_coverage:.2f}m x {ground_coverage:.2f}m (100%!)
""")
