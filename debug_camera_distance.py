"""
實驗：改變相機高度來驗證 Gaussians 覆蓋範圍
（修正版：固定焦距，改變高度）
"""
import numpy as np

print("=" * 60)
print("🔬 實驗：不同相機高度的可見範圍")
print("=" * 60)

# 固定參數
image_size = 256
gaussian_range = 100  # Gaussians 分佈範圍約 100m

# 固定焦距 = 200 像素（這是我們設定的）
fixed_focal = 200

print("\n場景設定:")
print(f"  Gaussians 分佈範圍: ±{gaussian_range/2}m (共 {gaussian_range}m)")
print(f"  圖像尺寸: {image_size} × {image_size} 像素")
print(f"  固定焦距: {fixed_focal} 像素")

print("\n" + "-" * 60)
print("情況 A: 使用目前的公式 focal = distance / gsd")
print("-" * 60)

distances = [50, 100, 200, 500]
gsd = 0.5

for d in distances:
    focal = d / gsd  # 這個公式讓 GSD 保持固定
    fov_rad = 2 * np.arctan(image_size / (2 * focal))
    fov_deg = np.degrees(fov_rad)
    ground_coverage = 2 * d * np.tan(fov_rad / 2)
    
    print(f"\n📷 高度={d}m → 焦距={focal:.0f} → FOV={fov_deg:.1f}° → 可見={ground_coverage:.0f}m")

print("\n" + "-" * 60)
print("情況 B: 固定焦距，只改變高度")
print("-" * 60)

for d in distances:
    focal = fixed_focal  # 固定焦距
    fov_rad = 2 * np.arctan(image_size / (2 * focal))
    fov_deg = np.degrees(fov_rad)
    ground_coverage = 2 * d * np.tan(fov_rad / 2)
    
    # Gaussians 在視野中的佔比
    gs_in_view = min(gaussian_range, ground_coverage)
    gs_ratio = (gs_in_view / ground_coverage) ** 2 * 100
    
    print(f"\n📷 高度={d}m → 焦距={focal} → FOV={fov_deg:.1f}° → 可見={ground_coverage:.0f}m")
    if ground_coverage <= gaussian_range:
        print(f"   ✅ Gaussians 填滿整個畫面")
    else:
        print(f"   📊 Gaussians 佔 {gs_ratio:.1f}%, 黑邊佔 {100-gs_ratio:.1f}%")

print("\n" + "=" * 60)
print("💡 關鍵發現")
print("=" * 60)
print("""
目前的公式 `focal = distance / gsd` 的效果是:
  - 無論相機高度如何，GSD 都保持 0.5m/像素
  - 可見地面範圍永遠是 256 × 0.5 = 128m

這意味著:
  1. 把相機拉高 → 焦距也會變大 → FOV 變小
  2. 最終可見範圍保持不變 (128m)
  
所以你的想法（拉高相機會看到黑邊）需要用**固定焦距**才能實現！

如果你想測試黑邊效果，可以臨時改成固定焦距模式。
""")
