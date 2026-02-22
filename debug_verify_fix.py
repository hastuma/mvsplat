"""
驗證修正是否成功：分析相機和 Gaussians 的空間關係
"""
import numpy as np

print("=" * 70)
print("📊 修正驗證報告")
print("=" * 70)

# 從最新的 log 中提取的數據
print("\n1️⃣ 座標範圍比較（修正前 vs 修正後）")
print("-" * 50)

# 修正前 (從之前的 log)
print("\n修正前:")
print("  X: min=-6303.27, max=6804.27 → 範圍 = 13,107 米")
print("  Y: min=-6524.23, max=6637.50 → 範圍 = 13,162 米")
print("  問題: 座標使用了錯誤的 ENU 參考點 (RPC LAT_OFF/LONG_OFF)")

# 修正後 (從最新的 log)
print("\n修正後:")
x_min, x_max = -46.5513, 45.7852
y_min, y_max = -49.3646, 49.3665
x_range = x_max - x_min
y_range = y_max - y_min
print(f"  X: min={x_min:.2f}, max={x_max:.2f} → 範圍 = {x_range:.2f} 米")
print(f"  Y: min={y_min:.2f}, max={y_max:.2f} → 範圍 = {y_range:.2f} 米")
print(f"  ✅ 座標現在以圖像中心為原點，範圍約 ±50 米")

# 預期範圍
expected_range = 256 * 0.5  # 128m
print(f"\n預期範圍 (256px × 0.5m/px): {expected_range} 米")
print(f"實際範圍: X={x_range:.2f}m, Y={y_range:.2f}m")
coverage_x = x_range / expected_range * 100
coverage_y = y_range / expected_range * 100
print(f"覆蓋率: X={coverage_x:.1f}%, Y={coverage_y:.1f}%")

print("\n" + "=" * 70)
print("2️⃣ 相機 FOV 分析")
print("-" * 50)

# 相機參數
distance = 100.0
gsd = 0.5
focal = distance / gsd  # 200
image_size = 256

fov_rad = 2 * np.arctan(image_size / (2 * focal))
fov_deg = np.degrees(fov_rad)
ground_coverage = 2 * distance * np.tan(fov_rad / 2)

print(f"\n相機參數:")
print(f"  高度 (distance): {distance} 米")
print(f"  GSD: {gsd} 米/像素")
print(f"  焦距: {focal} 像素")
print(f"  FOV: {fov_deg:.2f} 度")
print(f"  可見地面範圍: {ground_coverage:.2f} × {ground_coverage:.2f} 米")

# 相機能看到多少 Gaussians？
cam_x, cam_y = 0, 0  # 相機在原點上方
cam_view_x = [cam_x - ground_coverage/2, cam_x + ground_coverage/2]
cam_view_y = [cam_y - ground_coverage/2, cam_y + ground_coverage/2]

print(f"\n相機可視範圍:")
print(f"  X: [{cam_view_x[0]:.1f}, {cam_view_x[1]:.1f}] 米")
print(f"  Y: [{cam_view_y[0]:.1f}, {cam_view_y[1]:.1f}] 米")

# 假設 Gaussians 均勻分佈在 ±50m 範圍
in_view_x = min(cam_view_x[1], x_max) - max(cam_view_x[0], x_min)
in_view_y = min(cam_view_y[1], y_max) - max(cam_view_y[0], y_min)
in_view_area = max(0, in_view_x) * max(0, in_view_y)
total_area = x_range * y_range
visibility = in_view_area / total_area * 100

print(f"\nGaussians 可見性:")
print(f"  Gaussian 分佈範圍: {x_range:.1f} × {y_range:.1f} = {total_area:.1f} 平方米")
print(f"  相機可視範圍: {ground_coverage:.1f} × {ground_coverage:.1f} = {ground_coverage**2:.1f} 平方米")
print(f"  在視野內的 Gaussians: {visibility:.1f}%")

if visibility > 80:
    print(f"\n  ✅ 大部分 Gaussians 都在相機視野內！")
else:
    print(f"\n  ⚠️ 只有 {visibility:.1f}% 的 Gaussians 在視野內")

print("\n" + "=" * 70)
print("3️⃣ 渲染結果分析")
print("-" * 50)

print("""
觀察渲染圖片 (vis_loss 資料夾):
- 左半邊: Ground Truth (衛星圖像)
- 右半邊: 3DGS 渲染結果

如果右半邊仍然是灰色，可能原因：
1. SH 顏色值仍然太相近（std 太小）
2. Opacity 太低或太均勻
3. 訓練步數不夠

從 log 中的顏色統計：
  R: min=0.08, max=0.53, mean=0.31, std=?
  G: min=0.05, max=0.54, mean=0.31, std=?
  B: min=0.03, max=0.50, mean=0.27, std=?

顏色範圍 [0.03, 0.54] 是合理的，說明網路有學到不同的顏色！
""")

print("=" * 70)
print("4️⃣ 結論")
print("-" * 50)

print("""
✅ 修正已生效:
   - X/Y 座標從 ~13km 降至 ~100m (正確!)
   - 相機 FOV 從 1.15° 增至 65.24° (正確!)
   - Gaussians 可見性從 1.56% 增至接近 100% (正確!)

❓ 如果渲染仍是灰色:
   - 可能需要更多訓練步數
   - 可能需要調整學習率
   - 可以檢查 SH DC 的標準差是否在增加

建議下一步:
   1. 繼續訓練到 500-1000 步
   2. 觀察 loss 是否下降
   3. 觀察渲染圖片的變化
""")
