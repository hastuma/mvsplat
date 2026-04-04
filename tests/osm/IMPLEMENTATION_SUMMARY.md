# 論文方法實現總結 - OSM 到 3D Volume

## ✓ 實現完成

根據論文要求的 **3D Structure Generation** 方法已完整實現：

```
"The 3D volume V ∈ R^(H×W×D) is then constructed by lifting each pixel 
(i, j) from S to its corresponding 3D position according to D, where each 
voxel v(x, y, z) stores its instance ID and semantic class."
```

---

## 工作流程實現

### 1️⃣ **Step 1: Instance Segmentation (S)**
- ✓ 從 OpenStreetMap 提取建築物邊界
- ✓ 生成 instance_segmentation_map (每像素一個 instance ID)
- ✓ 輸出: `instance_segmentation.png` (1149×1166 pixels)

### 2️⃣ **Step 2: Depth Estimation (D)**
- ✓ 從 OSM `height` 或 `building:levels` 提取資訊
- ✓ 預設規則: `建築高度 = building:levels × 3.5m`
- ✓ 輸出: `height_map.png` (0-34 公尺分布)

### 3️⃣ **Step 3: 3D Voxel Lifting**
- ✓ 將每個像素 (i,j) 提升至 3D 座標 (x,y,z)
- ✓ X, Y = 地理座標 (經度/緯度)
- ✓ Z = 每棟建築物的高度 (0 ~ h)
- **體素編碼**: 每個 voxel 存儲 `(instance_id, semantic_id)`

### 4️⃣ **Step 4: 3D Volume 構建**
- ✓ Volume 大小: **459 × 505 × 100** (height × width × depth)
- ✓ 非零 Voxel 數量: **1,274,397**
- ✓ 實例數: **25 棟建築物**
- ✓ 每個 Voxel 存儲 instance_id 和 semantic_id

### 5️⃣ **Step 5: SuperSplat 導出**
- ✓ 格式: **PLY 點雲** (ASCII 格式)
- ✓ 檔案大小: **70 MB**
- ✓ 點數: **1,274,397**
- ✓ 屬性: RGB 顏色 + instance_id + semantic_id

---

## 📊 處理結果

### 地理涵蓋範圍
- **位置**: 國立交通大學校園
- **座標**: (24.786945°N, 120.997926°E)
- **搜尋半徑**: 200 公尺
- **涵蓋面積**: 0.222 km²

### 建築物統計
| 指標 | 值 |
|-----|-----|
| 統計建築物數量 | 25 棟 |
| 平均高度 | 12.0 公尺 |
| 最高建築 | 34.0 公尺 (Instance #12, 166,985 點) |
| 最矮建築 | 3.0 公尺 (Instance #14, 3,780 點) |
| 建築密度 | 112.5 棟/km² |

### 高度分佈
```
  0-3.4m   : ████████░ 244,264 點 (19.2%)
  3.4-6.8m : ██████░░░ 183,198 點 (14.4%)
  6.8-10.2m: ███████░░ 211,046 點 (16.6%)
  10.2-13.6m: █████░░░░ 137,976 點 (10.8%)
  13.6-17.0m: ████░░░░░ 109,901 點 (8.6%)
  17.0-20.4m: █████░░░░ 144,576 點 (11.3%)
  20.4-23.8m: ███░░░░░░ 82,626 點 (6.5%)
  23.8-27.2m: ███░░░░░░ 90,288 點 (7.1%)
  27.2-30.6m: ██░░░░░░░ 51,438 點 (4.0%)
  30.6-34.0m: ░░░░░░░░░ 19,084 點 (1.5%)
```

---

## 📁 生成的檔案

```
output/
├── 📄 buildings_3d_volume.ply           [70 MB] ← SuperSplat 主檔
├── 🖼️ instance_segmentation.png         [7.2 KB]
├── 🖼️ instance_segmentation_color.png   [8.9 KB]
├── 🖼️ height_map.png                    [14 KB]
├── 📊 metadata.json                     [4.2 KB]
├── 📊 instance_statistics.json          [自動生成]
├── 📄 README.md                         [詳細說明]
└── 🐍 analyze_ply.py                    [分析工具]
```

---

## 🎯 使用方法

### 1. 在 SuperSplat 中查看
```bash
cd /project/winston/mvsplat/tests/osm/output
# 使用 SuperSplat 選擇 → 打開檔案 → 選擇 buildings_3d_volume.ply
```

### 2. 在 CloudCompare 或 Meshlab 中查看
```bash
cloudcompare buildings_3d_volume.ply
meshlab buildings_3d_volume.ply
```

### 3. 用 Python 處理
```python
import numpy as np
import json

# 讀取統計數據
with open('instance_statistics.json') as f:
    stats = json.load(f)
    
# 第一棟建築物
print(stats[0]['centroid'])  # {'x': ..., 'y': ..., 'z': ...}
print(stats[0]['bounds']['z'])  # [0, 30.0]
```

---

## 🔧 技術細節

### 座標系統
- **X 軸 (經度)**: 東西方向 [120.9954, 121.0003]°
- **Y 軸 (緯度)**: 南北方向 [24.7850, 24.7890]°
- **Z 軸 (高度)**: 垂直方向 [0, 34]m

### 座標轉換
```
地理座標 (lon, lat, h) → 像素坐標 (i, j, k)

scale_x = 111.32 km/°  × cos(lat)
scale_y = 111.32 km/°

px = (lon - min_lon) × scale_x / grid_resolution
py = (max_lat - lat) × scale_y / grid_resolution  
pz = h / grid_resolution
```

### 語義分類
| 類別 ID | 名稱 | 點數 | 比例 |
|--------|------|------|------|
| 5 | other (未分類) | 1,274,397 | 100% |

*注: OSM 中交大校園的建築物大多未被具體分類*

---

## ✅ 論文方法驗證

| 論文要求 | 實現狀態 | 對應檔案 |
|--------|--------|--------|
| Instance Segmentation S | ✓ 完成 | `instance_segmentation_*.png` |
| Depth Estimation D | ✓ 完成 | `height_map.png` |
| 3D Lifting | ✓ 完成 | PLY 點雲座標 |
| Voxel Storage (instance_id, semantic_id) | ✓ 完成 | PLY properties |
| Volume V ∈ R^(H×W×D) | ✓ 完成 | 459 × 505 × 100 |

---

## 📈 進階使用

### 調整參數重新生成
編輯 `osm.py`:
```python
POINT = (latitude, longitude)   # 改變位置
DIST = 500                       # 改變搜尋半徑 (公尺)
GRID_RESOLUTION = 0.5            # 改變解析度 (更小 = 更精細)
MAX_HEIGHT = 150                 # 改變最大高度
```

### 分析 PLY 檔案
```bash
cd output
python analyze_ply.py
```

### 批次處理多個區域
```python
locations = [
    (24.786945, 120.997926),  # 交大
    (25.033611, 121.564722),  # 其他地點
]
for lat, lon in locations:
    # 重新設置 POINT 並運行
```

---

## ⚠️ 已知限制

- ⚠ OSM 建築物高度數據品質不一，部分為估計值
- ⚠ 複雜多邊形建築會被簡化為凸包
- ⚠ 簡化投影 (應使用 UTM 投影以提高精度)
- ⚠ 建築物若編號不連續，instance_id 會有間隙

## 🚀 改進建議

1. **使用航拍深度圖**: 直接從 DEM/DSM 獲取而非 OSM 估計
2. **改進投影**: 使用 UTM 投影系統以提高座標精度
3. **複雜形狀支持**: 支援非凸多邊形的精確表示
4. **語義分割**: 集成深度學習模型以提升建築物分類

---

## 📞 技術支持

生成進度日誌保存在: `/project/winston/mvsplat/tests/osm/`
- `osm.py` - 主程式 (1-5 步驟完整實現)
- `output/README.md` - 詳細文檔
- `output/analyze_ply.py` - 分析工具

---

**✓ 恭賀！論文方法已成功實現**

實現日期: 2026-03-09
實現語言: Python 3.12
相依套件: osmnx, numpy, opencv-python, shapely

