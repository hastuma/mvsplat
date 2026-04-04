# 3D 建築物 Volume 重建結果

## 論文方法實現總結

本項目實現了論文中描述的 **3D Volume 構建方法**：
> "The 3D volume V ∈ R^(H×W×D) is then constructed by lifting each pixel (i, j) from S to its corresponding 3D position according to D, where each voxel v(x, y, z) stores its instance ID and semantic class."

### 工作流程

```
OpenStreetMap → Instance Segmentation → Height Estimation → 3D Voxel Lifting → PLY Point Cloud
```

---

## 生成的數據

### 1. **3D 點雲** (建築物輪廓)
- **文件**: `buildings_3d_volume.ply`
- **格式**: PLY ASCII (SuperSplat 相容)
- **大小**: 70 MB
- **點數**: 1,274,397
- **座標系**: 地理座標 (經度, 緯度, 高度/公尺)
- **屬性**:
  - RGB 顏色 (基於 instance ID 的偽隨機顏色)
  - Instance ID (每棟建築物的唯一識別碼)
  - Semantic ID (建築物語義類別)

**使用方法**:
```bash
# 在 SuperSplat 中開啟
cd /project/winston/mvsplat/tests/osm/output
# 使用 SuperSplat GUI 開啟 buildings_3d_volume.ply
```

---

### 2. **Instance Segmentation 地圖**

#### a. `instance_segmentation.png` (灰度)
- 顯示每個像素的 instance ID
- 白色區域表示建築物，黑色區域表示背景
- 用於驗證建築物邊界檢測

#### b. `instance_segmentation_color.png` (彩色)
- 使用 OpenCV Jet 色彩映射
- 不同顏色代表不同的建築物實例
- 適合視覺化分析

---

### 3. **高度圖** (`height_map.png`)
- 顯示每個建築物的高度分布
- 使用 Turbo 色彩映射 (藍色=低, 紅色=高)
- 旁邊說明建築物高度的變異性

---

### 4. **元數據** (`metadata.json`)
- 位置資訊 (經度/緯度固定在交大校園)
- 掃描半徑: 200 公尺
- 體素網格大小: 459 × 505 × 100
- 偵測到的建築物數量: 25 棟
- 建築物語義分類表
- 前 20 棟建築物的詳細信息

```json
{
  "location": {"latitude": 24.786945, "longitude": 120.997926},
  "num_buildings": 25,
  "volume_shape": [459, 505, 100],
  "building_classes": {
    "residential": 1,
    "commercial": 2,
    "industrial": 3,
    "public": 4,
    "other": 5
  }
}
```

---

## 技術參數

| 參數 | 值 |
|-----|-----|
| **柵格解析度** | 1.0 m/pixel |
| **最大高度** | 100 公尺 |
| **搜尋半徑** | 200 公尺 |
| **OSM 建築物數量** | 25 棟 |
| **3D Volume 大小** | 459 × 505 × 100 |
| **總體素數** | 1,274,397 |

---

## 座標系說明

- **X 軸 (經度)**: 東西方向 [120.9954, 121.0003]
- **Y 軸 (緯度)**: 南北方向 [24.7850, 24.7890]
- **Z 軸 (高度)**: 垂直方向 [0, 100] 公尺

**座標轉換**:
- 經度 1° ≈ 111.32 km × cos(lat)
- 緯度 1° ≈ 111.32 km
- 解析度 1.0 m/pixel

---

## 論文方法對應

| 論文步驟 | 實現對應 |
|--------|--------|
| **Instance Segmentation (S)** | `instance_segmentation.png` - 每像素的 instance ID |
| **Depth Estimation (D)** | `height_map.png` - 從 OSM 高度/樓層數推估 |
| **3D Lifting** | Volume construction - 將 (i,j,h) 提升為 (x,y,z) |
| **Voxel Grid Storage** | PLY 點雲 - 每 voxel 存儲 instance_id + semantic_id |

---

## 使用建議

### 1. **在 SuperSplat 中查看**
```bash
mmsplat buildings_3d_volume.ply
```

### 2. **用 Python 處理 PLY**
```python
import numpy as np

# 讀取 PLY
with open('buildings_3d_volume.ply') as f:
    lines = f.readlines()

# 提取點雲座標
points = []
for line in lines[9:]:  # 跳過 header
    parts = line.split()
    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
    instance_id = int(parts[6])
    points.append((x, y, z, instance_id))
```

### 3. **用 CloudCompare 查看**
```bash
cloudcompare -GLOBAL_SHIFT AUTO buildings_3d_volume.ply
```

### 4. **用 Meshlab 查看**
```bash
meshlab buildings_3d_volume.ply
```

---

## 輸出檔案清單

```
output/
├── buildings_3d_volume.ply          (✓ SuperSplat 點雲 - 70 MB)
├── instance_segmentation.png        (✓ 灰度 instance 地圖)
├── instance_segmentation_color.png  (✓ 彩色 instance 地圖)
├── height_map.png                   (✓ 高度分佈可視化)
├── metadata.json                    (✓ 元數據 & 統計)
└── README.md                        (✓ 本檔)
```

---

## 進階參數調整

編輯 `osm.py` 中的以下參數：

```python
POINT = (24.786944654672414, 120.9979260167771)  # 改變位置
DIST = 500                                         # 改變搜尋半徑
GRID_RESOLUTION = 1.0                             # 改變像素大小
MAX_HEIGHT = 100                                   # 改變最大高度
```

---

## 依賴套件

```
osmnx >= 1.0.0        # OpenStreetMap 數據下載
numpy >= 1.19.0       # 數值計算
opencv-python >= 4.5  # 圖像處理
shapely >= 1.7.0      # 幾何操作
```

安裝:
```bash
pip install osmnx numpy opencv-python shapely
```

---

## 限制與改進

- ✓ 高度從 OSM 的 `height` 或 `building:levels` 推估
- ⚠ 簡化投影 (應使用 UTM 以提高精度)
- ⚠ 建築物高度可能不準確（OSM 數據品質不一）
- ⚠ 複雜多邊形建築物會被簡化
- 建議: 若有航拍深度圖，應直接使用而不是 OSM 高度

---

## 確認論文實現

✓ **Instance Segmentation (S)**
- ✓ 每像素關聯 instance ID
- ✓ `instance_segmentation.png` 存儲每個像素的實例標籤

✓ **Depth Estimation (D)**
- ✓ 從 OSM 建築物高度推估
- ✓ `height_map.png` 可視化深度分布

✓ **3D Volume (V)**
- ✓ Z 軸方向堆疊每個建築物
- ✓ `1,274,397` voxels 存儲 instance_id 和 semantic_id

✓ **輸出格式**
- ✓ PLY 點雲可匯入 SuperSplat/Meshlab/CloudCompare
- ✓ 每點包含 RGB、instance_id、semantic_id

---

**最後更新**: 2026-03-09
**生成者**: OSM to 3D Volume Pipeline
**論文參考**: "3D Structure Generation from Satellite Images"
