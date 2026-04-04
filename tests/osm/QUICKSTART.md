# 🚀 快速開始指南

## 輸出檔案位置
```
/project/winston/mvsplat/tests/osm/output/
```

## 📦 主要成果 - 3 個必看檔案

### 1️⃣ PLY 點雲 (SuperSplat 用)
```
buildings_3d_volume.ply  (70 MB)
```
**用途**: 在 SuperSplat 或其他 3D 軟體中查看 3D 建築物
**格式**: ASCII PLY，包含 1,274,397 個點

### 2️⃣ 可視化圖像
```
instance_segmentation_color.png  (彩色 instance 地圖)
height_map.png                   (高度分佈)
```
**用途**: 快速查看 instance 分佈和高度資訊

### 3️⃣ 元數據
```
metadata.json              (基本信息 & 設置參數)
instance_statistics.json   (各建築物詳細統計)
```
**用途**: 了解數據統計和建築物清單

---

## 📊 快速統計

| 項目 | 值 |
|-----|-----|
| **建築物數量** | 25 棟 |
| **總點數** | 1,274,397 |
| **涵蓋面積** | 0.222 km² |
| **高度範圍** | 0-34 公尺 |
| **建築密度** | 112.5 棟/km² |

---

## 🎯 使用方法 (3 種)

### 方法 1: 用 SuperSplat 查看 (推薦)
```bash
cd /project/winston/mvsplat/tests/osm/output
# 打開 SuperSplat → File → Open → 選擇 buildings_3d_volume.ply
```

### 方法 2: 用 Python 讀取
```python
import json

# 讀取元數據
with open('metadata.json') as f:
    meta = json.load(f)
    print(f"建築物數: {meta['num_buildings']}")

# 讀取各建築物統計
with open('instance_statistics.json') as f:
    stats = json.load(f)
    print(f"最大建築: {stats[0]['point_count']} 點")
```

### 方法 3: 執行分析工具
```bash
cd output
python analyze_ply.py
```
輸出詳細統計報告和圖表

---

## 🎨 可視化結果

| 圖像 | 說明 |
|-----|-----|
| `instance_segmentation.png` | 灰度地圖，白色=建築物，黑色=背景 |
| `instance_segmentation_color.png` | 五彩地圖，不同顏色=不同建築物 |
| `height_map.png` | 熱力圖，藍色(矮)→紅色(高) |

---

## 📋 論文方法驗證清單

- ✅ **Instance Segmentation**: `instance_segmentation_*.png`
- ✅ **Depth Estimation**: `height_map.png` (0-34m)
- ✅ **3D Lifting**: PLY 座標系 (lon, lat, height)
- ✅ **Voxel Storage**: instance_id + semantic_id
- ✅ **3D Volume**: 459 × 505 × 100

---

## 💡 進階使用

### 改變位置重新生成
編輯 `/project/winston/mvsplat/tests/osm/osm.py`:
```python
POINT = (緯度, 經度)  # 改變為你要的位置
DIST = 500            # 搜尋半徑
```
然後執行: `python osm.py`

### 處理其他城市
```python
# 台北市
POINT = (25.0330, 121.5654)

# 台中市
POINT = (24.1372, 120.6737)

# 高雄市
POINT = (22.6273, 120.3014)
```

---

## 📚 檔案樹狀圖

```
osm/
├── osm.py                           (主程式 - 完整實現論文方法)
├── IMPLEMENTATION_SUMMARY.md        (技術細節)
├── QUICKSTART.md                    (本檔)
└── output/
    ├── buildings_3d_volume.ply      ⭐ SuperSplat 用
    ├── instance_segmentation.png    (灰度)
    ├── instance_segmentation_color.png  (彩色)
    ├── height_map.png               (高度分佈)
    ├── metadata.json                (基本信息)
    ├── instance_statistics.json     (詳細統計)
    ├── analyze_ply.py               (分析工具)
    └── README.md                    (詳細文檔)
```

---

## ✨ 論文方法流程圖

```
OpenStreetMap
    ↓
提取建築物 (175 棟)
    ↓
Instance Segmentation Map (1149×1166)
    ↓
高度預估 (OSM height/levels)
    ↓
3D Lifting (座標轉換)
    ↓
Voxel Grid (459×505×100)
    ↓
PLY 點雲 (1,274,397 點)
    ↓
SuperSplat ✅
```

---

## 🎓 技術要點

**座標系統**:
- X: 經度 (東西)
- Y: 緯度 (南北)
- Z: 高度 (公尺)

**Voxel 編碼**:
- instance_id: 1-25 (25 棟建築物)
- semantic_id: 1-5 (語義分類)

**點雲屬性**:
- RGB 顏色 (基於 instance ID)
- instance_id (整數)
- semantic_id (整數)

---

## ❓ 常見問題

**Q: PLY 檔案很大，如何減小?**
A: 改變 `GRID_RESOLUTION` 為更大值 (如 2.0 代表 2m/pixel)

**Q: 如何改變建築物顏色?**
A: 編輯 `osm.py` 中的配色方案

**Q: 能否導出為其他格式?**
A: 用 CloudCompare/Meshlab 進行格式轉換

**Q: 高度數據準確嗎?**
A: OSM 高度為估計值，若需精確用 LiDAR/DEM 數據

---

**最後更新**: 2026-03-09
**作者**: CV Research Team
**相關論文**: "3D Structure Generation from Satellite Images"
