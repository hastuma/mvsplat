# SkySplat × Depth Anything V2：本週進度報告

**日期**：2026-04-27  
**最新訓練資料夾**：`outputs/2026-04-27/16-15-31`（loss weight=1.0）  
**場景**：JAX_004（overfit 單一 scene，online random crop）

---

## 本週目標

1. 加入 **Depth Anything V2 (DAMv2)** 作為 monocular depth 教師，解決 cost-volume depth 趨近平面的問題
2. 引入 **Online Random Crop**，增加視差多樣性，讓 photometric loss 真正能約束深度
3. 分析並嘗試解決 depth 從 **smooth → binary（非白即黑）** 的現象

---

## 問題 1：深度趨近於平面（根本原因）

### 症狀
訓練後期，所有 Gaussian 的高度預測趨近於一個平面（HEIGHT_OFF），.ply 點雲 Z 軸幾乎無起伏。

### 根本原因
> **Input views 用一個平面就能解釋（insufficient parallax）**

overfit 資料集使用固定裁切位置，3 張 context view 的視差不足。Photometric loss（MSE + LPIPS）用單一高度的平面就能滿足，模型沒有動機學習真實高度差。

---

## 問題 2：DAMv2 的方向相反（已修正）

### 症狀
訓練後期 vis_damv2 中間圖顯示建物=白（大距離）、地面=黑（小距離）——物理上相反，建物應比地面更靠近衛星（小距離=黑）。

### 根本原因
DAMv2 是用地面視角影像訓練的。在衛星正射影像中，它把建物屋頂的複雜紋理解讀為「遠處景物」，給出**大** depth 值；把平滑地面解讀為「近處」，給出**小** depth 值。這與 cost-volume 的方向（建物=小距離）完全相反，導致 Pearson loss 把 cost-volume 推向錯誤方向。

### Sign Convention

| 訊號 | 建物 | 地面 |
|---|---|---|
| `cv_depth`（公尺，物理正確） | **小**（靠近衛星） | **大**（遠離衛星） |
| `damv2_original` | **大**（複雜紋理→遠景） | **小**（平滑→近景） |
| `-damv2`（取反，對齊後） | **小** ✓ | **大** ✓ |

### 修復
```python
# src/model/model_wrapper.py（兩處：loss 計算 + observation）
damv2_depth_v = -damv2(ctx_v.to(device))   # 加負號
```

### 解讀 WandB `loss/damv2_pearson`
| 顯示值 | 實際 Pearson ρ | 意義 |
|---|---|---|
| **−0.7 ~ −0.9** | +0.7 ~ +0.9 | ✅ 理想：depth 和 DAMv2 高度正相關 |
| **≈ 0** | ≈ 0 | ❌ 崩潰訊號：depth std→0（完全平面） |
| **> 0** | 負值 | ❌ Sign convention 仍錯 |

> loss = −0.9 是最佳狀態。從 −0.9 突然升到 ≈ 0 才是問題。

---

## 問題 3：訓練中期 depth 突然崩潰（已修正）

### 症狀（`outputs/2026-04-25/17-30-13`，warm_up_steps=10000）
- Step 6000：Z std=2.1m，點雲有合理起伏 ✅
- **Step 6800：Z 均值從 25m 突然跳至 34m，永久固定** ❌
- Step 7500+：vis_depth 幾乎全黑，Pearson loss → 0

### 根本原因
```
warm_up_steps = 10000
Step 6800: LR = 68% of lr_max，仍在上升中
```
某個 batch 觸發大梯度更新，把 soft-argmax PDF 推到 near 極端。一旦全部像素都預測 near，depth std→0，Pearson loss 梯度消失，永遠無法恢復。

### 修復
```yaml
# config/experiment/dfc2019.yaml
optimizer:
  warm_up_steps: 2000   # LR 在 step 2000 達峰後開始 cosine 下降

trainer:
  gradient_clip_val: 0.3  # 從 0.5 降低，防止大梯度觸發崩潰
```

**效果**：Step 6800 時 LR 已降至峰值的 ~30%，不再觸發大更新。`outputs/2026-04-26/20-37-21` 確認無崩潰。

---

## 問題 4：Online Random Crop 實作

### 背景與動機
固定裁切位置的 overfit 資料集每 epoch 見到相同的 4 張影像 + 相同視角組合，視差固定且不足。目標：讓每個 training step 有不同的 crop 位置和影像組合，透過多樣化視差讓 photometric loss 真正約束深度。

### 實作細節（`src/dataset/dataset_dfc2019.py`）

**啟動條件**：config `raw_scenes_dir` 非空且 stage=train 時啟用。Val/test 仍使用原本 pre-cropped patches。

**每個 `__getitem__` 的流程**：
1. 從 `raw_scenes_dir`（全解析度 2048×2048 TIF）依場景分組，依 `overfit_to_scene` 過濾
2. 隨機選 1 master image + 3 slave images（從該 scene 的所有影像中 sample）
3. 在 master 上隨機選取 crop 位置 `(x, y)` ∈ [0, 1792]²（確保 +256 不超出邊界）
4. 用 RPC inverse 將 master crop 的左上角像素轉為地理座標 (lat, lon)
5. 用 RPC forward 將 (lat, lon) 投影到每個 slave，得到對應的 crop 位置
6. 每張影像各自裁切 256×256，更新 RPC 的 LINE_OFF 和 SAMP_OFF

**邊界保護（retry loop）**：
- Master：`randint(0, W-256)` 保證完全在影像內
- Slave：計算 crop 在影像內的 overlap ratio，要求 ≥ 80%
- 不滿足則重新隨機 crop 位置，最多重試 10 次

**旋轉影像過濾**：
```
JAX_068_001: col axis = +87.3° (朝北，90° 旋轉)  →  過濾
JAX_068_004: col axis = +0.0°  (正常朝東)        →  保留
```
用 RPC Jacobian 計算每張影像的列軸偏東角度。過濾 `|angle| > 45°` 的影像，避免不同方向的影像在同一 batch 中破壞 cost-volume feature correlation。

**效果（JAX_068 場景）**：94 張旋轉影像被過濾，保留 18 張正常影像。

### 配置
```yaml
# config/dataset/dfc2019.yaml
raw_scenes_dir: /project/winston/mvsplat/datasets/DFC2019/Track3-RGB-1
online_crops_per_scene: 500  # 每 scene 每 epoch 500 個隨機 crop

# config/experiment/dfc2019.yaml
dataset:
  overfit_to_scene: JAX_004   # 限定單一 scene；null = 全部 scene
```

**切換方式**：
- 單 scene overfit：`overfit_to_scene: JAX_004`
- 全部 scene 訓練：`overfit_to_scene: null`
- 關閉 online crop（回到預處理模式）：`raw_scenes_dir: ""`

---

## 問題 5：Depth 從 Smooth → Binary（本週新發現，未解決）

### 症狀
訓練初期（step 0~3000）：depth map 有連續灰階（smooth），點雲 Z std ≈ 2~5m  
訓練後期（step 4000+）：depth map 變成非白即黑（binary），Pearson loss 從 -0.9 升至 -0.75

| 訓練 | Bimodal 出現時機 | Step 2000 時 Z std |
|---|---|---|
| damv2_weight=0.1（上上次） | Step ~3000 | 3.9m |
| damv2_weight=1.0（最新） | Step ~5000 | 4.8m |

### 根本原因：Softmax 訓練的數學必然性

Cost-volume 的深度預測：
```python
logits = depth_head_lowres(raw_correlation)  # [B*V, D=32, H, W]
pdf    = F.softmax(logits, dim=1)
depth  = (candidates * pdf).sum(dim=1)       # soft-argmax
```

`depth_head_lowres` 是兩層 Conv2d（共 ~37,000 個 weights）。不論 loss 設定如何，只要有任何 loss 偏好某個深度候選，梯度就會：

```
∂L/∂l_winner < 0  →  l_winner 增大
∂L/∂l_loser  > 0  →  l_loser  減小
```

這個 logit 分離效應是**累積的**、**不可逆的**。隨訓練進行，logit 差距持續增大，softmax 趨近 one-hot，soft-argmax 退化為 hard-argmax，輸出只剩 near（黑）或 far（白）兩個值。

### 為什麼調高 Loss Weight 沒用？

Loss weight 改變的是「哪個深度候選是 winner」的競爭強度，但無法阻止「winner 的 logit 持續增大」這個動力學。調高 weight 只是延後了 bimodalization（約 1500 步），沒有根本解決。

### 為什麼 Pearson loss 從 -0.9 升到 -0.75？

1. **Gradient saturation**：PDF 接近 one-hot 時，`∂softmax/∂logit = p(1-p) → 0`，Pearson loss 的梯度幾乎無法回流到 `depth_head_lowres`
2. **Multi-objective tradeoff**：Photometric loss 找到一個 depth 配置，雖然讓 Pearson 稍微變差（-0.9→-0.75），但 total loss 仍然下降（∆L_mse 降低量 > 0.1 × ∆L_pearson 增加量）
3. **Binary vs smooth Pearson**：ρ(binary_signal, smooth_signal) < ρ(smooth_signal, smooth_signal) 是數學事實，即使 binary 方向正確，Pearson 上限約 0.75-0.85

### 提出的解法：PDF Entropy Regularization

直接懲罰 softmax 分布「過度自信化」：

$$L_\text{entropy} = -H(\text{pdf}) = \sum_k p_k \log p_k$$

- 均勻分布（D=32 候選）時：$H = \log 32 \approx 3.47$，$L_\text{entropy} \approx -3.47$（最低）
- One-hot 時：$H = 0$，$L_\text{entropy} = 0$（最高）

最小化 $L_\text{entropy}$（即最大化 entropy）會建立**持續對抗 logit 增大的反向梯度**。即使其他 loss 讓 logits 增大，entropy loss 會對抗這個趨勢。這是 loss weight 調整做不到的事——它直接作用在「不讓 softmax 變 one-hot」這件事本身。

---

## 當前配置（最新）

```yaml
# config/loss/mse.yaml
mse:
  weight: 1.0

# config/loss/lpips.yaml
lpips:
  weight: 1.0
  apply_after_step: 0

# config/experiment/dfc2019.yaml
optimizer:
  lr: 2e-4
  warm_up_steps: 2000

trainer:
  gradient_clip_val: 0.3
  max_steps: 20000

dataset:
  overfit_to_scene: JAX_004
  image_shape: [256, 256]

train:
  damv2_loss_weight: 0.1
  damv2_loss_warmup_steps: 500
```

---

## 下週方向

1. **實作 PDF Entropy Regularization**：在 `depth_predictor_multiview.py` 中把 `pdf` 存入 `vis_dump`，在 `model_wrapper.py` 加入 entropy loss 項，新增 `depth_entropy_weight` 超參數（建議初始值 0.1）
2. **Scale-shift aligned MSE**（silog loss）：同時約束 depth 的方向和幅度，取代純 Pearson
3. **擴大訓練資料**：從 overfit 單一 scene（JAX_004）擴展到多 scene（`overfit_to_scene: null`）

---

## 程式碼改動摘要

| 檔案 | 改動內容 |
|---|---|
| `src/model/model_wrapper.py` | DAMv2 輸出取反（sign fix）；`_damv2_observe` 和 `_damv2_pearson_loss` 兩處同步修改 |
| `src/dataset/dataset_dfc2019.py` | Online random crop 模式；旋轉影像過濾（|angle|>45°）；slave overlap 檢查（≥80%，最多 retry 10 次）；`overfit_to_scene` 支援 |
| `config/dataset/dfc2019.yaml` | 新增 `raw_scenes_dir`（指向全解析度 TIF）、`online_crops_per_scene: 500` |
| `config/experiment/dfc2019.yaml` | `warm_up_steps` 10000→2000；`gradient_clip_val` 0.5→0.3；`overfit_to_scene: JAX_004` |
| `config/loss/mse.yaml` | `weight: 1.0`（不變） |
| `config/loss/lpips.yaml` | `weight: 1.0`（從 0.05 調高） |
| `skysplat.md` | 更新 DAMv2 loss 計算細節、sign convention、WandB 解讀指南 |
