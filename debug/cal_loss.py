import torch
import lpips
import numpy as np
from PIL import Image
import os

def calculate_metrics(image_path):
    # 1. 檢查檔案是否存在
    if not os.path.exists(image_path):
        print(f"Error: 找不到檔案 {image_path}")
        return

    # 2. 讀取 .tif 影像
    # 使用 PIL 讀取，並確保轉為 RGB (3通道)
    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    
    # 3. 建立一張全黑的影像 (與原圖尺寸相同)
    black_np = np.zeros_like(img_np)

    # --- 計算 MSE Loss ---
    # 通常 MSE 會在 [0, 1] 範圍內計算
    img_01 = img_np.astype(np.float32) / 255.0
    black_01 = black_np.astype(np.float32) / 255.0
    mse_loss = np.mean((img_01 - black_01) ** 2)

    # --- 計算 LPIPS Loss ---
    # LPIPS 需要輸入為 Torch Tensor，且數值範圍需縮放到 [-1, 1]
    # 維度順序需為 (Batch, Channel, Height, Width)
    
    # 初始化 LPIPS 模型 (使用 AlexNet 作為後端，這比較常用)
    loss_fn_alex = lpips.LPIPS(net='alex')

    def to_tensor(img):
        # HWC -> CHW, [0, 255] -> [-1, 1]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        img_t = (img_t / 127.5) - 1.0
        return img_t.unsqueeze(0) # 增加 Batch 維度

    img_tensor = to_tensor(img_np)
    black_tensor = to_tensor(black_np)

    # 關閉梯度計算以節省記憶體
    with torch.no_grad():
        lpips_loss = loss_fn_alex(img_tensor, black_tensor).item()

    # 4. 輸出結果
    print(f"影像路徑: {image_path}")
    print(f"影像尺寸: {img_np.shape}")
    print("-" * 30)
    print(f"MSE Loss:   {mse_loss:.6f}")
    print(f"LPIPS Loss: {lpips_loss:.6f}")

if __name__ == "__main__":
    target_path = "/project/winston/datasets/DFC2019/overfit/validation/JAX_004_p0000/JAX_004_010_RGB.tif"
    calculate_metrics(target_path)