import cv2
import numpy as np
import os

def diagnose_rpc_bias(ref_path, warped_path):
    # 1. 讀取影像
    img_ref = cv2.imread(ref_path)
    img_warped = cv2.imread(warped_path)

    if img_ref is None or img_warped is None:
        print("錯誤：找不到影像檔案，請檢查路徑。")
        return

    # 轉為灰階進行相位相關計算
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2. 自動計算位移量 (Phase Correlation)
    # 這裡會找出 img_warped 相對於 img_ref 需要平移多少
    shift, response = cv2.phaseCorrelate(gray_ref, gray_warped)
    dx, dy = shift
    print(f"--- 診斷結果 ---")
    print(f"偵測到的像素位移量: dx = {dx:.3f}, dy = {dy:.3f}")
    print(f"匹配信心度 (0-1): {response:.4f}")

    # 3. 執行補償平移
    M = np.float32([[1, 0, -dx], [0, 1, -dy]]) # 注意負號，因為是要把 warped 移回去
    img_corrected = cv2.warpAffine(img_warped, M, (img_ref.shape[1], img_ref.shape[0]))

    # 4. 視覺化對比
    # 建立一個交錯顯示 (Checkerboard) 或是 差異圖 (Difference)
    diff_before = cv2.absdiff(img_ref, img_warped)
    diff_after = cv2.absdiff(img_ref, img_corrected)
    
    # 疊加顯示供目測 (Alpha blending)
    overlay_before = cv2.addWeighted(img_ref, 0.5, img_warped, 0.5, 0)
    overlay_after = cv2.addWeighted(img_ref, 0.5, img_corrected, 0.5, 0)

    # 儲存結果供你檢查
    output_dir = os.path.dirname(warped_path)
    cv2.imwrite(os.path.join(output_dir, "diag_corrected.png"), img_corrected)
    cv2.imwrite(os.path.join(output_dir, "diag_diff_after.png"), diff_after)
    cv2.imwrite(os.path.join(output_dir, "diag_overlay_after.png"), overlay_after)

    print(f"\n診斷影像已儲存至: {output_dir}")
    print("請檢查 diag_overlay_after.png，如果物體邊緣重影消失，則確定是 RPC Bias。")

if __name__ == "__main__":
    REF_IMG = "/project/winston/mvsplat/debug/outputs/rgb_ref.png"
    WARPED_IMG = "/project/winston/mvsplat/debug/outputs/rgb_src_warped.png"
    
    diagnose_rpc_bias(REF_IMG, WARPED_IMG)