#!/usr/bin/env python
"""
將資料夾中的所有照片製作成影片
"""

import os
import sys
import cv2
import numpy as np
import re
from pathlib import Path

def natural_sort_key(path):
    """提取檔名中的數字用於自然排序"""
    filename = str(path)
    # 將字符串分割為數字和非數字部分
    parts = []
    for match in re.finditer(r'(\d+|\D+)', filename):
        part = match.group()
        # 如果是數字，轉換為整數；否則保持字符串
        parts.append((int(part) if part.isdigit() else part))
    return parts

def make_video_from_images(image_folder, output_video_path, fps=10, video_format='mp4'):
    """
    將資料夾中的照片製作成影片
    
    Args:
        image_folder: 照片所在的資料夾路徑
        output_video_path: 輸出影片路徑
        fps: 幀率（預設 10 fps）
        video_format: 影片格式（'mp4'、'avi' 等，預設 'mp4'）
    """
    
    # 檢查資料夾是否存在
    if not os.path.isdir(image_folder):
        print(f"❌ 錯誤：資料夾不存在 - {image_folder}")
        return False
    
    # 支援的圖片格式
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # 獲取資料夾中的所有照片
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_folder).glob(f'*{ext}'))
        image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
    
    # 移除重複並按自然排序（數字大小順序）
    image_files = list(set(image_files))
    image_files = sorted(image_files, key=natural_sort_key)
    if not image_files:
        print(f"❌ 錯誤：資料夾中沒有找到照片 - {image_folder}")
        return False
    
    print(f"✅ 找到 {len(image_files)} 張照片")
    print(f"資料夾：{image_folder}")
    print(f"輸出：{output_video_path}")
    
    # 顯示前 5 張照片的檔名
    print("\n前 5 張照片：")
    for i, img_file in enumerate(image_files[:5]):
        print(f"  {i+1}. {img_file.name}")
    if len(image_files) > 5:
        print(f"  ... 共 {len(image_files)} 張照片")
    
    # 讀取第一張照片以確定視頻尺寸
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"❌ 錯誤：無法讀取第一張照片 - {image_files[0]}")
        return False
    
    height, width = first_image.shape[:2]
    print(f"\n影片尺寸：{width}x{height}")
    print(f"幀率（FPS）：{fps}")
    
    # 創建影片編寫器
    if video_format.lower() == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif video_format.lower() == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ 錯誤：無法建立影片編寫器")
        return False
    
    # 逐張讀取照片並寫入影片
    print("\n開始製作影片...")
    for idx, image_file in enumerate(image_files, 1):
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"⚠️ 警告：無法讀取照片 {idx}/{len(image_files)} - {image_file.name}")
            continue
        
        # 若圖片尺寸不符，調整大小
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
        
        # 每 10 張照片顯示進度
        if idx % max(1, len(image_files) // 10) == 0 or idx == len(image_files):
            print(f"  進度：{idx}/{len(image_files)}")
    
    out.release()
    
    # 驗證輸出檔案
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # 轉換為 MB
        print(f"\n✅ 影片製作完成！")
        print(f"輸出路徑：{output_video_path}")
        print(f"檔案大小：{file_size:.2f} MB")
        return True
    else:
        print(f"\n❌ 錯誤：影片製作失敗")
        return False


if __name__ == "__main__":
    # 預設參數
    image_folder = "/project/winston/mvsplat/outputs/visualize_col"
    output_video = "/project/winston/mvsplat/outputs/visualize_output.mp4"
    fps = 10
    
    # 允許通過命令行參數自訂
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_video = sys.argv[2]
    if len(sys.argv) > 3:
        fps = int(sys.argv[3])
    
    print("=" * 60)
    print("照片轉影片工具")
    print("=" * 60)
    print(f"資料夾：{image_folder}")
    print(f"輸出影片：{output_video}")
    print(f"幀率：{fps} fps")
    print("=" * 60)
    
    make_video_from_images(image_folder, output_video, fps=fps, video_format='mp4')
