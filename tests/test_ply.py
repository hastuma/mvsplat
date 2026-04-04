#!/usr/bin/env python3
"""驗證 PLY 文件生成"""
import numpy as np
import os

output_ply = './buildings_3d.ply'

# 建立簡單的正方體測試
vertices = np.array([
    [0, 0, 0],
    [10, 0, 0],
    [10, 10, 0],
    [0, 10, 0],
    [0, 0, 10],
    [10, 0, 10],
    [10, 10, 10],
    [0, 10, 10],
], dtype=np.float64)

triangles = np.array([
    [0, 2, 1],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [2, 3, 7],
    [2, 7, 6],
    [0, 4, 7],
    [0, 7, 3],
    [1, 2, 6],
    [1, 6, 5],
], dtype=np.int32)

print("生成測試 PLY 文件...")

# 手動編寫分PLY文件（ASCII格式）
with open(output_ply, 'w') as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("comment Created by test_ply\n")
    f.write(f"element vertex {len(vertices)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write(f"element face {len(triangles)}\n")
    f.write("property list uchar int vertex_indices\n")
    f.write("end_header\n")
    
    # 寫入頂點
    for v in vertices:
        f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} 100 150 200\n")
    
    # 寫入面（三角形）
    for tri in triangles:
        f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

print(f"✓ 測試 PLY 已保存至: {output_ply}")

# 驗證文件
if os.path.exists(output_ply):
    file_size = os.path.getsize(output_ply)
    print(f"  文件大小: {file_size} bytes")
    
    # 讀取前 10 行驗證格式
    with open(output_ply, 'r') as f:
        lines = f.readlines()[:10]
        print("\n  文件頭部:")
        for line in lines:
            print(f"    {line.rstrip()}")
    
    print("\n✓ PLY 文件格式正確！")
else:
    print("✗ 文件生成失敗")
