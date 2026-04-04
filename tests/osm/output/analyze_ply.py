#!/usr/bin/env python3
"""
PLY 點雲分析工具 - 分析生成的 3D Volume 數據
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import statistics

def load_ply(ply_path):
    """讀取 PLY 檔案並提取數據"""
    points = []
    colors = []
    instance_ids = []
    semantic_ids = []
    
    with open(ply_path, 'r') as f:
        # 讀取 header
        line = f.readline().strip()
        assert line == 'ply', "Invalid PLY format"
        
        # 跳至 end_header
        while True:
            line = f.readline().strip()
            if line == 'end_header':
                break
        
        # 讀取數據
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
            inst_id = int(parts[6])
            sem_id = int(parts[7])
            
            points.append([x, y, z])
            colors.append([r, g, b])
            instance_ids.append(inst_id)
            semantic_ids.append(sem_id)
    
    return {
        'points': np.array(points),
        'colors': np.array(colors),
        'instance_ids': np.array(instance_ids),
        'semantic_ids': np.array(semantic_ids)
    }

def analyze_ply(ply_path):
    """分析 PLY 檔案並輸出統計信息"""
    print("=" * 70)
    print("3D Volume 分析報告")
    print("=" * 70)
    
    data = load_ply(ply_path)
    points = data['points']
    instance_ids = data['instance_ids']
    semantic_ids = data['semantic_ids']
    
    # 基本統計
    print("\n[基本信息]")
    print(f"  總點數: {len(points):,}")
    print(f"  文件大小: {ply_path.stat().st_size / (1024**2):.1f} MB")
    
    # 座標範圍
    print("\n[座標範圍]")
    print(f"  X (經度): [{points[:, 0].min():.6f}, {points[:, 0].max():.6f}]")
    print(f"  Y (緯度): [{points[:, 1].min():.6f}, {points[:, 1].max():.6f}]")
    print(f"  Z (高度): [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] 公尺")
    
    # Instance 統計
    print("\n[Instance 統計]")
    unique_instances = np.unique(instance_ids)
    print(f"  建築物數量 (unique instances): {len(unique_instances)}")
    
    # 計算每個 instance 的點數
    instance_sizes = defaultdict(int)
    for inst_id in instance_ids:
        instance_sizes[inst_id] += 1
    
    sizes = list(instance_sizes.values())
    print(f"  平均每棟大小: {statistics.mean(sizes):.0f} 點")
    print(f"  中位數大小: {statistics.median(sizes):.0f} 點")
    print(f"  最大: {max(sizes):,} 點 (Instance #{max(instance_sizes, key=instance_sizes.get)})")
    print(f"  最小: {min(sizes):,} 點")
    
    # Semantic 統計
    print("\n[Semantic 統計]")
    semantic_classes = {1: 'residential', 2: 'commercial', 3: 'industrial', 4: 'public', 5: 'other'}
    unique_semantic = np.unique(semantic_ids)
    for sem_id in sorted(unique_semantic):
        count = np.sum(semantic_ids == sem_id)
        class_name = semantic_classes.get(sem_id, f'unknown_{sem_id}')
        pct = 100 * count / len(semantic_ids)
        print(f"  {class_name:20s}: {count:9,} 點 ({pct:5.1f}%)")
    
    # 高度分布
    print("\n[高度分布]")
    heights = points[:, 2]
    print(f"  平均高度: {np.mean(heights):.1f} 公尺")
    print(f"  標準差: {np.std(heights):.1f} 公尺")
    print(f"  最低: {np.min(heights):.1f} 公尺")
    print(f"  最高: {np.max(heights):.1f} 公尺")
    
    # 建立高度 histogram
    height_bins = np.linspace(0, np.max(heights), 11)
    hist, _ = np.histogram(heights, bins=height_bins)
    print("\n  高度直方圖:")
    for i in range(len(hist)):
        bar_len = int(hist[i] / hist.max() * 50) if hist.max() > 0 else 0
        print(f"    {height_bins[i]:5.1f}m-{height_bins[i+1]:5.1f}m: {'█' * bar_len} {hist[i]:,}")
    
    # 地理統計
    print("\n[地理統計]")
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    
    # 粗略轉換為公尺
    scale_x = 111320 * np.cos(np.radians((points[:, 1].min() + points[:, 1].max()) / 2))
    scale_y = 111320
    
    area_km2 = (x_range * scale_x) * (y_range * scale_y) / (1e6)
    density = len(unique_instances) / area_km2 if area_km2 > 0 else 0
    print(f"  涵蓋面積: {area_km2:.3f} km²")
    print(f"  建築物密度: {density:.1f} 棟/km²")
    
    print("\n" + "=" * 70)

def instance_statistics(ply_path, output_json=None):
    """生成詳細的 instance 統計報告"""
    data = load_ply(ply_path)
    points = data['points']
    instance_ids = data['instance_ids']
    semantic_ids = data['semantic_ids']
    
    semantic_classes = {1: 'residential', 2: 'commercial', 3: 'industrial', 4: 'public', 5: 'other'}
    
    instance_stats = []
    unique_instances = np.unique(instance_ids)
    
    print("\n生成 Instance 詳細統計...")
    for inst_id in sorted(unique_instances):
        mask = instance_ids == inst_id
        inst_points = points[mask]
        inst_semantic = semantic_ids[mask][0]
        
        stats = {
            'instance_id': int(inst_id),
            'point_count': int(np.sum(mask)),
            'semantic_class': semantic_classes.get(int(inst_semantic), 'unknown'),
            'semantic_id': int(inst_semantic),
            'centroid': {
                'x': float(np.mean(inst_points[:, 0])),
                'y': float(np.mean(inst_points[:, 1])),
                'z': float(np.mean(inst_points[:, 2]))
            },
            'bounds': {
                'x': [float(inst_points[:, 0].min()), float(inst_points[:, 0].max())],
                'y': [float(inst_points[:, 1].min()), float(inst_points[:, 1].max())],
                'z': [float(inst_points[:, 2].min()), float(inst_points[:, 2].max())]
            },
            'footprint_area_approx': float((inst_points[:, 0].max() - inst_points[:, 0].min()) * 
                                           (inst_points[:, 1].max() - inst_points[:, 1].min()) * 
                                           111320**2)  # 粗略估計
        }
        instance_stats.append(stats)
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(instance_stats, f, indent=2)
        print(f"✓ 詳細統計已保存至: {output_json}")
    
    return instance_stats

if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    ply_file = Path('buildings_3d_volume.ply')
    
    if not ply_file.exists():
        print(f"✗ 找不到 {ply_file}")
        print("✓ 請確保在 output/ 目錄中執行此腳本")
        sys.exit(1)
    
    # 分析 PLY
    analyze_ply(ply_file)
    
    # 生成詳細統計
    stats_json = ply_file.parent / 'instance_statistics.json'
    instance_statistics(ply_file, output_json=stats_json)
    
    print("\n✓ 分析完成！")
