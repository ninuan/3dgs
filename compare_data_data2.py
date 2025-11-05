#!/usr/bin/env python3
"""
对比data和data2的深度图，找出为什么data2重建效果不好
"""

import numpy as np
import cv2
import os
from plyfile import PlyData

def analyze_depth_stats(dataset_name):
    """分析深度图的统计信息"""
    depth_dir = f"{dataset_name}/depth"

    if not os.path.exists(depth_dir):
        print(f"{dataset_name}: 深度目录不存在")
        return

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])

    print(f"\n{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*80}")
    print(f"深度图数量: {len(depth_files)}")

    all_depths = []
    depth_ranges = []

    for depth_file in depth_files[:5]:  # 检查前5张
        depth_path = os.path.join(depth_dir, depth_file)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            continue

        # 16-bit深度图
        depth_valid = depth[depth > 0].astype(float)

        if len(depth_valid) == 0:
            continue

        all_depths.extend(depth_valid)
        depth_ranges.append((depth_valid.min(), depth_valid.max(), np.median(depth_valid)))

        print(f"\n{depth_file}:")
        print(f"  分辨率: {depth.shape}")
        print(f"  原始值范围: [{depth_valid.min():.0f}, {depth_valid.max():.0f}]")
        print(f"  中位数: {np.median(depth_valid):.0f}")
        print(f"  有效像素比例: {100*len(depth_valid)/depth.size:.1f}%")

    if len(all_depths) > 0:
        all_depths = np.array(all_depths)
        print(f"\n整体统计:")
        print(f"  深度范围: [{all_depths.min():.0f}, {all_depths.max():.0f}]")
        print(f"  深度中位数: {np.median(all_depths):.0f}")
        print(f"  深度均值: {all_depths.mean():.0f}")
        print(f"  深度标准差: {all_depths.std():.0f}")

        # 分析是否是逆深度还是深度
        print(f"\n深度值解析:")
        print(f"  如果是逆深度（inverse depth）: 实际深度 = 1/深度值 (需要缩放)")
        print(f"  如果是深度（depth）: 实际深度 = 深度值 / 缩放因子")
        print(f"  当前深度值 ~{np.median(all_depths):.0f}，如果是米，则物体在1500米外（不合理）")
        print(f"  更可能是: 深度值 / 1000 = 实际深度(米) → ~{np.median(all_depths)/1000:.2f}米")

def check_colmap_scale(dataset_name):
    """检查COLMAP重建的尺度"""
    from scene.colmap_loader import read_extrinsics_text, qvec2rotmat

    images_txt = f"{dataset_name}/sparse/0/images.txt"
    cam_extrinsics = read_extrinsics_text(images_txt)

    centers = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        R = qvec2rotmat(extr.qvec)
        t = np.array(extr.tvec)
        center = -R.T @ t
        centers.append(center)

    centers = np.array(centers)

    # 计算相机间平均距离
    dists = []
    for i in range(len(centers)):
        for j in range(i+1, min(i+5, len(centers))):
            dists.append(np.linalg.norm(centers[i] - centers[j]))

    print(f"\n相机位姿统计:")
    print(f"  相机数量: {len(centers)}")
    print(f"  相机中心范围: [{centers.min(axis=0)}, {centers.max(axis=0)}]")
    print(f"  相机间平均距离: {np.mean(dists):.3f}")
    print(f"  场景尺度估计: {np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)):.3f}")

    # 读取初始点云
    ply_path = f"{dataset_name}/sparse/0/points3D.ply"
    if os.path.exists(ply_path):
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

        print(f"\n初始点云统计:")
        print(f"  点数: {len(points)}")
        print(f"  点云范围: [{points.min(axis=0)}, {points.max(axis=0)}]")
        print(f"  点云尺寸: {points.max(axis=0) - points.min(axis=0)}")
        print(f"  点云中心: {points.mean(axis=0)}")

# 分析两个数据集
for dataset in ['data', 'data2']:
    if os.path.exists(dataset):
        analyze_depth_stats(dataset)
        check_colmap_scale(dataset)

print("\n" + "="*80)
print("对比分析")
print("="*80)
print("""
关键问题要检查:
1. 深度值的单位和缩放因子是否正确
2. data和data2的深度尺度是否一致
3. 深度对齐算法是否对两个数据集都生效
4. 相机运动幅度和点云密度是否合适
""")
