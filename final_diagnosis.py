#!/usr/bin/env python3
"""
深度尺度代码已经修复，那么data2效果差的真正原因是什么？
需要检查其他可能的问题：
1. 初始点云质量
2. 深度mask覆盖率
3. 训练参数
4. 场景特性
"""

import numpy as np
import cv2
import os
from plyfile import PlyData

print("="*80)
print("深入分析data2效果差的原因")
print("="*80)

def check_depth_mask_coverage(dataset):
    """检查深度mask的覆盖率"""
    depth_dir = f"{dataset}/depth"
    mask_dir = f"{dataset}/mask"

    if not os.path.exists(mask_dir):
        print(f"\n{dataset}: 没有mask目录")
        return

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])

    print(f"\n{dataset} - 深度mask覆盖率:")
    print("-"*60)

    total_coverage = []
    for depth_file in depth_files[:5]:
        base = os.path.splitext(depth_file)[0]
        mask_path = os.path.join(mask_dir, f"{base}.png")
        depth_path = os.path.join(depth_dir, depth_file)

        if not os.path.exists(mask_path):
            print(f"{depth_file}: ⚠️ 没有对应的mask")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if mask is None or depth is None:
            continue

        mask_ratio = (mask > 0).sum() / mask.size
        depth_valid_ratio = (depth > 0).sum() / depth.size

        total_coverage.append(mask_ratio)

        print(f"{depth_file}:")
        print(f"  Mask覆盖率: {100*mask_ratio:.1f}%")
        print(f"  深度有效率: {100*depth_valid_ratio:.1f}%")
        print(f"  有效监督区域: {100*mask_ratio*depth_valid_ratio:.1f}%")

    if total_coverage:
        print(f"\n平均mask覆盖率: {100*np.mean(total_coverage):.1f}%")
        if np.mean(total_coverage) < 0.15:
            print("⚠️ 警告: mask覆盖率太低！这会导致深度监督不足")

def check_initial_pointcloud_quality(dataset):
    """检查初始点云的质量"""
    ply_path = f"{dataset}/sparse/0/points3D.ply"

    if not os.path.exists(ply_path):
        print(f"\n{dataset}: 没有初始点云")
        return

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    print(f"\n{dataset} - 初始点云质量:")
    print("-"*60)
    print(f"点数: {len(points):,}")
    print(f"点云范围: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
          f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
          f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

    # 检查点云分布
    extent = points.max(axis=0) - points.min(axis=0)
    print(f"点云尺寸: {extent}")

    # 检查点云密度（平均最近邻距离）
    from scipy.spatial import cKDTree
    sample_points = points[::max(1, len(points)//1000)]  # 采样1000个点
    tree = cKDTree(sample_points)
    dists, _ = tree.query(sample_points, k=2)
    avg_nn_dist = dists[:, 1].mean()
    print(f"平均最近邻距离: {avg_nn_dist:.4f}")

    # 检查是否有异常值
    center = points.mean(axis=0)
    dists_from_center = np.linalg.norm(points - center, axis=1)
    outlier_ratio = (dists_from_center > dists_from_center.mean() + 3*dists_from_center.std()).sum() / len(points)
    print(f"离群点比例: {100*outlier_ratio:.2f}%")

    if len(points) > 200000:
        print("✓ 点云足够密集")
    elif len(points) < 10000:
        print("⚠️ 点云太稀疏，可能影响初始化")

def check_camera_geometry(dataset):
    """检查相机几何关系"""
    from scene.colmap_loader import read_extrinsics_text, qvec2rotmat

    images_txt = f"{dataset}/sparse/0/images.txt"
    cam_extrinsics = read_extrinsics_text(images_txt)

    centers = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        R = qvec2rotmat(extr.qvec)
        t = np.array(extr.tvec)
        center = -R.T @ t
        centers.append(center)

    centers = np.array(centers)

    print(f"\n{dataset} - 相机几何:")
    print("-"*60)
    print(f"相机数量: {len(centers)}")

    # 计算相机间距离的分布
    dists = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dists.append(np.linalg.norm(centers[i] - centers[j]))

    if dists:
        dists = np.array(dists)
        print(f"相机间距离: 最小={dists.min():.3f}, 最大={dists.max():.3f}, 平均={dists.mean():.3f}")

        if dists.min() < 0.1:
            print("⚠️ 有些相机位置太接近")
        if dists.max() > 20:
            print("⚠️ 相机分布太稀疏")

# 分析两个数据集
for dataset in ['data', 'data2']:
    if os.path.exists(dataset):
        check_depth_mask_coverage(dataset)
        check_initial_pointcloud_quality(dataset)
        check_camera_geometry(dataset)

print("\n" + "="*80)
print("总结")
print("="*80)
print("""
可能导致data2效果差的原因：
1. Mask覆盖率低 → 深度监督不足
2. 初始点云质量差 → Gaussian初始化不好
3. 相机分布不合理 → 难以优化
4. 场景复杂度高 → 需要更多迭代/更好的参数
""")
