#!/usr/bin/env python3
"""
分析训练后的点云，检查有多少点应该在mask内 vs mask外
"""
import numpy as np
from plyfile import PlyData
import cv2
import torch
import os

# 加载最终点云
ply_path = "output/data3/point_cloud/iteration_30000/point_cloud.ply"
plydata = PlyData.read(ply_path)
vertices = plydata['vertex']
xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

print(f"="*80)
print(f"FINAL POINT CLOUD ANALYSIS")
print(f"="*80)
print(f"\nTotal points: {len(xyz)}")

# 加载相机参数和mask
from scene import Scene
from argparse import Namespace

# 模拟scene参数
class FakeArgs:
    def __init__(self):
        self.source_path = "data3"
        self.model_path = "output/data3"
        self.images = "images"
        self.depths = "depth"
        self.depth_mask_dir = "mask"
        self.resolution = -1
        self.data_device = "cuda"
        self.eval = False
        self.train_test_exp = False
        self.white_background = False

args = FakeArgs()

# 读取相机poses
from scene.dataset_readers import readColmapSceneInfo, storePly

sparse_dir = os.path.join(args.source_path, "sparse/0")
cameras_txt = os.path.join(sparse_dir, "cameras.txt")
images_txt = os.path.join(sparse_dir, "images.txt")

# 简单读取第一个相机和mask
mask_dir = os.path.join(args.source_path, "mask")
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

if len(mask_files) > 0:
    # 读取第一个mask
    first_mask = cv2.imread(os.path.join(mask_dir, mask_files[0]), cv2.IMREAD_GRAYSCALE)
    mask_coverage = (first_mask > 0).sum() / first_mask.size
    print(f"\nMask statistics (first frame):")
    print(f"  Coverage: {100.0 * mask_coverage:.1f}%")
    print(f"  Mask pixels: {(first_mask > 0).sum()}")

# 检查点云的空间分布
print(f"\nPoint cloud spatial distribution:")
print(f"  X range: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
print(f"  Y range: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
print(f"  Z range: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")

# 计算点云的密度分布
# 将点云分成grid，看分布
from scipy.spatial import cKDTree

# 计算每个点到最近邻的距离（密度指标）
tree = cKDTree(xyz)
distances, indices = tree.query(xyz, k=11)  # k=11: 自己+10个最近邻
avg_distances = distances[:, 1:].mean(axis=1)  # 排除自己

print(f"\nPoint cloud density:")
print(f"  Mean distance to 10-NN: {avg_distances.mean():.4f}")
print(f"  Std distance to 10-NN: {avg_distances.std():.4f}")
print(f"  Max distance to 10-NN: {avg_distances.max():.4f}")

# 检测outliers（散点）
# 如果一个点的avg_distance远大于中位数，说明是散点
median_dist = np.median(avg_distances)
outlier_threshold = median_dist * 3
outliers = avg_distances > outlier_threshold

print(f"\nOutlier detection:")
print(f"  Median distance: {median_dist:.4f}")
print(f"  Outlier threshold (3x median): {outlier_threshold:.4f}")
print(f"  Number of outliers: {outliers.sum()} ({100.0 * outliers.sum() / len(xyz):.1f}%)")

print(f"\n" + "="*80)
