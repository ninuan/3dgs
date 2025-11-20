#!/usr/bin/env python3
"""
测试compute_mask_constraint是否正确工作
"""
import torch
import numpy as np
from plyfile import PlyData
import cv2
import os

# 模拟测试：用简单的投影检查点云-mask对齐

def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

# 加载最终点云
ply_path = "output/data3/point_cloud/iteration_30000/point_cloud.ply"
plydata = PlyData.read(ply_path)
vertices = plydata['vertex']
xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

print(f"="*80)
print(f"MASK CONSTRAINT VERIFICATION")
print(f"="*80)
print(f"\nTotal points: {len(xyz)}")

# 深度相机内参
fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5
width, height = 640, 480

# 读取第一个相机pose
images_txt = "data3/sparse/0/images.txt"

with open(images_txt, 'r') as f:
    lines = f.readlines()

# 跳过注释
line_idx = 0
while line_idx < len(lines) and lines[line_idx].startswith('#'):
    line_idx += 1

# 读取第一个相机
first_line = lines[line_idx].strip().split()
qw, qx, qy, qz = map(float, first_line[1:5])
tx, ty, tz = map(float, first_line[5:8])
image_name = first_line[9]

print(f"\nFirst camera: {image_name}")
print(f"  Translation: [{tx:.3f}, {ty:.3f}, {tz:.3f}]")

# 构建world-to-camera变换（COLMAP格式）
R_w2c = quat_to_rotation_matrix(qw, qx, qy, qz)
t_w2c = np.array([tx, ty, tz])

# 加载对应的mask
mask_path = f"data3/mask/{image_name.replace('.png', '.png')}"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
print(f"  Mask path: {mask_path}")
print(f"  Mask coverage: {100.0 * (mask > 0).sum() / mask.size:.1f}%")

# 投影所有点到这个视图
points_in_mask = 0
points_out_mask = 0
points_behind = 0
points_outside_image = 0

for i, point in enumerate(xyz):
    # World to camera
    point_cam = R_w2c @ point + t_w2c

    # Check if behind camera
    if point_cam[2] <= 0:
        points_behind += 1
        continue

    # Project
    u = fx * point_cam[0] / point_cam[2] + cx
    v = fy * point_cam[1] / point_cam[2] + cy

    # Check bounds
    if u < 0 or u >= width or v < 0 or v >= height:
        points_outside_image += 1
        continue

    # Check mask
    u_int, v_int = int(u), int(v)
    if mask[v_int, u_int] > 0:
        points_in_mask += 1
    else:
        points_out_mask += 1

print(f"\nProjection results for first camera:")
print(f"  Points in mask: {points_in_mask}")
print(f"  Points OUTSIDE mask: {points_out_mask}")
print(f"  Points behind camera: {points_behind}")
print(f"  Points outside image: {points_outside_image}")

percentage_outside = 100.0 * points_out_mask / (points_in_mask + points_out_mask) if (points_in_mask + points_out_mask) > 0 else 0
print(f"\n  Percentage outside mask: {percentage_outside:.1f}%")

if points_out_mask > 1000:
    print(f"\n⚠️  PROBLEM: {points_out_mask} points are outside the mask!")
    print(f"    This explains the scattered points in the visualization.")
    print(f"    The mask-based pruning is NOT working correctly!")

print(f"\n" + "="*80)
