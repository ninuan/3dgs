import cv2
import numpy as np
from plyfile import PlyData
import json

print("=" * 70)
print("Data2 数据分析")
print("=" * 70)

# 1. 检查点云
try:
    ply = PlyData.read("data2/sparse/0/points3D.ply")
    xyz = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=1)

    print(f"\n初始点云:")
    print(f"  点数: {len(xyz)}")
    print(f"  X范围: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
    print(f"  Y范围: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
    print(f"  Z范围: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")

    center = xyz.mean(axis=0)
    radius = np.linalg.norm(xyz - center, axis=1).max()
    print(f"  中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  半径: {radius:.3f}")
except Exception as e:
    print(f"点云读取失败: {e}")

# 2. 检查深度图
import os
depth_files = sorted([f for f in os.listdir("data2/depth") if f.endswith('.png')])
print(f"\n深度图:")
print(f"  文件数: {len(depth_files)}")

if depth_files:
    depth_sample = cv2.imread(f"data2/depth/{depth_files[0]}", cv2.IMREAD_UNCHANGED)
    print(f"  示例文件: {depth_files[0]}")
    print(f"  形状: {depth_sample.shape}")
    print(f"  数据类型: {depth_sample.dtype}")
    print(f"  值范围: [{depth_sample.min()}, {depth_sample.max()}]")

    # 分析深度值分布
    valid_depth = depth_sample[depth_sample > 0]
    if len(valid_depth) > 0:
        depth_m = valid_depth.astype(np.float32) / 1000.0
        invdepth = 1.0 / depth_m
        print(f"  真实深度范围 (米): [{depth_m.min():.3f}, {depth_m.max():.3f}]")
        print(f"  逆深度范围: [{invdepth.min():.3f}, {invdepth.max():.3f}]")
        print(f"  有效像素比例: {len(valid_depth) / depth_sample.size * 100:.1f}%")

# 3. 检查mask
mask_files = sorted([f for f in os.listdir("data2/mask") if f.endswith('.png')])
print(f"\nMask:")
print(f"  文件数: {len(mask_files)}")

if mask_files:
    mask_sample = cv2.imread(f"data2/mask/{mask_files[0]}", cv2.IMREAD_UNCHANGED)
    print(f"  示例文件: {mask_files[0]}")
    print(f"  形状: {mask_sample.shape}")
    print(f"  数据类型: {mask_sample.dtype}")
    print(f"  值范围: [{mask_sample.min()}, {mask_sample.max()}]")
    print(f"  有效像素比例: {(mask_sample > 0).sum() / mask_sample.size * 100:.1f}%")

# 4. 检查depth_params
try:
    with open("data2/sparse/0/depth_params.json") as f:
        depth_params = json.load(f)

    print(f"\nDepth Params:")
    print(f"  视角数: {len(depth_params)}")

    # 统计scale分布
    scales = [p['scale'] for p in depth_params.values()]
    print(f"  Scale范围: [{min(scales):.3f}, {max(scales):.3f}]")
    print(f"  Scale均值: {np.mean(scales):.3f}")

    # 检查是否有异常的scale
    med_scales = [p['med_scale'] for p in depth_params.values()]
    print(f"  Med_scale范围: [{min(med_scales):.3f}, {max(med_scales):.3f}]")

    unusual_scales = [k for k, v in depth_params.items()
                      if v['scale'] < 0.2 * v['med_scale'] or v['scale'] > 5 * v['med_scale']]
    if unusual_scales:
        print(f"  ⚠️  异常scale的视角: {unusual_scales}")
except Exception as e:
    print(f"depth_params读取失败: {e}")

# 5. 对比data和data2
print(f"\n" + "=" * 70)
print("与Data对比:")
print("=" * 70)

try:
    data1_ply = PlyData.read("data/sparse/0/points3D.ply")
    data1_xyz = np.stack([data1_ply['vertex']['x'], data1_ply['vertex']['y'], data1_ply['vertex']['z']], axis=1)
    data1_center = data1_xyz.mean(axis=0)
    data1_radius = np.linalg.norm(data1_xyz - data1_center, axis=1).max()

    print(f"Data1: {len(data1_xyz)}点, 半径={data1_radius:.3f}")
    print(f"Data2: {len(xyz)}点, 半径={radius:.3f}")
    print(f"Data2点数比例: {len(xyz)/len(data1_xyz):.2f}x")
    print(f"Data2尺度比例: {radius/data1_radius:.2f}x")
except:
    pass
