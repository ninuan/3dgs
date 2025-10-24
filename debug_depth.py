import cv2
import numpy as np
import json

# 读取深度图
depth_path = "data/depth/000009.png"
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
print(f"深度图形状: {depth.shape}")
print(f"深度图数据类型: {depth.dtype}")
print(f"深度图范围: min={depth.min()}, max={depth.max()}")

# 归一化（和camera_utils.py中一样）
depth_float = depth.astype(np.float32)
maxv = 65535.0 if depth_float.max() > 255 else 255.0
depth_normalized = depth_float / maxv
print(f"归一化后范围: min={depth_normalized.min():.6f}, max={depth_normalized.max():.6f}")

# 读取depth_params
with open("data/sparse/0/depth_params.json") as f:
    depth_params = json.load(f)
    params = depth_params["000009"]
    print(f"\ndepth_params: {params}")

# 应用scale和offset（和cameras.py中一样）
invdepth_scaled = depth_normalized * params["scale"] + params["offset"]
print(f"应用scale/offset后范围: min={invdepth_scaled.min():.6f}, max={invdepth_scaled.max():.6f}")

# 计算真实深度（逆深度的倒数）
depth_real = 1.0 / (invdepth_scaled + 1e-6)
print(f"真实深度范围: min={depth_real.min():.6f}, max={depth_real.max():.6f}")

# 读取点云，看看点云的深度范围
from plyfile import PlyData
ply = PlyData.read("data/sparse/0/points3D.ply")
points = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=1)
print(f"\n点云范围:")
print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
print(f"点云数量: {len(points)}")

# 计算点云的深度范围（从原点）
depth_from_origin = np.linalg.norm(points, axis=1)
print(f"点云距离原点: min={depth_from_origin.min():.3f}, max={depth_from_origin.max():.3f}")
