import cv2
import numpy as np
import json

# 使用修复后的逻辑
depth_path = "data/depth/000009.png"
raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# 新逻辑：深度图是真实深度（毫米）
depth_mm = raw.astype(np.float32)
depth_m = depth_mm / 1000.0  # 毫米转米

invdepthmap = np.zeros_like(depth_m)
valid_mask = depth_m > 0
invdepthmap[valid_mask] = 1.0 / depth_m[valid_mask]

print(f"原始深度图范围: {raw.min()} - {raw.max()}")
print(f"真实深度范围 (米): {depth_m[valid_mask].min():.3f} - {depth_m[valid_mask].max():.3f}")
print(f"逆深度范围: {invdepthmap[valid_mask].min():.3f} - {invdepthmap[valid_mask].max():.3f}")

# 应用depth_params
with open("data/sparse/0/depth_params.json") as f:
    depth_params = json.load(f)
    params = depth_params["000009"]

invdepth_scaled = invdepthmap * params["scale"] + params["offset"]
print(f"\n应用depth_params后:")
print(f"  scale={params['scale']}, offset={params['offset']}")
print(f"  逆深度范围: {invdepth_scaled[valid_mask].min():.3f} - {invdepth_scaled[valid_mask].max():.3f}")

# 从点云角度验证
from plyfile import PlyData
ply = PlyData.read("data/sparse/0/points3D.ply")
points = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=1)
depth_from_origin = np.linalg.norm(points, axis=1)
print(f"\n点云深度范围: {depth_from_origin.min():.3f} - {depth_from_origin.max():.3f} 米")
print(f"点云对应逆深度: {1/depth_from_origin.max():.3f} - {1/depth_from_origin.min():.3f}")
print(f"\n✓ 逆深度范围和点云基本吻合!" if abs(invdepth_scaled[valid_mask].mean() - (1/depth_from_origin.mean())) < 0.1 else "✗ 逆深度范围不匹配")
