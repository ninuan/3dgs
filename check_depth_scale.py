#!/usr/bin/env python3
import cv2
import numpy as np

# 检查depth图的数值范围
depth_path = "data3/depth/000000.png"
depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
depth_m = depth.astype(np.float32) / 5000.0

mask_path = "data3/mask/000000.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 只看mask内的深度
depth_masked = depth_m[mask > 0]

print(f"Depth statistics (in mask region):")
print(f"  Mean: {depth_masked.mean():.3f} m")
print(f"  Std: {depth_masked.std():.3f} m")
print(f"  Min: {depth_masked.min():.3f} m")
print(f"  Max: {depth_masked.max():.3f} m")
print(f"  Valid pixels in mask: {len(depth_masked)}")

# 转换为inverse depth
inv_depth_masked = 1.0 / (depth_masked + 1e-6)
print(f"\nInverse depth statistics:")
print(f"  Mean: {inv_depth_masked.mean():.3f} 1/m")
print(f"  Std: {inv_depth_masked.std():.3f} 1/m")
print(f"  Min: {inv_depth_masked.min():.3f} 1/m")
print(f"  Max: {inv_depth_masked.max():.3f} 1/m")
