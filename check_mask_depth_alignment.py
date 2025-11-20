#!/usr/bin/env python3
import cv2
import numpy as np

# 检查cabinet mask内是否有无效深度
depth_path = "data3/depth/000000.png"
depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
depth_m = depth.astype(np.float32) / 5000.0

cabinet_mask_path = "data3/mask/000000.png"
cabinet_mask = cv2.imread(cabinet_mask_path, cv2.IMREAD_GRAYSCALE)

# Cabinet mask内的像素
cabinet_pixels = cabinet_mask > 0

# 有效深度mask（0.1m < depth < 6.0m）
valid_depth_mask = (depth_m > 0.1) & (depth_m < 6.0) & (depth_m > 0)

# Cabinet mask内有多少像素的深度无效？
invalid_in_cabinet = cabinet_pixels & (~valid_depth_mask)

print(f"Cabinet mask pixels: {cabinet_pixels.sum()}")
print(f"Valid depth pixels in cabinet mask: {(cabinet_pixels & valid_depth_mask).sum()}")
print(f"INVALID depth pixels in cabinet mask: {invalid_in_cabinet.sum()}")
print(f"Percentage invalid: {100.0 * invalid_in_cabinet.sum() / cabinet_pixels.sum():.2f}%")

if invalid_in_cabinet.sum() > 0:
    invalid_depth_values = depth_m[invalid_in_cabinet]
    print(f"\nInvalid depth values:")
    print(f"  Min: {invalid_depth_values.min():.6f} m")
    print(f"  Max: {invalid_depth_values.max():.6f} m")
    print(f"  Mean: {invalid_depth_values.mean():.6f} m")
    print(f"  Count of zeros: {(invalid_depth_values == 0).sum()}")
