import cv2
import numpy as np

# 读取深度图
depth = cv2.imread("data/depth/000009.png", cv2.IMREAD_UNCHANGED)

# 如果深度图的值是逆深度（单位：1/米），那么：
# - 逆深度 = 1/真实深度
# - 对于2-3米的物体，逆深度应该在 0.33-0.5 左右

print("假设深度图已经是逆深度（单位：1/米）:")
print(f"  深度图值范围: {depth.min()} - {depth.max()}")
print(f"  如果深度图存储时放大了1000倍:")
invdepth = depth.astype(np.float32) / 1000.0
print(f"    逆深度范围: {invdepth.min():.3f} - {invdepth.max():.3f}")
depth_real = 1.0 / (invdepth + 1e-6)
print(f"    对应真实深度: {depth_real.max():.3f} - {depth_real.min():.3f} 米")

print(f"\n  如果深度图存储时放大了10000倍:")
invdepth2 = depth.astype(np.float32) / 10000.0
print(f"    逆深度范围: {invdepth2.min():.3f} - {invdepth2.max():.3f}")
depth_real2 = 1.0 / (invdepth2 + 1e-6)
print(f"    对应真实深度: {depth_real2.max():.3f} - {depth_real2.min():.3f} 米")

print(f"\n假设深度图是真实深度（单位：毫米）:")
depth_mm = depth.astype(np.float32)
depth_m = depth_mm / 1000.0
print(f"  真实深度范围: {depth_m.min():.3f} - {depth_m.max():.3f} 米")
print(f"  对应逆深度: {(1/depth_m[depth_m>0]).min():.3f} - {(1/depth_m[depth_m>0]).max():.3f}")
