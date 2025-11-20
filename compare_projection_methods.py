#!/usr/bin/env python3
"""
对比两种投影方法，找出compute_mask_constraint的bug
"""
import torch
import numpy as np
from plyfile import PlyData
import cv2

def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

print("="*80)
print("COMPARING TWO PROJECTION METHODS")
print("="*80)

# 加载点云
ply_path = "output/data3/point_cloud/iteration_30000/point_cloud.ply"
plydata = PlyData.read(ply_path)
xyz = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
xyz_torch = torch.from_numpy(xyz).float().cuda()

# 相机内参
fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5
width, height = 640, 480

# 读取第一个相机pose
images_txt = "data3/sparse/0/images.txt"
with open(images_txt, 'r') as f:
    lines = f.readlines()
line_idx = 0
while line_idx < len(lines) and lines[line_idx].startswith('#'):
    line_idx += 1
first_line = lines[line_idx].strip().split()
qw, qx, qy, qz = map(float, first_line[1:5])
tx, ty, tz = map(float, first_line[5:8])

# COLMAP格式：world-to-camera
R_w2c = quat_to_rotation_matrix(qw, qx, qy, qz)
t_w2c = np.array([tx, ty, tz])

# 加载mask
mask = cv2.imread("data3/mask/000000.png", cv2.IMREAD_GRAYSCALE)

print(f"\nCamera pose (COLMAP w2c):")
print(f"  R_w2c[0]: {R_w2c[0]}")
print(f"  t_w2c: {t_w2c}")

# 方法1：简单投影（正确）
print(f"\n[Method 1] Simple projection (COLMAP convention):")
points_in_mask_m1 = 0
points_out_mask_m1 = 0

for i in range(min(10000, len(xyz))):  # 只测试前10000个点
    point = xyz[i]
    # World to camera
    point_cam = R_w2c @ point + t_w2c
    if point_cam[2] <= 0:
        continue
    # Project
    u = fx * point_cam[0] / point_cam[2] + cx
    v = fy * point_cam[1] / point_cam[2] + cy
    if 0 <= u < width and 0 <= v < height:
        if mask[int(v), int(u)] > 0:
            points_in_mask_m1 += 1
        else:
            points_out_mask_m1 += 1

print(f"  In mask: {points_in_mask_m1}")
print(f"  Out mask: {points_out_mask_m1}")
print(f"  Percentage out: {100.0 * points_out_mask_m1 / (points_in_mask_m1 + points_out_mask_m1):.1f}%")

# 方法2：使用full_proj_transform（compute_mask_constraint的方法）
print(f"\n[Method 2] Using full_proj_transform (like compute_mask_constraint):")

# 需要加载camera对象来获取full_proj_transform
from scene import Scene, GaussianModel

class Args:
    def __init__(self):
        self.source_path = "data3"
        self.model_path = "output/data3"
        self.images = "images"
        self.depths = "depth"
        self.depth_mask_dir = "mask"
        self.resolution = -1
        self.data_device = "cuda"
        self.eval = False
        self.white_background = False
        self.sh_degree = 3
        self.train_test_exp = False

args = Args()
gaussians = GaussianModel(args.sh_degree)
scene = Scene(args, gaussians, load_iteration=30000, shuffle=False)

# 获取第一个相机
first_cam = scene.getTrainCameras()[0]
print(f"  Camera name: {first_cam.image_name}")
print(f"  Camera resolution: {first_cam.image_width} x {first_cam.image_height}")
print(f"  full_proj_transform shape: {first_cam.full_proj_transform.shape}")

# 投影前10000个点
xyz_h = torch.cat([xyz_torch[:10000], torch.ones(10000, 1, device="cuda")], dim=1)
xyz_clip = xyz_h @ first_cam.full_proj_transform.T
w_threshold = 0.0001
valid_w = xyz_clip[:, 3] > w_threshold
xyz_ndc = xyz_clip[:, :3] / xyz_clip[:, 3:4].clamp(min=w_threshold)
x = ((xyz_ndc[:, 0] + 1) * 0.5 * first_cam.image_width).long()
y = ((xyz_ndc[:, 1] + 1) * 0.5 * first_cam.image_height).long()
valid = valid_w & (x >= 0) & (x < first_cam.image_width) & (y >= 0) & (y < first_cam.image_height)

# 检查mask
mask_torch = torch.from_numpy(mask).float().cuda()
points_in_mask_m2 = 0
points_out_mask_m2 = 0

for i in range(10000):
    if not valid[i]:
        continue
    if mask_torch[y[i], x[i]] > 0:
        points_in_mask_m2 += 1
    else:
        points_out_mask_m2 += 1

print(f"  In mask: {points_in_mask_m2}")
print(f"  Out mask: {points_out_mask_m2}")
print(f"  Percentage out: {100.0 * points_out_mask_m2 / (points_in_mask_m2 + points_out_mask_m2) if (points_in_mask_m2 + points_out_mask_m2) > 0 else 0:.1f}%")

print(f"\n" + "="*80)
print(f"CONCLUSION:")
if abs(points_out_mask_m1 - points_out_mask_m2) > 100:
    print(f"  ❌ TWO METHODS GIVE DIFFERENT RESULTS!")
    print(f"     Method 1 (correct): {points_out_mask_m1} points out of mask")
    print(f"     Method 2 (buggy): {points_out_mask_m2} points out of mask")
    print(f"     → full_proj_transform is WRONG!")
else:
    print(f"  ✓ Both methods agree")
print("="*80)
