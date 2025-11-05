#!/usr/bin/env python3
"""
检查data和data2的COLMAP外参惯例是否一致

COLMAP标准: tvec = -R @ C_world，其中C_world是相机中心在世界坐标系中的位置
"""

import numpy as np
from scene.colmap_loader import read_extrinsics_text, qvec2rotmat

def check_colmap_convention(dataset_name):
    print(f"\n{'='*80}")
    print(f"检查 {dataset_name} 的COLMAP外参惯例")
    print(f"{'='*80}")

    images_txt = f"{dataset_name}/sparse/0/images.txt"
    cam_extrinsics = read_extrinsics_text(images_txt)

    # 检查前3个相机
    count = 0
    for key, extr in cam_extrinsics.items():
        if count >= 3:
            break
        count += 1

        R = qvec2rotmat(extr.qvec)
        t = extr.tvec

        print(f"\n{extr.name}:")
        print(f"  qvec: {extr.qvec}")
        print(f"  tvec: {t}")

        # 假设1: 标准COLMAP (tvec = -R @ C_world)
        C_world_standard = -R.T @ t
        print(f"  [标准COLMAP] C_world = -R^T @ tvec = {C_world_standard}")

        # 假设2: tvec直接是相机中心 (某些工具的非标准惯例)
        print(f"  [非标准] C_world = tvec = {t}")

        # 假设3: tvec = R @ C_world (OpenCV惯例)
        C_world_opencv = R.T @ t
        print(f"  [OpenCV] C_world = R^T @ tvec = {C_world_opencv}")

check_colmap_convention("data")
check_colmap_convention("data2")

print(f"\n{'='*80}")
print("对比同一张图片 (000299.png)")
print(f"{'='*80}")

cam_data = read_extrinsics_text("data/sparse/0/images.txt")
cam_data2 = read_extrinsics_text("data2/sparse/0/images.txt")

for key, extr in cam_data.items():
    if extr.name == "000299.png":
        R_data = qvec2rotmat(extr.qvec)
        t_data = extr.tvec
        C_data_standard = -R_data.T @ t_data

        print(f"\ndata/000299.png:")
        print(f"  tvec: {t_data}")
        print(f"  C_world (标准): {C_data_standard}")

for key, extr in cam_data2.items():
    if extr.name == "000299.png":
        R_data2 = qvec2rotmat(extr.qvec)
        t_data2 = extr.tvec
        C_data2_standard = -R_data2.T @ t_data2

        print(f"\ndata2/000299.png:")
        print(f"  tvec: {t_data2}")
        print(f"  C_world (标准): {C_data2_standard}")

        print(f"\n对比:")
        print(f"  data的tvec   = {t_data}")
        print(f"  data2的C_world = {C_data2_standard}")
        print(f"  是否相等? {np.allclose(t_data, C_data2_standard)}")

        print(f"\n  data的C_world  = {C_data_standard}")
        print(f"  data2的tvec   = {t_data2}")
        print(f"  是否接近? {np.allclose(C_data_standard, t_data2, atol=0.5)}")
