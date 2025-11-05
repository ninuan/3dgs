#!/usr/bin/env python3
"""
检查初始点云和深度图的尺度是否匹配
"""

import numpy as np
import cv2
import os
from plyfile import PlyData
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat

def check_pointcloud_depth_scale_match(dataset):
    print(f"\n{'='*80}")
    print(f"检查 {dataset} 的点云-深度尺度匹配")
    print(f"{'='*80}\n")

    # 1. 读取初始点云
    ply_path = f"{dataset}/sparse/0/points3D.ply"
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    print(f"初始点云统计:")
    print(f"  点数: {len(points)}")
    print(f"  中心: {np.mean(points, axis=0)}")
    print(f"  范围: {np.ptp(points, axis=0)}")

    # 2. 加载相机
    images_txt = f"{dataset}/sparse/0/images.txt"
    cameras_txt = f"{dataset}/sparse/0/cameras.txt"
    cam_extrinsics = read_extrinsics_text(images_txt)
    cam_intrinsics = read_intrinsics_text(cameras_txt)

    cam_intr = list(cam_intrinsics.values())[0]
    if cam_intr.model == "PINHOLE":
        fx, fy = cam_intr.params[0], cam_intr.params[1]
        cx, cy = cam_intr.params[2], cam_intr.params[3]
    else:
        fx = fy = cam_intr.params[0]
        cx = cam_intr.params[1] if len(cam_intr.params) > 1 else cam_intr.width / 2
        cy = cam_intr.params[2] if len(cam_intr.params) > 2 else cam_intr.height / 2

    width, height = cam_intr.width, cam_intr.height

    # 3. 对于每个视角，比较点云投影深度 vs 深度图
    depth_dir = f"{dataset}/depth"
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])[:3]

    print(f"\n逐视角检查点云投影深度 vs 深度图:")
    print("-" * 80)

    for depth_file in depth_files:
        # 找到对应的相机
        extr = None
        for key, e in cam_extrinsics.items():
            if e.name == depth_file:
                extr = e
                break

        if extr is None:
            continue

        # 加载深度图
        depth_path = os.path.join(depth_dir, depth_file)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue

        depth_map = depth_raw.astype(np.float32) / 1000.0  # mm to m

        # 投影点云到相机
        R = qvec2rotmat(extr.qvec)
        t = extr.tvec
        points_cam = (R @ points.T).T + t

        # 过滤在相机前方的点
        valid_depth_mask = points_cam[:, 2] > 0.1
        points_cam_valid = points_cam[valid_depth_mask]

        if len(points_cam_valid) == 0:
            continue

        # 投影到像素
        u = points_cam_valid[:, 0] / points_cam_valid[:, 2] * fx + cx
        v = points_cam_valid[:, 1] / points_cam_valid[:, 2] * fy + cy

        # 过滤在图像内的点
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u_valid = u[in_image].astype(int)
        v_valid = v[in_image].astype(int)
        depths_proj = points_cam_valid[in_image, 2]

        if len(u_valid) == 0:
            print(f"\n  {depth_file}: 没有点投影到图像内")
            continue

        # 采样深度图的深度
        depths_map_sampled = depth_map[v_valid, u_valid]

        # 过滤深度图有效区域
        valid_both = (depths_map_sampled > 0.1) & (depths_map_sampled < 10.0)
        depths_proj_valid = depths_proj[valid_both]
        depths_map_valid = depths_map_sampled[valid_both]

        if len(depths_proj_valid) < 10:
            print(f"\n  {depth_file}: 重叠区域太少 ({len(depths_proj_valid)}个点)")
            continue

        # 统计
        median_proj = np.median(depths_proj_valid)
        median_map = np.median(depths_map_valid)
        scale_ratio = median_map / median_proj

        print(f"\n  {depth_file}:")
        print(f"    点云投影深度中位数: {median_proj:.3f}m")
        print(f"    深度图深度中位数:   {median_map:.3f}m")
        print(f"    比例 (depth_map / pointcloud): {scale_ratio:.3f}")

        if abs(scale_ratio - 1.0) > 0.2:
            print(f"    ⚠️ 尺度差异较大 ({abs(scale_ratio - 1.0)*100:.1f}%)")

check_pointcloud_depth_scale_match('data')
check_pointcloud_depth_scale_match('data2')

print(f"\n{'='*80}")
print("总结")
print(f"{'='*80}")
print("""
如果点云投影深度和深度图的深度比例接近1.0，说明尺度匹配。
如果比例偏离1.0较多（>20%），说明存在尺度不匹配，会导致：
  - 不同视角优化出不同尺度的点云
  - 最终ply中出现多个偏移的点云副本
""")
