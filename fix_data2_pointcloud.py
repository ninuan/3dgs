#!/usr/bin/env python3
"""
修复data2的初始点云，使其更好地覆盖mask区域

策略：
1. 从深度图和mask重建点云
2. 只在mask有效区域采样点
3. 替换原有的init.ply
"""

import numpy as np
import cv2
import os
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat

def reconstruct_pointcloud_from_masked_depth(dataset, output_path, max_points=100000):
    """
    从深度图和mask重建点云，确保点云覆盖mask区域
    """

    depth_dir = f"{dataset}/depth"
    mask_dir = f"{dataset}/mask"
    images_txt = f"{dataset}/sparse/0/images.txt"
    cameras_txt = f"{dataset}/sparse/0/cameras.txt"

    # 加载相机参数
    cam_extrinsics = read_extrinsics_text(images_txt)
    cam_intrinsics = read_intrinsics_text(cameras_txt)
    cam_intr = list(cam_intrinsics.values())[0]

    # 内参
    if cam_intr.model == "SIMPLE_PINHOLE":
        fx = fy = cam_intr.params[0]
        cx = cam_intr.params[1] if len(cam_intr.params) > 1 else cam_intr.width / 2
        cy = cam_intr.params[2] if len(cam_intr.params) > 2 else cam_intr.height / 2
    else:  # PINHOLE
        fx, fy = cam_intr.params[0], cam_intr.params[1]
        cx, cy = cam_intr.params[2], cam_intr.params[3]

    width, height = cam_intr.width, cam_intr.height

    print(f"相机内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"图像尺寸: {width}x{height}")

    # 收集所有视角的点
    all_points = []
    all_colors = []

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])

    print(f"\n处理 {len(depth_files)} 个视角...")

    for depth_file in depth_files:
        base = os.path.splitext(depth_file)[0]
        depth_path = os.path.join(depth_dir, depth_file)
        mask_path = os.path.join(mask_dir, f"{base}.png")

        # 找到对应的相机
        extr = None
        for key, e in cam_extrinsics.items():
            if e.name == depth_file:
                extr = e
                break

        if extr is None:
            print(f"  跳过 {depth_file}: 找不到相机外参")
            continue

        # 加载深度和mask
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if depth_raw is None or mask is None:
            print(f"  跳过 {depth_file}: 无法加载")
            continue

        # 转换深度：毫米 -> 米
        depth_m = depth_raw.astype(np.float32) / 1000.0

        # 只在mask有效区域采样
        valid_mask = (mask > 128) & (depth_m > 0.1) & (depth_m < 10.0)
        valid_indices = np.where(valid_mask)

        if len(valid_indices[0]) == 0:
            print(f"  跳过 {depth_file}: 没有有效点")
            continue

        # 采样点（避免太密集）
        num_valid = len(valid_indices[0])
        if num_valid > max_points // len(depth_files):
            sample_step = num_valid // (max_points // len(depth_files))
            valid_indices = (valid_indices[0][::sample_step], valid_indices[1][::sample_step])

        v_coords = valid_indices[0]
        u_coords = valid_indices[1]
        depths = depth_m[v_coords, u_coords]

        # 反投影到相机坐标系
        x_cam = (u_coords - cx) * depths / fx
        y_cam = (v_coords - cy) * depths / fy
        z_cam = depths

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # 转换到世界坐标系
        R_world2cam = qvec2rotmat(extr.qvec)
        t = np.array(extr.tvec)

        # X_world = R_world2cam^T @ (X_cam - t)
        R_cam2world = R_world2cam.T
        points_world = (points_cam - t) @ R_cam2world.T

        all_points.append(points_world)

        # 为点云添加颜色（这里用随机颜色，也可以从RGB图像采样）
        colors = np.random.randint(100, 200, size=(len(points_world), 3), dtype=np.uint8)
        all_colors.append(colors)

        print(f"  {depth_file}: 采样了 {len(points_world)} 个点")

    if len(all_points) == 0:
        print("\n❌ 错误: 没有生成任何点！")
        return False

    # 合并所有点
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    print(f"\n总共生成 {len(all_points)} 个点")

    # 保存为PLY格式
    print(f"保存到 {output_path}...")

    # 创建PLY数据
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(all_points)
    elements = np.empty(len(all_points), dtype=dtype)
    elements['x'] = all_points[:, 0]
    elements['y'] = all_points[:, 1]
    elements['z'] = all_points[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['red'] = all_colors[:, 0]
    elements['green'] = all_colors[:, 1]
    elements['blue'] = all_colors[:, 2]

    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(output_path)

    print(f"✓ 完成！")
    return True

if __name__ == "__main__":
    dataset = "data2"

    print("="*80)
    print(f"从深度图和mask重建 {dataset} 的初始点云")
    print("="*80)

    # 备份原始点云
    original_ply = f"{dataset}/sparse/0/points3D.ply"
    backup_ply = f"{dataset}/sparse/0/points3D_backup.ply"

    if os.path.exists(original_ply):
        if not os.path.exists(backup_ply):
            import shutil
            shutil.copy(original_ply, backup_ply)
            print(f"✓ 备份原始点云到 {backup_ply}")

    # 重建点云
    output_path = f"{dataset}/sparse/0/points3D.ply"
    success = reconstruct_pointcloud_from_masked_depth(dataset, output_path, max_points=50000)

    if success:
        print("\n" + "="*80)
        print("✓ 点云重建完成！")
        print("="*80)
        print(f"\n现在可以重新训练:")
        print(f"python train.py -s {dataset} -m output/{dataset}_fixed \\")
        print(f"  --depths depth --depth_mask_dir mask --iterations 30000")
    else:
        print("\n❌ 点云重建失败")
