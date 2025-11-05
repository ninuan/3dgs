#!/usr/bin/env python3
"""
修复data2的初始点云尺度，使其与深度图匹配

原理：
1. 计算点云投影深度 vs 深度图深度的比例
2. 用中位数比例缩放点云
3. 只修正data2，不影响其他数据集
"""

import numpy as np
import cv2
import os
import shutil
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat

def compute_optimal_scale(dataset, num_views=10):
    """
    计算点云需要缩放的比例，使其与深度图匹配

    返回：
        scale: float, 缩放比例（深度图/点云投影）
        None: 如果计算失败
    """
    print(f"\n正在分析 {dataset} 的尺度...")
    print("-" * 80)

    # 加载点云
    ply_path = f"{dataset}/sparse/0/points3D.ply"
    if not os.path.exists(ply_path):
        print(f"错误: 找不到 {ply_path}")
        return None

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    print(f"✓ 加载点云: {len(points)} 个点")

    # 加载相机
    images_txt = f"{dataset}/sparse/0/images.txt"
    cameras_txt = f"{dataset}/sparse/0/cameras.txt"

    if not os.path.exists(images_txt) or not os.path.exists(cameras_txt):
        print(f"错误: 找不到相机文件")
        return None

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
    print(f"✓ 加载相机参数: {width}x{height}, fx={fx:.1f}, fy={fy:.1f}")

    # 对多个视角计算比例
    depth_dir = f"{dataset}/depth"
    if not os.path.exists(depth_dir):
        print(f"错误: 找不到深度目录 {depth_dir}")
        return None

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])

    if len(depth_files) == 0:
        print(f"错误: {depth_dir} 中没有深度图")
        return None

    # 限制检查的视角数量
    depth_files = depth_files[:min(num_views, len(depth_files))]
    print(f"✓ 检查 {len(depth_files)} 个视角")
    print()

    ratios = []
    for depth_file in depth_files:
        # 找到对应的相机
        extr = None
        for key, e in cam_extrinsics.items():
            if e.name == depth_file:
                extr = e
                break

        if extr is None:
            print(f"  ⚠️  {depth_file}: 找不到相机外参，跳过")
            continue

        # 加载深度图
        depth_path = os.path.join(depth_dir, depth_file)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print(f"  ⚠️  {depth_file}: 无法加载深度图，跳过")
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
            print(f"  ⚠️  {depth_file}: 没有点在相机前方，跳过")
            continue

        # 投影到像素坐标
        u = points_cam_valid[:, 0] / points_cam_valid[:, 2] * fx + cx
        v = points_cam_valid[:, 1] / points_cam_valid[:, 2] * fy + cy

        # 过滤在图像范围内的点
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u_valid = u[in_image].astype(int)
        v_valid = v[in_image].astype(int)
        depths_proj = points_cam_valid[in_image, 2]

        if len(u_valid) == 0:
            print(f"  ⚠️  {depth_file}: 没有点投影到图像内，跳过")
            continue

        # 采样深度图的深度
        depths_map_sampled = depth_map[v_valid, u_valid]

        # 过滤深度图有效区域
        valid_both = (depths_map_sampled > 0.1) & (depths_map_sampled < 10.0)
        depths_proj_valid = depths_proj[valid_both]
        depths_map_valid = depths_map_sampled[valid_both]

        if len(depths_proj_valid) < 10:
            print(f"  ⚠️  {depth_file}: 重叠区域太少 ({len(depths_proj_valid)}个点)，跳过")
            continue

        # 计算比例
        median_proj = np.median(depths_proj_valid)
        median_map = np.median(depths_map_valid)
        ratio = median_map / median_proj

        ratios.append(ratio)

        # 显示每个视角的结果
        match_status = "✓" if abs(ratio - 1.0) < 0.15 else "✗"
        print(f"  {match_status} {depth_file:20s} | 点云:{median_proj:.3f}m  深度图:{median_map:.3f}m  比例:{ratio:.3f}")

    print()

    if len(ratios) == 0:
        print("错误: 无法计算缩放比例（所有视角都失败）")
        return None

    # 使用中位数作为最优比例（对异常值更鲁棒）
    optimal_scale = np.median(ratios)
    scale_std = np.std(ratios)

    print(f"统计结果:")
    print(f"  比例范围: [{np.min(ratios):.3f}, {np.max(ratios):.3f}]")
    print(f"  比例中位数: {optimal_scale:.3f}")
    print(f"  比例标准差: {scale_std:.3f}")
    print()

    return optimal_scale


def scale_pointcloud(dataset, scale):
    """
    缩放点云的所有坐标

    参数：
        dataset: 数据集路径
        scale: 缩放比例
    """
    ply_path = f"{dataset}/sparse/0/points3D.ply"
    backup_path = f"{dataset}/sparse/0/points3D_before_scale_fix.ply"

    # 备份原始点云
    if not os.path.exists(backup_path):
        shutil.copy(ply_path, backup_path)
        print(f"✓ 备份原始点云到: {backup_path}")
    else:
        print(f"✓ 备份已存在: {backup_path}")

    # 读取点云
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']

    # 缩放x, y, z坐标
    print(f"✓ 缩放点云坐标 (scale={scale:.3f})...")
    new_vertices = np.empty(len(vertices.data), dtype=vertices.data.dtype)

    for name in vertices.data.dtype.names:
        if name in ['x', 'y', 'z']:
            new_vertices[name] = vertices[name] * scale
        else:
            # 颜色、法向等属性不变
            new_vertices[name] = vertices[name]

    # 保存缩放后的点云
    vertex_element = PlyElement.describe(new_vertices, 'vertex')
    PlyData([vertex_element]).write(ply_path)

    print(f"✓ 已保存缩放后的点云到: {ply_path}")


if __name__ == "__main__":
    print("="*80)
    print("修复data2的点云-深度尺度不匹配问题")
    print("="*80)
    print()
    print("说明:")
    print("  - 此脚本只修正data2数据集")
    print("  - 会自动备份原始点云")
    print("  - 修正后可以正常训练，不会产生多个点云副本")
    print()

    dataset = "data2"

    # 检查数据集是否存在
    if not os.path.exists(dataset):
        print(f"错误: 找不到数据集目录 {dataset}")
        exit(1)

    # 计算最优缩放比例
    scale = compute_optimal_scale(dataset, num_views=10)

    if scale is None:
        print("\n失败: 无法计算缩放比例")
        print("请检查:")
        print("  1. data2/sparse/0/points3D.ply 是否存在")
        print("  2. data2/sparse/0/images.txt 和 cameras.txt 是否存在")
        print("  3. data2/depth/ 目录是否包含深度图")
        exit(1)

    # 判断是否需要修正
    if abs(scale - 1.0) < 0.1:
        print("="*80)
        print("结果: 尺度匹配良好，不需要修正")
        print("="*80)
        print(f"比例 {scale:.3f} 接近 1.0，点云和深度图已经对齐")
        exit(0)

    # 需要修正
    print("="*80)
    print(f"检测到尺度不匹配: 比例 = {scale:.3f}")
    print("="*80)

    if scale < 1.0:
        print(f"点云比深度图大了 {(1/scale - 1)*100:.1f}%，需要缩小")
    else:
        print(f"点云比深度图小了 {(scale - 1)*100:.1f}%，需要放大")

    print()

    # 用户确认
    response = input("是否继续修正? (y/n): ")
    if response.lower() != 'y':
        print("取消修正")
        exit(0)

    print()

    # 应用缩放
    scale_pointcloud(dataset, scale)

    print()
    print("="*80)
    print("✓ 修正完成！")
    print("="*80)
    print()
    print("下一步:")
    print("  重新训练模型:")
    print(f"  python train.py -s {dataset} -m output/{dataset}_fixed \\")
    print(f"    --depths depth --depth_mask_dir mask \\")
    print(f"    --iterations 30000 --disable_viewer")
    print()
    print("如果需要恢复原始点云:")
    print(f"  cp {dataset}/sparse/0/points3D_before_scale_fix.ply {dataset}/sparse/0/points3D.ply")
