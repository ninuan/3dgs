#!/usr/bin/env python3
"""
TUM RGB-D to 3DGS - Direct Conversion (No COLMAP)
使用TUM groundtruth轨迹直接生成点云和COLMAP格式文件

Usage:
    python convert_tum_to_3dgs_direct.py \
        --input rgbd_dataset_freiburg3_large_cabinet/rgbd_dataset_freiburg3_large_cabinet \
        --output data_tum_cabinet \
        --skip_frames 5
"""

import argparse
import os
import shutil
import numpy as np
from PIL import Image
import cv2
import sys


def read_file_list(filename):
    """读取TUM格式的文件列表（时间戳 文件名）"""
    file_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                timestamp = float(parts[0])
                data = parts[1:]
                file_dict[timestamp] = data
    return file_dict


def read_trajectory(filename):
    """读取groundtruth轨迹（时间戳 tx ty tz qx qy qz qw）"""
    trajectory = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                timestamp = float(parts[0])
                # 位置 + 四元数
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                trajectory[timestamp] = {
                    'position': np.array([tx, ty, tz]),
                    'quaternion': np.array([qw, qx, qy, qz])  # 注意：COLMAP用wxyz顺序
                }
    return trajectory


def associate(first_list, second_list, max_difference=0.02):
    """关联两个时间戳列表"""
    matches = []
    first_keys = sorted(first_list.keys())
    second_keys = sorted(second_list.keys())

    for first_ts in first_keys:
        # 找最接近的second时间戳
        closest_ts = min(second_keys, key=lambda x: abs(x - first_ts))
        if abs(closest_ts - first_ts) < max_difference:
            matches.append((first_ts, closest_ts))

    return matches


def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵 (qw, qx, qy, qz)"""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R


def process_dataset(args):
    """主处理流程"""
    tum_root = args.input
    output_root = args.output

    print("\n" + "="*70)
    print("TUM RGB-D to 3DGS Direct Converter (No COLMAP Required)")
    print("="*70 + "\n")

    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    images_dir = os.path.join(output_root, 'images')
    depth_dir = os.path.join(output_root, 'depth')
    mask_dir = os.path.join(output_root, 'mask')
    sparse_dir = os.path.join(output_root, 'sparse', '0')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # === 步骤1: 读取数据 ===
    print("[1/7] 读取TUM数据集文件...")
    rgb_file = os.path.join(tum_root, 'rgb.txt')
    depth_file = os.path.join(tum_root, 'depth.txt')
    gt_file = os.path.join(tum_root, 'groundtruth.txt')

    rgb_list = read_file_list(rgb_file)
    depth_list = read_file_list(depth_file)
    trajectory = read_trajectory(gt_file)

    print(f"  RGB图像: {len(rgb_list)}")
    print(f"  深度图像: {len(depth_list)}")
    print(f"  轨迹点: {len(trajectory)}")

    # === 步骤2: 关联RGB-Depth-Pose ===
    print("\n[2/7] 关联RGB、深度和位姿...")
    # 先关联RGB和depth
    rgb_depth_matches = associate(rgb_list, depth_list, max_difference=0.02)
    print(f"  RGB-Depth匹配: {len(rgb_depth_matches)} 对")

    # 再关联到轨迹
    matched_frames = []
    for rgb_ts, depth_ts in rgb_depth_matches:
        # 找最接近的轨迹时间戳
        closest_pose_ts = min(trajectory.keys(), key=lambda x: abs(x - rgb_ts))
        if abs(closest_pose_ts - rgb_ts) < 0.05:  # 50ms容差
            matched_frames.append({
                'rgb_ts': rgb_ts,
                'depth_ts': depth_ts,
                'pose_ts': closest_pose_ts,
                'rgb_file': rgb_list[rgb_ts][0],
                'depth_file': depth_list[depth_ts][0],
                'pose': trajectory[closest_pose_ts]
            })

    print(f"  完整匹配（RGB+Depth+Pose）: {len(matched_frames)} 帧")

    # 跳帧采样
    if args.skip_frames > 1:
        matched_frames = matched_frames[::args.skip_frames]
        print(f"  跳帧采样（每{args.skip_frames}帧取1）: {len(matched_frames)} 帧")

    # === 步骤3: 复制并重命名图像 ===
    print("\n[3/7] 复制图像...")
    for idx, frame in enumerate(matched_frames):
        new_name = f"{idx:06d}.png"

        # 复制RGB
        src_rgb = os.path.join(tum_root, frame['rgb_file'])
        dst_rgb = os.path.join(images_dir, new_name)
        shutil.copy2(src_rgb, dst_rgb)

        # 复制深度
        src_depth = os.path.join(tum_root, frame['depth_file'])
        dst_depth = os.path.join(depth_dir, new_name)
        shutil.copy2(src_depth, dst_depth)

        frame['new_name'] = new_name
        frame['frame_id'] = idx

    print(f"  完成: {len(matched_frames)} 张图像")

    # === 步骤4: 生成深度mask ===
    print("\n[4/7] 生成深度mask...")
    for frame in matched_frames:
        depth_path = os.path.join(depth_dir, frame['new_name'])
        mask_path = os.path.join(mask_dir, frame['new_name'])

        # 读取深度（16位PNG）
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            continue

        # TUM深度转米：depth_value / 5000.0
        depth_m = depth.astype(np.float32) / 5000.0

        # 生成mask：有效深度范围
        mask = ((depth_m > args.min_depth) & (depth_m < args.max_depth) & (depth > 0)).astype(np.uint8) * 255

        cv2.imwrite(mask_path, mask)

    print(f"  完成: {len(matched_frames)} 个mask")

    # === 步骤5: 创建相机内参文件 ===
    print("\n[5/7] 创建相机内参...")

    # 使用TUM官方 generate_pointcloud.py 中的参数
    # 这些参数经过TUM官方验证，适用于所有Freiburg数据集
    # 参考：rgbd_dataset_freiburg*/generate_pointcloud.py
    fx_depth, fy_depth = 525.0, 525.0
    cx_depth, cy_depth = 319.5, 239.5

    # RGB相机内参（用于COLMAP cameras.txt，但点云生成使用深度相机内参）
    if 'freiburg3' in tum_root.lower():
        fx_rgb, fy_rgb = 535.4, 539.2
        cx_rgb, cy_rgb = 320.1, 247.6
    elif 'freiburg2' in tum_root.lower():
        fx_rgb, fy_rgb = 520.9, 521.0
        cx_rgb, cy_rgb = 325.1, 249.7
    else:  # freiburg1 默认
        fx_rgb, fy_rgb = 517.3, 516.5
        cx_rgb, cy_rgb = 318.6, 255.3

    # 检测图像尺寸
    first_image = os.path.join(images_dir, matched_frames[0]['new_name'])
    img = Image.open(first_image)
    width, height = img.size

    print(f"  图像尺寸: {width}x{height}")
    print(f"  RGB内参: fx={fx_rgb:.2f}, fy={fy_rgb:.2f}, cx={cx_rgb:.2f}, cy={cy_rgb:.2f}")

    # 保存RGB内参
    K_rgb = np.array([
        [fx_rgb, 0.0, cx_rgb],
        [0.0, fy_rgb, cy_rgb],
        [0.0, 0.0, 1.0]
    ])
    np.savetxt(os.path.join(output_root, 'intrinsic_rgb.txt'), K_rgb, fmt='%.18e')

    # 保存深度内参
    K_depth = np.array([
        [fx_depth, 0.0, cx_depth],
        [0.0, fy_depth, cy_depth],
        [0.0, 0.0, 1.0]
    ])
    np.savetxt(os.path.join(output_root, 'intrinsic_ir.txt'), K_depth, fmt='%.18e')

    # === 步骤6: 生成全局点云 ===
    print("\n[6/7] 从RGB-D生成全局点云...")
    print(f"  处理 {len(matched_frames)} 帧（这可能需要几分钟）...")

    all_points = []
    all_colors = []

    for idx, frame in enumerate(matched_frames):
        if idx % 10 == 0:
            print(f"    处理帧 {idx+1}/{len(matched_frames)}...")

        # 读取RGB、深度和mask
        rgb_path = os.path.join(images_dir, frame['new_name'])
        depth_path = os.path.join(depth_dir, frame['new_name'])
        mask_path = os.path.join(mask_dir, frame['new_name'])

        rgb_img = np.array(Image.open(rgb_path))
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        # 读取mask（如果存在）
        mask_img = None
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 相机位姿
        position = frame['pose']['position']
        quaternion = frame['pose']['quaternion']
        R = quaternion_to_rotation_matrix(quaternion)

        # 反投影生成局部点云
        h, w = depth_img.shape
        for v in range(0, h, args.point_sample_step):
            for u in range(0, w, args.point_sample_step):
                Z = depth_img[v, u] / 5000.0  # TUM depth单位

                if Z < args.min_depth or Z > args.max_depth or Z == 0:
                    continue

                # 如果有mask，只生成mask区域内的点
                if mask_img is not None and mask_img[v, u] == 0:
                    continue

                # 相机坐标系（使用深度相机内参，而非RGB内参！）
                X = (u - cx_depth) * Z / fx_depth
                Y = (v - cy_depth) * Z / fy_depth
                point_cam = np.array([X, Y, Z])

                # 转到世界坐标系
                point_world = R @ point_cam + position

                # 颜色
                color = rgb_img[v, u]

                all_points.append(point_world)
                all_colors.append(color)

    all_points = np.array(all_points)
    all_colors = np.array(all_colors)

    print(f"  生成点云: {len(all_points)} 个点")

    # 下采样（避免点云过大）
    if len(all_points) > args.max_points:
        print(f"  下采样到 {args.max_points} 个点...")
        indices = np.random.choice(len(all_points), args.max_points, replace=False)
        all_points = all_points[indices]
        all_colors = all_colors[indices]

    # 保存点云为PLY
    ply_file = os.path.join(sparse_dir, 'points3D.ply')
    save_ply(ply_file, all_points, all_colors)
    print(f"  保存点云: {ply_file}")

    # === 步骤7: 创建COLMAP格式文件 ===
    print("\n[7/7] 创建COLMAP格式文件...")

    # cameras.txt
    # **BUG修复**: 必须使用深度相机内参，因为：
    # 1. 点云生成使用深度相机内参（Line 273-274）
    # 2. 深度监督的GT来自深度相机
    # 3. 如果使用RGB相机内参，会导致深度值系统性misaligned
    cameras_file = os.path.join(sparse_dir, 'cameras.txt')
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        # 使用深度相机内参，与点云生成和深度图保持一致
        f.write(f"1 PINHOLE {width} {height} {fx_depth} {fy_depth} {cx_depth} {cy_depth}\n")

    print(f"  创建: cameras.txt")

    # images.txt
    images_file = os.path.join(sparse_dir, 'images.txt')
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(matched_frames)}\n")

        for frame in matched_frames:
            image_id = frame['frame_id'] + 1

            # TUM provides camera-to-world transformation
            q_c2w = frame['pose']['quaternion']  # [qw, qx, qy, qz] - camera to world
            t_c2w = frame['pose']['position']    # [tx, ty, tz] - camera position in world

            # COLMAP requires world-to-camera transformation
            # 1. Build camera-to-world rotation matrix
            R_c2w = quaternion_to_rotation_matrix(q_c2w)

            # 2. Invert to get world-to-camera
            R_w2c = R_c2w.T

            # 3. Convert camera position to COLMAP's t format: t = -R_w2c @ C
            t_w2c = -R_w2c @ t_c2w

            # 4. Convert rotation matrix back to quaternion (world-to-camera)
            # Quaternion from rotation matrix
            trace = np.trace(R_w2c)
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                qw = 0.25 / s
                qx = (R_w2c[2, 1] - R_w2c[1, 2]) * s
                qy = (R_w2c[0, 2] - R_w2c[2, 0]) * s
                qz = (R_w2c[1, 0] - R_w2c[0, 1]) * s
            else:
                if R_w2c[0, 0] > R_w2c[1, 1] and R_w2c[0, 0] > R_w2c[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R_w2c[0, 0] - R_w2c[1, 1] - R_w2c[2, 2])
                    qw = (R_w2c[2, 1] - R_w2c[1, 2]) / s
                    qx = 0.25 * s
                    qy = (R_w2c[0, 1] + R_w2c[1, 0]) / s
                    qz = (R_w2c[0, 2] + R_w2c[2, 0]) / s
                elif R_w2c[1, 1] > R_w2c[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R_w2c[1, 1] - R_w2c[0, 0] - R_w2c[2, 2])
                    qw = (R_w2c[0, 2] - R_w2c[2, 0]) / s
                    qx = (R_w2c[0, 1] + R_w2c[1, 0]) / s
                    qy = 0.25 * s
                    qz = (R_w2c[1, 2] + R_w2c[2, 1]) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + R_w2c[2, 2] - R_w2c[0, 0] - R_w2c[1, 1])
                    qw = (R_w2c[1, 0] - R_w2c[0, 1]) / s
                    qx = (R_w2c[0, 2] + R_w2c[2, 0]) / s
                    qy = (R_w2c[1, 2] + R_w2c[2, 1]) / s
                    qz = 0.25 * s

            f.write(f"{image_id} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {frame['new_name']}\n")
            f.write("\n")  # 空行（没有2D特征点）

    print(f"  创建: images.txt")

    # points3D.txt（可选，3DGS主要用.ply）
    points3d_file = os.path.join(sparse_dir, 'points3D.txt')
    with open(points3d_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(all_points)}\n")

        for i, (pt, color) in enumerate(zip(all_points[:1000], all_colors[:1000])):  # 只写前1000个点
            f.write(f"{i+1} {pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]} 0.0\n")

    print(f"  创建: points3D.txt (前1000个点)")

    # 保存转换总结
    summary_file = os.path.join(output_root, 'conversion_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TUM RGB-D to 3DGS Conversion Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"输入: {tum_root}\n")
        f.write(f"输出: {output_root}\n\n")
        f.write(f"统计信息:\n")
        f.write(f"  - 图像数量: {len(matched_frames)}\n")
        f.write(f"  - 图像尺寸: {width}x{height}\n")
        f.write(f"  - 点云大小: {len(all_points)} 个点\n")
        f.write(f"  - 跳帧: 每{args.skip_frames}帧取1帧\n")
        f.write(f"  - 深度范围: {args.min_depth}m ~ {args.max_depth}m\n\n")
        f.write(f"相机内参:\n")
        f.write(f"  fx={fx_rgb:.2f}, fy={fy_rgb:.2f}, cx={cx_rgb:.2f}, cy={cy_rgb:.2f}\n\n")
        f.write(f"训练命令:\n")
        f.write(f"  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH\n")
        f.write(f"  python train.py -s {output_root} -m output/tum_result \\\n")
        f.write(f"      --depths depth --depth_mask_dir mask \\\n")
        f.write(f"      --iterations 30000 --disable_viewer\n")

    print(f"  创建: conversion_summary.txt")

    print("\n" + "="*70)
    print("✓ 转换完成！")
    print("="*70)
    print(f"\n输出目录: {output_root}")
    print(f"包含:")
    print(f"  - images/: {len(matched_frames)} 张RGB图像")
    print(f"  - depth/: {len(matched_frames)} 张深度图")
    print(f"  - mask/: {len(matched_frames)} 个mask")
    print(f"  - sparse/0/: COLMAP格式数据 + {len(all_points)}点云")
    print(f"\n下一步: 运行训练")
    print(f"  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH")
    print(f"  python train.py -s {output_root} -m output/tum_result \\")
    print(f"      --depths depth --depth_mask_dir mask \\")
    print(f"      --iterations 30000 --disable_viewer\n")


def save_ply(filename, points, colors):
    """保存点云为PLY格式（带法向量）"""
    # 估计法向量（简单方法：使用随机法向量，3DGS训练时不太依赖初始法向）
    normals = np.zeros_like(points)
    normals[:, 2] = 1.0  # 默认指向Z轴正方向

    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Points
        for pt, normal, color in zip(points, normals, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                   f"{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f} "
                   f"{int(color[0])} {int(color[1])} {int(color[2])}\n")


def main():
    parser = argparse.ArgumentParser(description='Convert TUM RGB-D to 3DGS format (No COLMAP)')
    parser.add_argument('--input', required=True,
                        help='TUM dataset directory (containing rgb.txt, depth.txt, groundtruth.txt)')
    parser.add_argument('--output', required=True,
                        help='Output directory')
    parser.add_argument('--skip_frames', type=int, default=5,
                        help='Use every Nth frame (default: 5)')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Min valid depth in meters (default: 0.1)')
    parser.add_argument('--max_depth', type=float, default=6.0,
                        help='Max valid depth in meters (default: 6.0)')
    parser.add_argument('--max_points', type=int, default=100000,
                        help='Max points in final point cloud (default: 100000)')
    parser.add_argument('--point_sample_step', type=int, default=4,
                        help='Sample every Nth pixel for point cloud (default: 4)')

    args = parser.parse_args()

    # 检查输入
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        sys.exit(1)

    required_files = ['rgb.txt', 'depth.txt', 'groundtruth.txt']
    for fname in required_files:
        fpath = os.path.join(args.input, fname)
        if not os.path.exists(fpath):
            print(f"错误: 找不到必需文件: {fpath}")
            sys.exit(1)

    process_dataset(args)


if __name__ == '__main__':
    main()
