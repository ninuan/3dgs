#!/usr/bin/env python3
"""
关键发现：
- data的mask覆盖率更低(2%)，但重建效果好
- data2的mask覆盖率更高(7-9%)，但重建效果差

这说明问题不在mask覆盖率！需要找其他原因。

可能的原因：
1. 初始点云和mask区域的对齐问题
2. 深度值的准确性问题
3. 深度对齐算法在data2上失效
4. 训练过程的其他问题
"""

import numpy as np
import cv2
import os
from plyfile import PlyData
from scene.colmap_loader import read_extrinsics_text, qvec2rotmat

def check_pointcloud_mask_alignment(dataset):
    """
    检查初始点云是否覆盖了mask区域
    如果初始点云在mask外，即使有深度监督也没用
    """
    print(f"\n{'='*80}")
    print(f"检查 {dataset} 的点云-mask对齐情况")
    print(f"{'='*80}")

    # 加载点云
    ply_path = f"{dataset}/sparse/0/points3D.ply"
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    # 加载相机
    images_txt = f"{dataset}/sparse/0/images.txt"
    cam_extrinsics = read_extrinsics_text(images_txt)

    # 选择几个视角检查
    depth_dir = f"{dataset}/depth"
    mask_dir = f"{dataset}/mask"
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])[:3]

    from scene.colmap_loader import read_intrinsics_text
    cameras_txt = f"{dataset}/sparse/0/cameras.txt"
    cam_intrinsics = read_intrinsics_text(cameras_txt)
    cam_intr = list(cam_intrinsics.values())[0]

    fx = cam_intr.params[0] if cam_intr.model == "SIMPLE_PINHOLE" else cam_intr.params[0]
    fy = cam_intr.params[1] if cam_intr.model == "PINHOLE" else fx
    cx = cam_intr.params[2] if cam_intr.model == "PINHOLE" else cam_intr.params[1]
    cy = cam_intr.params[3] if cam_intr.model == "PINHOLE" else cam_intr.params[2]
    width, height = cam_intr.width, cam_intr.height

    for depth_file in depth_files:
        base = os.path.splitext(depth_file)[0]
        mask_path = os.path.join(mask_dir, f"{base}.png")

        # 找到对应的相机
        extr = None
        for key, e in cam_extrinsics.items():
            if e.name == depth_file:
                extr = e
                break

        if extr is None:
            continue

        # 相机外参
        R_world2cam = qvec2rotmat(extr.qvec)
        t = np.array(extr.tvec)

        # 投影点云到相机
        points_cam = (R_world2cam @ points.T).T + t

        # 过滤在相机前方的点
        valid_depth_mask = points_cam[:, 2] > 0.1
        points_cam_valid = points_cam[valid_depth_mask]

        if len(points_cam_valid) == 0:
            print(f"{depth_file}: 没有点在相机前方！")
            continue

        # 投影到像素坐标
        u = points_cam_valid[:, 0] / points_cam_valid[:, 2] * fx + cx
        v = points_cam_valid[:, 1] / points_cam_valid[:, 2] * fy + cy

        # 过滤在图像范围内的点
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u_valid = u[in_image].astype(int)
        v_valid = v[in_image].astype(int)

        print(f"\n{depth_file}:")
        print(f"  投影到图像的点数: {len(u_valid)} / {len(points)}")

        if len(u_valid) == 0:
            print(f"  ⚠️ 没有点投影到图像内！")
            continue

        # 加载mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        # 检查投影点有多少在mask内
        in_mask_count = mask[v_valid, u_valid].sum() / 255
        print(f"  投影点在mask内的数量: {in_mask_count:.0f} / {len(u_valid)} ({100*in_mask_count/len(u_valid):.1f}%)")

        # 创建点云投影的可视化
        proj_mask = np.zeros_like(mask)
        proj_mask[v_valid, u_valid] = 255

        overlap = ((proj_mask > 0) & (mask > 0)).sum()
        print(f"  点云投影和mask的重叠像素: {overlap} ({100*overlap/mask.size:.2f}%)")

        if in_mask_count / len(u_valid) < 0.1:
            print(f"  ⚠️ 警告: 投影点几乎不在mask内！初始化会很差")

check_pointcloud_mask_alignment('data')
check_pointcloud_mask_alignment('data2')

print("\n" + "="*80)
print("关键诊断")
print("="*80)
print("""
如果初始点云投影后不在mask区域内：
- 即使有深度监督，Gaussian初始化也是错的
- 训练会从错误的位置开始，很难收敛
- data可能初始点云恰好对齐，data2可能不对齐

这可能就是为什么data2效果差的原因！
""")
