"""
改进的方案1: 更宽松的深度一致性滤波
策略：
1. 大幅放宽深度阈值，保留更多原始点
2. 不强制要求在mask内（因为mask很小，会过滤掉太多点）
3. 只移除明显错误的点（深度差异很大的）
4. 增加补充点的密度
"""
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat

def improved_depth_filter(depth_threshold=0.5, remove_far_points=True, supplement_step=2):
    """
    改进的深度一致性滤波 - 更宽松的策略

    Args:
        depth_threshold: 相对深度误差阈值（0.5=50%，非常宽松）
        remove_far_points: 是否移除明显远离相机视野的点
        supplement_step: 补充点的采样步长（越小越密集）
    """
    print("="*70)
    print("改进方案1: 宽松的深度一致性优化")
    print("="*70)

    # 读取init.ply
    ply_init = PlyData.read('data/init.ply')
    xyz_init = np.stack([
        np.asarray(ply_init.elements[0]['x']),
        np.asarray(ply_init.elements[0]['y']),
        np.asarray(ply_init.elements[0]['z'])
    ], axis=1)
    rgb_init = np.stack([
        np.asarray(ply_init.elements[0]['red']),
        np.asarray(ply_init.elements[0]['green']),
        np.asarray(ply_init.elements[0]['blue'])
    ], axis=1)

    print(f"\n初始化点云: {len(xyz_init)} 个点")
    print(f"  X: [{xyz_init[:, 0].min():.3f}, {xyz_init[:, 0].max():.3f}]")
    print(f"  Y: [{xyz_init[:, 1].min():.3f}, {xyz_init[:, 1].max():.3f}]")
    print(f"  Z: [{xyz_init[:, 2].min():.3f}, {xyz_init[:, 2].max():.3f}]")

    # 读取相机参数
    cam_extrinsics = read_extrinsics_text('data/sparse/0/images.txt')
    cam_intrinsics = read_intrinsics_text('data/sparse/0/cameras.txt')
    cam_intr = list(cam_intrinsics.values())[0]
    fx, fy, cx, cy = cam_intr.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 标记要移除的点（默认保留所有）
    remove_mask = np.zeros(len(xyz_init), dtype=bool)

    # 记录每个点被看到和深度匹配的情况
    point_visible = np.zeros(len(xyz_init), dtype=int)  # 在多少个视角可见
    point_depth_match = np.zeros(len(xyz_init), dtype=int)  # 在多少个视角深度匹配

    # 收集需要补充的点
    additional_points = []
    additional_colors = []

    print(f"\n策略:")
    print(f"  深度阈值: {depth_threshold*100}% (非常宽松)")
    print(f"  补充点采样步长: {supplement_step} 像素")
    print(f"  移除策略: {'移除从未被看到或深度严重不匹配的点' if remove_far_points else '保留所有原始点'}")

    print(f"\n步骤1: 分析每个点的可见性和深度匹配情况")

    for img_id, extr in cam_extrinsics.items():
        print(f"\n  处理 {extr.name}...")

        # 读取深度图和mask
        depth_path = f'data/depth/{extr.name}'
        mask_path = f'data/mask/{extr.name}'

        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if depth_gt is None or mask is None:
            print(f"    跳过：无法读取深度图或mask")
            continue

        depth_gt = depth_gt.astype(np.float32) / 1000.0  # mm -> m

        # 构建相机矩阵
        R = qvec2rotmat(extr.qvec).T
        C = -np.dot(R, np.array(extr.tvec))

        # X_cam = R * (X_world - C)
        # xyz_cam = (R @ (xyz_init - C).T).T

        # 在相机前方
        # in_front = xyz_cam[:, 2] > 0.01

        # 投影到图像
        xyz_cam_valid = xyz_cam[in_front]
        xyz_img = K @ xyz_cam_valid.T
        uv = (xyz_img[:2, :] / xyz_img[2, :]).T
        depths_proj = xyz_cam_valid[:, 2]

        # 在图像内
        in_image = (uv[:, 0] >= 0) & (uv[:, 0] < 512) & (uv[:, 1] >= 0) & (uv[:, 1] < 424)
        uv_valid = uv[in_image].astype(int)
        depths_proj_valid = depths_proj[in_image]

        indices_front = np.where(in_front)[0]
        indices_valid = indices_front[in_image]

        if len(uv_valid) == 0:
            print(f"    跳过：没有点在图像内")
            continue

        # 统计可见性和深度匹配
        depth_match_count = 0

        for i, (u, v) in enumerate(uv_valid):
            idx = indices_valid[i]
            point_visible[idx] += 1

            # 获取真实深度
            depth_measured = depth_gt[v, u]
            depth_projected = depths_proj_valid[i]

            if depth_measured > 0:
                relative_error = abs(depth_projected - depth_measured) / depth_measured

                if relative_error < depth_threshold:
                    point_depth_match[idx] += 1
                    depth_match_count += 1

        print(f"    可见点: {len(uv_valid)}, 深度匹配: {depth_match_count}")

        # 补充点（更密集）
        mask_binary = mask > 128
        ys, xs = np.where(mask_binary)

        xs_sample = xs[::supplement_step]
        ys_sample = ys[::supplement_step]

        added_count = 0
        for u, v in zip(xs_sample, ys_sample):
            depth_measured = depth_gt[v, u]
            if depth_measured <= 0:
                continue

            # 反投影到3D
            x_cam = (u - cx) * depth_measured / fx
            y_cam = (v - cy) * depth_measured / fy
            z_cam = depth_measured
            pt_cam = np.array([x_cam, y_cam, z_cam])

            # 相机坐标 -> 世界坐标
            pt_world = R.T @ pt_cam + C

            additional_points.append(pt_world)
            # 使用原始颜色的平均值
            additional_colors.append([100, 100, 100])  # 灰色
            added_count += 1

        print(f"    补充点: {added_count}")

    # 步骤2: 决定保留哪些点
    print(f"\n步骤2: 筛选要保留的点")

    if remove_far_points:
        # 宽松策略：只移除从未被任何相机看到的点，或者被看到但深度严重不匹配的点
        # 如果一个点被看到了，但深度从不匹配（可能是遮挡或错误），考虑移除
        never_seen = point_visible == 0
        seen_but_never_match = (point_visible > 0) & (point_depth_match == 0)

        # 更宽松：只移除从未被看到的点
        remove_mask = never_seen

        print(f"  移除策略: 只移除从未被相机看到的点")
    else:
        # 保留所有原始点
        print(f"  保留策略: 保留所有原始点")

    keep_mask = ~remove_mask
    xyz_filtered = xyz_init[keep_mask]
    rgb_filtered = rgb_init[keep_mask]

    print(f"\n  原始点: {len(xyz_init)}")
    print(f"  从未被看到: {(point_visible == 0).sum()}")
    print(f"  被看到但深度不匹配: {((point_visible > 0) & (point_depth_match == 0)).sum()}")
    print(f"  深度匹配的点: {(point_depth_match > 0).sum()}")
    print(f"  保留的点: {len(xyz_filtered)}")
    print(f"  移除的点: {len(xyz_init) - len(xyz_filtered)}")

    # 步骤3: 添加补充的点
    print(f"\n步骤3: 补充缺失区域的点")
    print(f"  从深度图补充: {len(additional_points)} 个点")

    if len(additional_points) > 0:
        xyz_additional = np.array(additional_points)
        rgb_additional = np.array(additional_colors)

        xyz_final = np.vstack([xyz_filtered, xyz_additional])
        rgb_final = np.vstack([rgb_filtered, rgb_additional])
    else:
        xyz_final = xyz_filtered
        rgb_final = rgb_filtered

    print(f"\n最终点云:")
    print(f"  总点数: {len(xyz_final)}")
    print(f"  原始保留: {len(xyz_filtered)} ({len(xyz_filtered)/len(xyz_init)*100:.1f}%)")
    print(f"  新增补充: {len(additional_points)}")
    print(f"  X: [{xyz_final[:, 0].min():.3f}, {xyz_final[:, 0].max():.3f}]")
    print(f"  Y: [{xyz_final[:, 1].min():.3f}, {xyz_final[:, 1].max():.3f}]")
    print(f"  Z: [{xyz_final[:, 2].min():.3f}, {xyz_final[:, 2].max():.3f}]")

    # 保存
    normals = np.zeros_like(xyz_final)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(xyz_final.shape[0], dtype=dtype)
    elements['x'] = xyz_final[:, 0].astype('f4')
    elements['y'] = xyz_final[:, 1].astype('f4')
    elements['z'] = xyz_final[:, 2].astype('f4')
    elements['nx'] = normals[:, 0].astype('f4')
    elements['ny'] = normals[:, 1].astype('f4')
    elements['nz'] = normals[:, 2].astype('f4')
    elements['red'] = rgb_final[:, 0]
    elements['green'] = rgb_final[:, 1]
    elements['blue'] = rgb_final[:, 2]

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write('data/sparse/0/points3D.ply')

    print(f"\n✅ 优化后的点云已保存到 data/sparse/0/points3D.ply")

    return xyz_final, rgb_final


if __name__ == "__main__":
    # 宽松参数：尽量保留原始点云
    depth_threshold = 0.5  # 50%误差容忍（非常宽松）
    remove_far_points = True  # 只移除从未被看到的点
    supplement_step = 2  # 更密集的补充点

    print(f"参数设置:")
    print(f"  深度阈值: {depth_threshold*100}% (宽松)")
    print(f"  移除远点: {remove_far_points}")
    print(f"  补充步长: {supplement_step} 像素")
    print()

    xyz_final, rgb_final = improved_depth_filter(
        depth_threshold=depth_threshold,
        remove_far_points=remove_far_points,
        supplement_step=supplement_step
    )
