"""
方案1: 深度一致性滤波
通过对比投影深度与真实深度图，保留一致的点，移除不一致的点，补充缺失区域的点
"""
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat

def depth_consistency_filter(depth_threshold=0.05, min_views=1):
    """
    深度一致性滤波

    Args:
        depth_threshold: 相对深度误差阈值（例如0.05表示5%）
        min_views: 至少在多少个视角中一致
    """
    print("="*70)
    print("方案1: 基于深度一致性的点云优化")
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

    print(f"\n原始点云: {len(xyz_init)} 个点")
    print(f"  X: [{xyz_init[:, 0].min():.3f}, {xyz_init[:, 0].max():.3f}]")
    print(f"  Y: [{xyz_init[:, 1].min():.3f}, {xyz_init[:, 1].max():.3f}]")
    print(f"  Z: [{xyz_init[:, 2].min():.3f}, {xyz_init[:, 2].max():.3f}]")

    # 读取相机参数
    cam_extrinsics = read_extrinsics_text('data/sparse/0/images.txt')
    cam_intrinsics = read_intrinsics_text('data/sparse/0/cameras.txt')
    cam_intr = list(cam_intrinsics.values())[0]
    fx, fy, cx, cy = cam_intr.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 为每个点记录一致性
    point_consistency = np.zeros(len(xyz_init), dtype=int)  # 在多少个视角中一致
    point_in_mask = np.zeros(len(xyz_init), dtype=bool)  # 是否在某个mask内

    # 收集需要补充的点
    additional_points = []
    additional_colors = []

    print(f"\n步骤1: 检查每个点的深度一致性")
    print(f"深度阈值: {depth_threshold*100}%")

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

        # 构建相机矩阵 (使用相机中心格式)
        R = qvec2rotmat(extr.qvec)
        C = np.array(extr.tvec)  # 相机中心

        # X_cam = R * (X_world - C)
        xyz_cam = (R @ (xyz_init - C).T).T

        # 只处理在相机前方的点
        in_front = xyz_cam[:, 2] > 0.01
        if in_front.sum() == 0:
            print(f"    跳过：没有点在相机前方")
            continue

        # 投影到图像
        xyz_cam_valid = xyz_cam[in_front]
        xyz_img = K @ xyz_cam_valid.T
        uv = (xyz_img[:2, :] / xyz_img[2, :]).T
        depths_proj = xyz_cam_valid[:, 2]

        # 检查在图像内
        in_image = (uv[:, 0] >= 0) & (uv[:, 0] < 512) & (uv[:, 1] >= 0) & (uv[:, 1] < 424)
        uv_valid = uv[in_image].astype(int)
        depths_proj_valid = depths_proj[in_image]

        # 获取原始点的索引
        indices_front = np.where(in_front)[0]
        indices_valid = indices_front[in_image]

        if len(uv_valid) == 0:
            print(f"    跳过：没有点在图像内")
            continue

        # 检查深度一致性
        consistent_count = 0
        in_mask_count = 0

        for i, (u, v) in enumerate(uv_valid):
            idx = indices_valid[i]

            # 检查是否在mask内
            in_mask_flag = mask[v, u] > 128
            if in_mask_flag:
                point_in_mask[idx] = True
                in_mask_count += 1

                # 获取真实深度
                depth_measured = depth_gt[v, u]
                depth_projected = depths_proj_valid[i]

                if depth_measured > 0:
                    # 计算相对误差
                    relative_error = abs(depth_projected - depth_measured) / depth_measured

                    if relative_error < depth_threshold:
                        point_consistency[idx] += 1
                        consistent_count += 1

        print(f"    在前方: {in_front.sum()}, 在图像内: {len(uv_valid)}, 在mask内: {in_mask_count}")
        print(f"    深度一致: {consistent_count}")

        # 补充缺失的点（在mask内但没有投影点覆盖的区域）
        mask_binary = mask > 128
        ys, xs = np.where(mask_binary)

        # 采样策略：每隔N个像素
        step = 3
        xs_sample = xs[::step]
        ys_sample = ys[::step]

        for u, v in zip(xs_sample, ys_sample):
            depth_measured = depth_gt[v, u]
            if depth_measured <= 0:
                continue

            # 检查这个像素附近是否已经有投影点了
            # 简单方法：如果(u,v)在uv_valid中距离很近的点，跳过
            if len(uv_valid) > 0:
                dist = np.sqrt((uv_valid[:, 0] - u)**2 + (uv_valid[:, 1] - v)**2)
                if dist.min() < 5:  # 5像素内已有点，跳过
                    continue

            # 反投影到3D
            x_cam = (u - cx) * depth_measured / fx
            y_cam = (v - cy) * depth_measured / fy
            z_cam = depth_measured
            pt_cam = np.array([x_cam, y_cam, z_cam])

            # 相机坐标 -> 世界坐标
            # X_world = R^T * X_cam + C
            pt_world = R.T @ pt_cam + C

            additional_points.append(pt_world)
            # 使用淡蓝色标记补充的点
            additional_colors.append([150, 150, 255])

    # 步骤2: 根据一致性筛选点
    print(f"\n步骤2: 筛选一致的点")
    print(f"  要求至少在 {min_views} 个视角中一致")

    # 保留的点：至少在min_views个视角中一致，或者在mask内但深度数据缺失
    keep_mask = (point_consistency >= min_views) | (point_in_mask & (point_consistency == 0))

    xyz_filtered = xyz_init[keep_mask]
    rgb_filtered = rgb_init[keep_mask]

    print(f"  原始点: {len(xyz_init)}")
    print(f"  在mask内的点: {point_in_mask.sum()}")
    print(f"  深度一致的点: {(point_consistency >= min_views).sum()}")
    print(f"  保留的点: {len(xyz_filtered)}")
    print(f"  移除的点: {len(xyz_init) - len(xyz_filtered)}")

    # 步骤3: 添加补充的点
    print(f"\n步骤3: 补充缺失区域的点")
    print(f"  从深度图补充: {len(additional_points)} 个点")

    if len(additional_points) > 0:
        xyz_additional = np.array(additional_points)
        rgb_additional = np.array(additional_colors)

        # 合并
        xyz_final = np.vstack([xyz_filtered, xyz_additional])
        rgb_final = np.vstack([rgb_filtered, rgb_additional])
    else:
        xyz_final = xyz_filtered
        rgb_final = rgb_filtered

    print(f"\n最终点云:")
    print(f"  总点数: {len(xyz_final)}")
    print(f"  原始保留: {len(xyz_filtered)}")
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
    # 可以调整参数
    depth_threshold = 0.1  # 10%深度误差容忍度
    min_views = 1  # 至少在1个视角中一致

    print(f"参数设置:")
    print(f"  深度阈值: {depth_threshold*100}%")
    print(f"  最少一致视角数: {min_views}")
    print()

    xyz_final, rgb_final = depth_consistency_filter(
        depth_threshold=depth_threshold,
        min_views=min_views
    )
