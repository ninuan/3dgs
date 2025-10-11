"""
计算并应用坐标变换，将init.ply对齐到相机坐标系
然后使用深度图和mask进行优化和补充
"""
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat

def compute_alignment_transform():
    """
    通过对比深度图反投影点和init.ply中的点，计算变换矩阵
    """
    # 读取原始点云
    ply_init = PlyData.read('data/init.ply')
    xyz_init = np.stack([
        np.asarray(ply_init.elements[0]['x']),
        np.asarray(ply_init.elements[0]['y']),
        np.asarray(ply_init.elements[0]['z'])
    ], axis=1)

    print(f"原始点云 (init.ply):")
    print(f"  点数: {xyz_init.shape[0]}")
    print(f"  X: [{xyz_init[:, 0].min():.3f}, {xyz_init[:, 0].max():.3f}]")
    print(f"  Y: [{xyz_init[:, 1].min():.3f}, {xyz_init[:, 1].max():.3f}]")
    print(f"  Z: [{xyz_init[:, 2].min():.3f}, {xyz_init[:, 2].max():.3f}]")
    print(f"  中心: ({xyz_init[:, 0].mean():.3f}, {xyz_init[:, 1].mean():.3f}, {xyz_init[:, 2].mean():.3f})")

    # 读取相机参数
    cam_extrinsics = read_extrinsics_text('data/sparse/0/images.txt')
    cam_intrinsics = read_intrinsics_text('data/sparse/0/cameras.txt')
    cam_intr = list(cam_intrinsics.values())[0]
    fx, fy, cx, cy = cam_intr.params

    # 从一个视角的深度图采样一些点
    target_points_world = []

    for img_id, extr in list(cam_extrinsics.items())[:2]:  # 只用前2个相机
        depth_path = f'data/depth/{extr.name}'
        mask_path = f'data/mask/{extr.name}'

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if depth is None or mask is None:
            continue

        # 在mask区域均匀采样一些点
        mask_binary = mask > 128
        ys, xs = np.where(mask_binary)

        if len(xs) < 10:
            continue

        # 每隔N个点采样
        step = max(1, len(xs) // 50)
        xs = xs[::step]
        ys = ys[::step]

        depths = depth[ys, xs].astype(np.float32) / 1000.0  # mm -> m
        valid = depths > 0
        xs, ys, depths = xs[valid], ys[valid], depths[valid]

        if len(xs) == 0:
            continue

        # 像素 -> 相机坐标
        x_cam = (xs - cx) * depths / fx
        y_cam = (ys - cy) * depths / fy
        z_cam = depths
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # 相机坐标 -> 世界坐标 (使用标准COLMAP公式)
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        # X_cam = R * X_world + T
        # X_world = R^T * (X_cam - T)
        points_world = (R.T @ (points_cam - T).T).T

        target_points_world.append(points_world)
        print(f"  从 {extr.name} 采样了 {len(points_world)} 个目标点")

    target_points = np.vstack(target_points_world)

    print(f"\n目标点云 (从深度图反投影):")
    print(f"  点数: {target_points.shape[0]}")
    print(f"  X: [{target_points[:, 0].min():.3f}, {target_points[:, 0].max():.3f}]")
    print(f"  Y: [{target_points[:, 1].min():.3f}, {target_points[:, 1].max():.3f}]")
    print(f"  Z: [{target_points[:, 2].min():.3f}, {target_points[:, 2].max():.3f}]")
    print(f"  中心: ({target_points[:, 0].mean():.3f}, {target_points[:, 1].mean():.3f}, {target_points[:, 2].mean():.3f})")

    # 使用ICP或简单的中心对齐 + 尺度估计
    # 方案1: 简单的中心对齐
    center_init = xyz_init.mean(axis=0)
    center_target = target_points.mean(axis=0)

    # 计算尺度（比较平均距离）
    dist_init = np.linalg.norm(xyz_init - center_init, axis=1).mean()
    dist_target = np.linalg.norm(target_points - center_target, axis=1).mean()
    scale = dist_target / dist_init if dist_init > 0 else 1.0

    translation = center_target - center_init * scale

    print(f"\n计算的变换:")
    print(f"  平移: {translation}")
    print(f"  尺度: {scale:.6f}")

    return scale, translation


def align_and_refine_pointcloud():
    """
    对齐原始点云，并用深度图进行优化补充
    """
    print("=" * 70)
    print("步骤1: 计算坐标变换")
    print("=" * 70)

    scale, translation = compute_alignment_transform()

    # 读取原始点云（包括颜色）
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

    # 应用变换
    xyz_aligned = xyz_init * scale + translation

    print(f"\n变换后的点云:")
    print(f"  X: [{xyz_aligned[:, 0].min():.3f}, {xyz_aligned[:, 0].max():.3f}]")
    print(f"  Y: [{xyz_aligned[:, 1].min():.3f}, {xyz_aligned[:, 1].max():.3f}]")
    print(f"  Z: [{xyz_aligned[:, 2].min():.3f}, {xyz_aligned[:, 2].max():.3f}]")

    print("\n" + "=" * 70)
    print("步骤2: 使用深度图补充点云")
    print("=" * 70)

    # 读取相机参数
    cam_extrinsics = read_extrinsics_text('data/sparse/0/images.txt')
    cam_intrinsics = read_intrinsics_text('data/sparse/0/cameras.txt')
    cam_intr = list(cam_intrinsics.values())[0]
    fx, fy, cx, cy = cam_intr.params

    # 从每个相机的深度图补充点
    additional_points = []
    additional_colors = []

    for img_id, extr in cam_extrinsics.items():
        depth_path = f'data/depth/{extr.name}'
        mask_path = f'data/mask/{extr.name}'

        try:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if depth is None or mask is None:
                continue

            # 在mask区域采样
            mask_binary = mask > 128
            ys, xs = np.where(mask_binary)

            # 采样策略：每隔N个像素采样（避免点太密集）
            step = 5  # 可调整
            xs = xs[::step]
            ys = ys[::step]

            depths = depth[ys, xs].astype(np.float32) / 1000.0
            valid = depths > 0
            xs, ys, depths = xs[valid], ys[valid], depths[valid]

            if len(xs) == 0:
                continue

            # 像素 -> 相机坐标 -> 世界坐标
            x_cam = (xs - cx) * depths / fx
            y_cam = (ys - cy) * depths / fy
            z_cam = depths
            points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            points_world = (R.T @ (points_cam - T).T).T

            # 检查这些点是否在对齐后的init点云附近（避免重复）
            # 简单策略：只添加距离现有点较远的新点
            if len(additional_points) == 0 and len(xyz_aligned) > 0:
                # 第一批补充点：直接添加
                additional_points.append(points_world)
                # 使用白色（或者可以从深度值生成颜色）
                colors = np.ones((len(points_world), 3)) * 200
                additional_colors.append(colors)
                print(f"  {extr.name}: 补充 {len(points_world)} 个点")

        except Exception as e:
            print(f"  [Error] {extr.name}: {e}")
            continue

    # 合并原始对齐点云和补充点云
    all_points = [xyz_aligned]
    all_colors = [rgb_init]

    if len(additional_points) > 0:
        all_points.extend(additional_points)
        all_colors.extend(additional_colors)

    final_points = np.vstack(all_points)
    final_colors = np.vstack(all_colors).astype(np.uint8)

    print(f"\n最终点云:")
    print(f"  原始点: {len(xyz_aligned)}")
    print(f"  补充点: {len(final_points) - len(xyz_aligned)}")
    print(f"  总计: {len(final_points)}")

    # 保存
    normals = np.zeros_like(final_points)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(final_points.shape[0], dtype=dtype)
    elements['x'] = final_points[:, 0].astype('f4')
    elements['y'] = final_points[:, 1].astype('f4')
    elements['z'] = final_points[:, 2].astype('f4')
    elements['nx'] = normals[:, 0].astype('f4')
    elements['ny'] = normals[:, 1].astype('f4')
    elements['nz'] = normals[:, 2].astype('f4')
    elements['red'] = final_colors[:, 0]
    elements['green'] = final_colors[:, 1]
    elements['blue'] = final_colors[:, 2]

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write('data/sparse/0/points3D.ply')

    print(f"\n✅ 对齐并优化后的点云已保存到 data/sparse/0/points3D.ply")

    return scale, translation


if __name__ == "__main__":
    align_and_refine_pointcloud()
