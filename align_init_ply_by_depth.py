"""
通过深度图建立对应关系，计算精确的变换矩阵
策略：
1. 将init.ply投影到某个相机
2. 对比投影深度与实际深度图
3. 找到匹配的点对
4. 用匹配点对计算刚体变换（旋转+平移+尺度）
"""
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat

def find_point_correspondences():
    """
    通过投影和深度匹配找到init.ply与深度图之间的对应点
    """
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

    # 读取相机参数（标准COLMAP格式）
    cam_extrinsics = read_extrinsics_text('data/sparse/0/images.txt')
    cam_intrinsics = read_intrinsics_text('data/sparse/0/cameras.txt')
    cam_intr = list(cam_intrinsics.values())[0]
    fx, fy, cx, cy = cam_intr.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 对每个视角，尝试找到对应关系
    all_source_points = []
    all_target_points = []

    for img_id, extr in list(cam_extrinsics.items())[:3]:  # 用前3个视角
        depth_path = f'data/depth/{extr.name}'
        mask_path = f'data/mask/{extr.name}'

        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # mm -> m
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if depth_gt is None or mask is None:
            continue

        print(f"\n处理 {extr.name}...")

        # 1. 将init.ply投影到这个相机（尝试多种变换）
        R_cam = qvec2rotmat(extr.qvec)
        T_cam = np.array(extr.tvec)

        # 世界 -> 相机：X_cam = R * X_world + T
        xyz_cam = (R_cam @ xyz_init.T + T_cam.reshape(3, 1)).T

        # 只保留在相机前方的点
        in_front = xyz_cam[:, 2] > 0.01
        if in_front.sum() < 100:
            print(f"  跳过：只有{in_front.sum()}个点在前方")
            continue

        xyz_cam_valid = xyz_cam[in_front]
        xyz_init_valid = xyz_init[in_front]

        # 投影到图像
        xyz_img = K @ xyz_cam_valid.T
        uv = (xyz_img[:2, :] / xyz_img[2, :]).T
        depths_proj = xyz_cam_valid[:, 2]

        # 检查在图像内
        in_image = (uv[:, 0] >= 0) & (uv[:, 0] < 512) & (uv[:, 1] >= 0) & (uv[:, 1] < 424)
        uv_valid = uv[in_image].astype(int)
        depths_proj_valid = depths_proj[in_image]
        xyz_init_image = xyz_init_valid[in_image]

        if len(uv_valid) < 100:
            print(f"  跳过：只有{len(uv_valid)}个点在图像内")
            continue

        # 2. 对比投影深度与真实深度
        depth_diffs = []
        source_pts = []
        target_pts_cam = []

        for i, (u, v) in enumerate(uv_valid):
            # 检查是否在mask内
            if mask[v, u] < 128:
                continue

            depth_measured = depth_gt[v, u]
            if depth_measured <= 0:
                continue

            depth_projected = depths_proj_valid[i]

            # 深度差异（相对）
            depth_diff = abs(depth_projected - depth_measured) / depth_measured

            if depth_diff < 0.5:  # 深度差异小于50%认为是匹配的
                # 源点（init.ply中的3D点）
                source_pts.append(xyz_init_image[i])

                # 目标点（从深度图反投影的3D点，在相机坐标系）
                x_cam = (u - cx) * depth_measured / fx
                y_cam = (v - cy) * depth_measured / fy
                z_cam = depth_measured
                target_pts_cam.append([x_cam, y_cam, z_cam])

        if len(source_pts) < 10:
            print(f"  跳过：只找到{len(source_pts)}个匹配点")
            continue

        # 将目标点转换到世界坐标系
        target_pts_cam = np.array(target_pts_cam)
        # X_world = R^T * (X_cam - T)
        target_pts_world = (R_cam.T @ (target_pts_cam - T_cam).T).T

        all_source_points.extend(source_pts)
        all_target_points.append(target_pts_world)

        print(f"  找到 {len(source_pts)} 个匹配点对")

    if len(all_source_points) == 0:
        print("\n❌ 没有找到足够的对应点！")
        return None

    source_points = np.array(all_source_points)
    target_points = np.vstack(all_target_points)

    print(f"\n总共找到 {len(source_points)} 个对应点对")
    print(f"源点云 (init.ply) 范围:")
    print(f"  X: [{source_points[:, 0].min():.3f}, {source_points[:, 0].max():.3f}]")
    print(f"  Y: [{source_points[:, 1].min():.3f}, {source_points[:, 1].max():.3f}]")
    print(f"  Z: [{source_points[:, 2].min():.3f}, {source_points[:, 2].max():.3f}]")

    print(f"目标点云 (深度图) 范围:")
    print(f"  X: [{target_points[:, 0].min():.3f}, {target_points[:, 0].max():.3f}]")
    print(f"  Y: [{target_points[:, 1].min():.3f}, {target_points[:, 1].max():.3f}]")
    print(f"  Z: [{target_points[:, 2].min():.3f}, {target_points[:, 2].max():.3f}]")

    return source_points, target_points, xyz_init, rgb_init


def compute_similarity_transform(source, target):
    """
    计算相似变换（旋转R + 平移t + 尺度s）
    target = s * R @ source + t
    """
    # 中心化
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    # 计算尺度
    source_scale = np.linalg.norm(source_centered, axis=1).mean()
    target_scale = np.linalg.norm(target_centered, axis=1).mean()
    scale = target_scale / source_scale

    # 计算旋转（使用SVD）
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 确保是右手坐标系
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 计算平移
    t = target_mean - scale * R @ source_mean

    return R, t, scale


def apply_transform_and_save():
    """
    计算变换并应用到init.ply
    """
    result = find_point_correspondences()
    if result is None:
        print("\n使用备用方案：从深度图重建...")
        return

    source_pts, target_pts, xyz_init, rgb_init = result

    # 计算相似变换
    R, t, scale = compute_similarity_transform(source_pts, target_pts)

    print(f"\n计算的变换:")
    print(f"  旋转矩阵 R:")
    print(f"    {R[0]}")
    print(f"    {R[1]}")
    print(f"    {R[2]}")
    print(f"  平移 t: {t}")
    print(f"  尺度 s: {scale:.6f}")

    # 应用变换到整个init.ply
    xyz_transformed = scale * (R @ xyz_init.T).T + t

    print(f"\n变换后的点云范围:")
    print(f"  X: [{xyz_transformed[:, 0].min():.3f}, {xyz_transformed[:, 0].max():.3f}]")
    print(f"  Y: [{xyz_transformed[:, 1].min():.3f}, {xyz_transformed[:, 1].max():.3f}]")
    print(f"  Z: [{xyz_transformed[:, 2].min():.3f}, {xyz_transformed[:, 2].max():.3f}]")

    # 保存
    normals = np.zeros_like(xyz_transformed)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(xyz_transformed.shape[0], dtype=dtype)
    elements['x'] = xyz_transformed[:, 0].astype('f4')
    elements['y'] = xyz_transformed[:, 1].astype('f4')
    elements['z'] = xyz_transformed[:, 2].astype('f4')
    elements['nx'] = normals[:, 0].astype('f4')
    elements['ny'] = normals[:, 1].astype('f4')
    elements['nz'] = normals[:, 2].astype('f4')
    elements['red'] = rgb_init[:, 0]
    elements['green'] = rgb_init[:, 1]
    elements['blue'] = rgb_init[:, 2]

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write('data/sparse/0/points3D.ply')

    print(f"\n✅ 变换后的点云已保存到 data/sparse/0/points3D.ply")
    print(f"   保留了原始的 {len(xyz_init)} 个点和颜色信息")


if __name__ == "__main__":
    apply_transform_and_save()
