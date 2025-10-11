"""
从深度图重建点云，保证与相机坐标系一致
"""
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat

def reconstruct_from_depth():
    # 读取相机参数
    cam_extrinsics = read_extrinsics_text('data/sparse/0/images.txt')
    cam_intrinsics = read_intrinsics_text('data/sparse/0/cameras.txt')

    # 获取相机内参
    cam_intr = list(cam_intrinsics.values())[0]
    fx, fy, cx, cy = cam_intr.params

    all_points = []
    all_colors = []

    # 对每个相机的深度图进行反投影
    for img_id, extr in cam_extrinsics.items():
        # 读取深度图和mask
        depth_path = f'data/depth/{extr.name}'
        mask_path = f'data/mask/{extr.name}'

        try:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if depth is None or mask is None:
                print(f"[Warn] 跳过 {extr.name}: 深度图或mask读取失败")
                continue

            # 只处理mask区域的像素
            mask_binary = mask > 128
            ys, xs = np.where(mask_binary)

            if len(xs) == 0:
                print(f"[Warn] 跳过 {extr.name}: mask为空")
                continue

            # 获取深度值 (从uint16转换为米, 假设单位是mm)
            depths = depth[ys, xs].astype(np.float32) / 1000.0  # mm -> m

            # 过滤无效深度
            valid = depths > 0
            xs = xs[valid]
            ys = ys[valid]
            depths = depths[valid]

            if len(xs) == 0:
                print(f"[Warn] 跳过 {extr.name}: 没有有效深度值")
                continue

            # 像素坐标 -> 相机坐标
            x_cam = (xs - cx) * depths / fx
            y_cam = (ys - cy) * depths / fy
            z_cam = depths

            # 相机坐标系中的点
            points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

            # 相机坐标 -> 世界坐标
            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)

            # 假设T是相机中心（非标准COLMAP）
            # X_cam = R * (X_world - T)
            # X_world = R^T * X_cam + T
            points_world = (R.T @ points_cam.T).T + T

            all_points.append(points_world)

            # 使用mask区域的颜色（这里用白色，因为没有RGB图像）
            colors = np.ones((len(points_world), 3)) * 255
            all_colors.append(colors)

            print(f"[Info] {extr.name}: 从mask区域重建了 {len(points_world)} 个点")

        except Exception as e:
            print(f"[Error] 处理 {extr.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_points) == 0:
        print("[Error] 没有成功重建任何点！")
        return

    # 合并所有点
    points = np.vstack(all_points)
    colors = np.vstack(all_colors).astype(np.uint8)

    print(f"\n总共重建了 {len(points)} 个点")
    print(f"点云范围:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    # 保存为PLY格式（带零法向量）
    normals = np.zeros_like(points)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(points.shape[0], dtype=dtype)
    elements['x'] = points[:, 0].astype('f4')
    elements['y'] = points[:, 1].astype('f4')
    elements['z'] = points[:, 2].astype('f4')
    elements['nx'] = normals[:, 0].astype('f4')
    elements['ny'] = normals[:, 1].astype('f4')
    elements['nz'] = normals[:, 2].astype('f4')
    elements['red'] = colors[:, 0]
    elements['green'] = colors[:, 1]
    elements['blue'] = colors[:, 2]

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write('data/sparse/0/points3D.ply')

    print(f"\n✅ 点云已保存到 data/sparse/0/points3D.ply")

if __name__ == "__main__":
    reconstruct_from_depth()
