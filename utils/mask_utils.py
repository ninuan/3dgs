"""
计算mask约束的辅助函数
"""
import torch
import numpy as np


def quat_to_rotation_matrix_torch(q):
    """
    Convert quaternion to rotation matrix (torch version)
    q: [qw, qx, qy, qz]
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    R = torch.zeros(3, 3, device=q.device, dtype=q.dtype)
    R[0, 0] = 1 - 2*qy**2 - 2*qz**2
    R[0, 1] = 2*qx*qy - 2*qz*qw
    R[0, 2] = 2*qx*qz + 2*qy*qw
    R[1, 0] = 2*qx*qy + 2*qz*qw
    R[1, 1] = 1 - 2*qx**2 - 2*qz**2
    R[1, 2] = 2*qy*qz - 2*qx*qw
    R[2, 0] = 2*qx*qz - 2*qy*qw
    R[2, 1] = 2*qy*qz + 2*qx*qw
    R[2, 2] = 1 - 2*qx**2 - 2*qy**2
    return R


def compute_mask_constraint(gaussians, scene, render_func, pipe, background):
    """
    计算每个Gaussian点是否在至少一个视图的mask内

    **修复**: 使用正确的COLMAP投影，而不是错误的full_proj_transform

    Args:
        gaussians: GaussianModel
        scene: Scene
        render_func: 渲染函数
        pipe: Pipeline参数
        background: 背景颜色

    Returns:
        valid_region_mask: Tensor[N], 布尔值，表示每个点是否在有效区域内
    """
    xyz = gaussians.get_xyz
    num_points = xyz.shape[0]

    # 初始化：所有点都标记为无效
    point_in_mask_count = torch.zeros(num_points, device="cuda", dtype=torch.int32)

    viewpoint_cameras = scene.getTrainCameras()

    # 检查哪些相机有mask
    cameras_with_mask = [cam for cam in viewpoint_cameras if cam.depth_mask is not None]

    if len(cameras_with_mask) == 0:
        print("[Warn] No masks found in any camera, skipping mask constraint")
        return None

    # 深度相机内参（硬编码，因为所有视图使用相同内参）
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5

    # 对每个有mask的相机进行检查
    for cam in cameras_with_mask:
        try:
            # **关键修复**: 使用正确的COLMAP world-to-camera投影
            # 不使用full_proj_transform（它是错误的）

            # COLMAP格式的R和T（world-to-camera）
            # 注意：cam.R和cam.T可能是numpy array，需要转torch tensor
            R_w2c = torch.from_numpy(cam.R).float() if isinstance(cam.R, np.ndarray) else cam.R
            t_w2c = torch.from_numpy(cam.T).float() if isinstance(cam.T, np.ndarray) else cam.T
            R_w2c = R_w2c.transpose(0, 1).cuda()  # cam.R是c2w的转置，所以R_w2c = cam.R.T.T = cam.R
            t_w2c = t_w2c.cuda()

            # World to camera
            xyz_cam = xyz @ R_w2c.T + t_w2c.unsqueeze(0)  # [N, 3]

            # 过滤在相机后面的点
            valid_depth = xyz_cam[:, 2] > 0.01

            # Project to image
            x = fx * xyz_cam[:, 0] / (xyz_cam[:, 2] + 1e-6) + cx
            y = fy * xyz_cam[:, 1] / (xyz_cam[:, 2] + 1e-6) + cy

            # 检查是否在图像范围内
            valid = valid_depth & (x >= 0) & (x < cam.image_width) & (y >= 0) & (y < cam.image_height)

            # 获取mask
            mask = cam.depth_mask  # [1, H, W] or [H, W]

            # 向量化操作：只检查有效点
            valid_indices = torch.where(valid)[0]

            if len(valid_indices) > 0:
                x_valid = x[valid_indices].long()
                y_valid = y[valid_indices].long()

                # 向量化索引mask值
                if mask.ndim == 3:
                    mask_values = mask[0, y_valid, x_valid]  # [num_valid]
                else:
                    mask_values = mask[y_valid, x_valid]

                # 更新计数：mask值>0.5的点
                in_mask = mask_values > 0.5
                point_in_mask_count[valid_indices[in_mask]] += 1

        except Exception as e:
            print(f"[Warn] Error computing mask constraint for camera {cam.image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 计算每个点被多少个相机看到（visibility count）
    point_visible_count = torch.zeros(num_points, device="cuda", dtype=torch.int32)

    for cam in cameras_with_mask:
        try:
            R_w2c = torch.from_numpy(cam.R).float() if isinstance(cam.R, np.ndarray) else cam.R
            t_w2c = torch.from_numpy(cam.T).float() if isinstance(cam.T, np.ndarray) else cam.T
            R_w2c = R_w2c.transpose(0, 1).cuda()
            t_w2c = t_w2c.cuda()
            xyz_cam = xyz @ R_w2c.T + t_w2c.unsqueeze(0)
            valid_depth = xyz_cam[:, 2] > 0.01
            x = fx * xyz_cam[:, 0] / (xyz_cam[:, 2] + 1e-6) + cx
            y = fy * xyz_cam[:, 1] / (xyz_cam[:, 2] + 1e-6) + cy
            valid = valid_depth & (x >= 0) & (x < cam.image_width) & (y >= 0) & (y < cam.image_height)
            point_visible_count[valid] += 1
        except:
            continue

    # 严格标准：
    # - 如果被至少3个相机看到：要求至少2个视图中在mask内（>= 50%）
    # - 如果被1-2个相机看到：要求全部都在mask内
    multi_view_points = point_visible_count >= 3
    few_view_points = (point_visible_count > 0) & (point_visible_count < 3)

    valid_region_mask = torch.zeros(num_points, device="cuda", dtype=torch.bool)
    valid_region_mask[multi_view_points] = point_in_mask_count[multi_view_points] >= 2
    valid_region_mask[few_view_points] = point_in_mask_count[few_view_points] == point_visible_count[few_view_points]

    num_valid = valid_region_mask.sum().item()
    num_invalid = num_points - num_valid

    print(f"[Mask Constraint] {num_valid}/{num_points} points in valid region, {num_invalid} outside")
    print(f"  Multi-view points (>=3 cams): {multi_view_points.sum().item()}, valid: {valid_region_mask[multi_view_points].sum().item()}")
    print(f"  Few-view points (1-2 cams): {few_view_points.sum().item()}, valid: {valid_region_mask[few_view_points].sum().item()}")

    return valid_region_mask
