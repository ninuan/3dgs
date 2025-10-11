"""
计算mask约束的辅助函数
"""
import torch
import numpy as np


def compute_mask_constraint(gaussians, scene, render_func, pipe, background):
    """
    计算每个Gaussian点是否在至少一个视图的mask内

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

    # 对每个有mask的相机进行检查
    for cam in cameras_with_mask:
        try:
            # 使用相机自带的完整投影矩阵（与渲染器相同）
            full_proj_transform = cam.full_proj_transform  # [4, 4]

            # 世界坐标 -> 裁剪坐标
            xyz_h = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device="cuda")], dim=1)  # [N, 4]
            xyz_clip = xyz_h @ full_proj_transform.T  # [N, 4]

            # 只对w>threshold的点进行齐次除法（避免除以接近0的数）
            w_threshold = 0.0001
            valid_w = xyz_clip[:, 3] > w_threshold

            # 齐次除法 -> 归一化设备坐标 (NDC)
            xyz_ndc = xyz_clip[:, :3] / xyz_clip[:, 3:4].clamp(min=w_threshold)  # [N, 3]

            # NDC [-1, 1] -> 像素坐标 [0, width/height]
            x = ((xyz_ndc[:, 0] + 1) * 0.5 * cam.image_width).long()
            y = ((xyz_ndc[:, 1] + 1) * 0.5 * cam.image_height).long()

            # 检查是否在图像范围内（同时要求w>threshold）
            valid = valid_w & (x >= 0) & (x < cam.image_width) & (y >= 0) & (y < cam.image_height)

            # 获取mask
            mask = cam.depth_mask  # 已经在GPU上，[1, H, W]

            # 向量化操作：只检查有效点
            valid_indices = torch.where(valid)[0]

            if len(valid_indices) > 0:
                x_valid = x[valid_indices]
                y_valid = y[valid_indices]

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

    # 至少在一个视图的mask内的点被标记为有效
    valid_region_mask = point_in_mask_count > 0

    num_valid = valid_region_mask.sum().item()
    num_invalid = num_points - num_valid

    print(f"[Mask Constraint] {num_valid}/{num_points} points in valid region, {num_invalid} outside")

    return valid_region_mask
