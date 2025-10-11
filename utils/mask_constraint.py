"""
在训练过程中添加mask约束的增强densification策略
用于限制Gaussian只在目标物体区域生长
"""
import torch

class MaskConstrainedDensification:
    """
    使用mask约束densification，防止点云发散到目标外
    """

    @staticmethod
    def check_points_in_mask(gaussians, viewpoint_cameras, render_func, pipe, background):
        """
        检查Gaussian点在所有视图mask中的可见性
        返回每个点在mask内的投影次数
        """
        xyz = gaussians.get_xyz
        point_mask_counts = torch.zeros(xyz.shape[0], device="cuda")

        for cam in viewpoint_cameras:
            if cam.depth_mask is None:
                continue

            # 渲染当前视图
            render_pkg = render_func(cam, gaussians, pipe, background)

            # 获取深度图和mask
            rendered_depth = render_pkg["depth"]
            mask = cam.depth_mask.cuda()

            # 投影点到图像平面并检查是否在mask内
            # 这里简化处理：使用渲染的深度图来判断可见性
            valid_mask = (mask > 0.5).float()

            # 统计每个点在有效mask区域的出现次数
            # 实际实现需要更精确的投影计算

        return point_mask_counts

    @staticmethod
    def prune_outside_mask(gaussians, scene, render_func, pipe, background, min_visibility=1):
        """
        裁剪不在任何mask内的Gaussian点

        Args:
            min_visibility: 最少需要在多少个视图的mask内可见
        """
        viewpoint_cameras = scene.getTrainCameras()

        # 检查哪些相机有mask
        cameras_with_mask = [cam for cam in viewpoint_cameras if cam.depth_mask is not None]

        if len(cameras_with_mask) == 0:
            print("[Warn] No masks found, skipping mask-based pruning")
            return

        xyz = gaussians.get_xyz
        keep_mask = torch.ones(xyz.shape[0], dtype=torch.bool, device="cuda")

        # 对每个相机进行检查
        for cam in cameras_with_mask:
            # 获取相机参数
            W2C = torch.tensor(getWorld2View2(cam.R, cam.T)).cuda().float()
            proj_matrix = getProjectionMatrix(
                znear=0.01, zfar=100.0,
                fovX=cam.FoVx, fovY=cam.FoVy
            ).transpose(0,1).cuda()
            full_proj = (W2C.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)

            # 投影3D点到2D
            xyz_h = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device="cuda")], dim=1)
            xyz_proj = xyz_h @ full_proj.T
            xyz_proj = xyz_proj / (xyz_proj[:, 3:4] + 1e-7)

            # 转换到图像坐标
            x = ((xyz_proj[:, 0] + 1) * 0.5 * cam.image_width).long()
            y = ((xyz_proj[:, 1] + 1) * 0.5 * cam.image_height).long()

            # 检查是否在图像范围内
            valid = (x >= 0) & (x < cam.image_width) & (y >= 0) & (y < cam.image_height) & (xyz_proj[:, 2] > 0)

            # 检查是否在mask内
            mask = cam.depth_mask.cuda()
            in_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device="cuda")
            valid_indices = torch.where(valid)[0]

            for idx in valid_indices:
                px, py = x[idx].item(), y[idx].item()
                if mask[0, py, px] > 0.5:  # mask是(1, H, W)
                    in_mask[idx] = True

            keep_mask = keep_mask & in_mask

        # 裁剪不在mask内的点
        prune_mask = ~keep_mask
        num_pruned = prune_mask.sum().item()

        if num_pruned > 0:
            print(f"[Mask Pruning] Removing {num_pruned} points outside mask regions")
            gaussians.prune_points(prune_mask)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """相机外参矩阵"""
    import numpy as np
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """投影矩阵"""
    import math
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = -1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P
