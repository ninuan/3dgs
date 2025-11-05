#
# Depth alignment utilities for monocular depth to COLMAP scale alignment
# Based on MonoSDF (NeurIPS 2022) and DN-Splatter methods
#

import torch
import numpy as np
import os


def align_depth_least_squares(mono_depth, colmap_sparse_depth, mask):
    """
    使用最小二乘法对齐单目深度到COLMAP稀疏深度

    基于线性模型: d_colmap = scale * d_mono + shift
    通过最小化 Σ mask * (d_colmap - scale * d_mono - shift)² 求解

    Args:
        mono_depth: [H, W] 单目深度预测 (torch.Tensor)
        colmap_sparse_depth: [H, W] COLMAP投影的稀疏深度，无效处为0 (torch.Tensor)
        mask: [H, W] 有效区域mask (torch.Tensor)

    Returns:
        aligned_depth: [H, W] 对齐后的深度
        scale: float, 缩放因子
        shift: float, 偏移量
    """
    mono_depth = mono_depth.detach()
    colmap_sparse_depth = colmap_sparse_depth.detach()
    mask = mask.detach()

    # 只在稀疏点有效且mask有效的区域计算
    valid = (colmap_sparse_depth > 0) * (mask > 0.5)

    if valid.sum() < 10:  # 稀疏点太少，返回原深度
        print(f"[Warning] Too few valid sparse points ({valid.sum()}), skip alignment")
        return mono_depth, 1.0, 0.0

    # 提取有效区域的深度值
    d_mono = mono_depth[valid].flatten()
    d_colmap = colmap_sparse_depth[valid].flatten()

    # 构建最小二乘问题: [d_mono, 1] @ [scale, shift]^T = d_colmap
    A = torch.stack([d_mono, torch.ones_like(d_mono)], dim=1)  # [N, 2]
    b = d_colmap  # [N]

    # 求解: (A^T A)^{-1} A^T b
    ATA = A.t() @ A  # [2, 2]
    ATb = A.t() @ b  # [2]

    try:
        params = torch.linalg.solve(ATA, ATb)  # [2]
        scale = params[0].item()
        shift = params[1].item()

        # 防止异常值
        if scale <= 0 or scale > 100 or abs(shift) > 100:
            print(f"[Warning] Abnormal alignment params: scale={scale:.4f}, shift={shift:.4f}, using identity")
            scale, shift = 1.0, 0.0
    except:
        print("[Warning] Linear system singular, using identity alignment")
        scale, shift = 1.0, 0.0

    # 对齐深度
    aligned_depth = scale * mono_depth + shift

    return aligned_depth, scale, shift


def project_colmap_points_to_view(colmap_points3D, camera):
    """
    将COLMAP 3D点投影到相机视角得到稀疏深度图

    DEPRECATED: 由于init.ply可能与COLMAP相机坐标系不匹配,
    改用 align_depth_cross_view 方法

    Args:
        colmap_points3D: [N, 3] numpy array, COLMAP稀疏点云（世界坐标）
        camera: Camera object

    Returns:
        sparse_depth: [H, W] torch.Tensor, 稀疏深度图（无点处为0）
    """
    # 返回空深度图,表示不使用此方法
    H, W = camera.image_height, camera.image_width
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.zeros((H, W), device=device)


def align_depth_cross_view(cameras, max_views=5):
    """
    跨视角深度对齐:使用不同视角间的深度一致性来估计全局尺度

    原理:
    对于两个重叠视角i和j,同一3D点在两个视角的深度应满足:
    depth_i = scale_i * mono_depth_i
    depth_j = scale_j * mono_depth_j

    通过多个视角的一致性约束,求解每个视角的scale

    Args:
        cameras: list of Camera objects with invdepthmap
        max_views: 最多使用多少个视角进行对齐

    Returns:
        dict: {image_name: (scale, shift)}
    """
    print(f"\n[Info] Cross-view depth alignment using {min(len(cameras), max_views)} views...")

    # 选择depth_reliable的相机
    reliable_cams = [cam for cam in cameras if cam.depth_reliable and cam.invdepthmap is not None][:max_views]

    if len(reliable_cams) < 2:
        print(f"[Warning] Too few reliable cameras ({len(reliable_cams)}), skip cross-view alignment")
        return {}

    # 简化方案:使用中位数深度作为全局参考
    median_depths = []
    for cam in reliable_cams:
        mono_invdepth = cam.invdepthmap.cuda() if hasattr(cam.invdepthmap, 'cuda') else cam.invdepthmap
        mono_depth = 1.0 / (mono_invdepth.squeeze() + 1e-6)

        # 使用mask有效区域的中位数深度
        if hasattr(cam, 'depth_mask') and cam.depth_mask is not None:
            mask = cam.depth_mask.cuda() if hasattr(cam.depth_mask, 'cuda') else cam.depth_mask
            valid = mono_depth[mask.squeeze() > 0.5]
        else:
            valid = mono_depth.flatten()

        if len(valid) > 0:
            median_depths.append(torch.median(valid).item())

    if len(median_depths) == 0:
        return {}

    # 全局参考深度(所有视角中位数的中位数)
    global_median = np.median(median_depths)

    # 为每个相机计算scale
    result = {}
    for cam in cameras:
        if not cam.depth_reliable or cam.invdepthmap is None:
            continue

        mono_invdepth = cam.invdepthmap.cuda() if hasattr(cam.invdepthmap, 'cuda') else cam.invdepthmap
        mono_depth = 1.0 / (mono_invdepth.squeeze() + 1e-6)

        if hasattr(cam, 'depth_mask') and cam.depth_mask is not None:
            mask = cam.depth_mask.cuda() if hasattr(cam.depth_mask, 'cuda') else cam.depth_mask
            valid = mono_depth[mask.squeeze() > 0.5]
        else:
            valid = mono_depth.flatten()

        if len(valid) > 0:
            cam_median = torch.median(valid).item()
            # scale使得对齐后的中位数深度等于全局参考
            scale = global_median / (cam_median + 1e-6)

            # 限制scale范围,避免异常值
            scale = max(0.1, min(10.0, scale))

            result[cam.image_name] = (scale, 0.0)

    print(f"[Info] Aligned {len(result)} cameras, global median depth: {global_median:.3f}")

    return result


def load_colmap_points3D(colmap_dir):
    """
    加载COLMAP稀疏点云
    支持 points3D.txt 和 points3D.ply 格式

    Args:
        colmap_dir: str, COLMAP sparse目录路径 (e.g., "data/sparse/0")

    Returns:
        points: [N, 3] numpy array, 3D点坐标
    """
    # 尝试txt格式
    points3D_txt = os.path.join(colmap_dir, "points3D.txt")
    if os.path.exists(points3D_txt):
        points = []
        with open(points3D_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or len(line) == 0:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
                    points.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if len(points) > 0:
            return np.array(points, dtype=np.float32)

    # 尝试ply格式
    points3D_ply = os.path.join(colmap_dir, "points3D.ply")
    if os.path.exists(points3D_ply):
        try:
            from plyfile import PlyData
            plydata = PlyData.read(points3D_ply)
            vertices = plydata['vertex']
            points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
            return points.astype(np.float32)
        except ImportError:
            print("[Warning] plyfile not installed, trying manual ply parsing")
            # 简单解析ply文件
            points = []
            with open(points3D_ply, 'r') as f:
                in_data = False
                for line in f:
                    if line.startswith('end_header'):
                        in_data = True
                        continue
                    if in_data:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])

            if len(points) > 0:
                return np.array(points, dtype=np.float32)

    # 尝试bin格式（需要读取cameras.bin, images.bin, points3D.bin）
    points3D_bin = os.path.join(colmap_dir, "points3D.bin")
    if os.path.exists(points3D_bin):
        print("[Info] Found points3D.bin, but binary format reading not implemented yet")
        print("[Info] Please convert to txt format using: colmap model_converter ...")

    print(f"[Error] Could not find COLMAP points3D file in {colmap_dir}")
    return None


def align_depth_median_scale(mono_depth, colmap_sparse_depth, mask):
    """
    使用中位数比率对齐深度（更鲁棒，适合outlier较多的情况）

    Args:
        mono_depth: [H, W] 单目深度预测
        colmap_sparse_depth: [H, W] COLMAP投影的稀疏深度
        mask: [H, W] 有效区域mask

    Returns:
        aligned_depth: [H, W] 对齐后的深度
        scale: float, 缩放因子
    """
    mono_depth = mono_depth.detach()
    colmap_sparse_depth = colmap_sparse_depth.detach()
    mask = mask.detach()

    valid = (colmap_sparse_depth > 0) * (mask > 0.5)

    if valid.sum() < 10:
        print(f"[Warning] Too few valid sparse points ({valid.sum()}), skip alignment")
        return mono_depth, 1.0

    d_mono = mono_depth[valid]
    d_colmap = colmap_sparse_depth[valid]

    # 计算深度比率的中位数
    ratios = d_colmap / (d_mono + 1e-6)
    scale = torch.median(ratios).item()

    if scale <= 0 or scale > 100:
        print(f"[Warning] Abnormal scale: {scale:.4f}, using 1.0")
        scale = 1.0

    aligned_depth = scale * mono_depth

    return aligned_depth, scale
