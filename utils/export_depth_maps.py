#!/usr/bin/env python3
"""
训练结束后导出每个视角的渲染深度图

用法：
    python utils/export_depth_maps.py -m output/data2_depth_only_v5 -s data2

功能：
    1. 加载训练好的模型
    2. 对每个训练视角渲染深度图
    3. 保存渲染深度图 + 原始深度图 + 差异图
    4. 生成可视化对比图
"""

import torch
import os
import sys
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams


def colorize_depth(depth, mask=None, vmin=None, vmax=None):
    """
    将深度图转换为彩色可视化图像

    Args:
        depth: numpy array [H, W]
        mask: numpy array [H, W], bool, 有效区域
        vmin, vmax: 深度范围

    Returns:
        RGB图像 [H, W, 3]，uint8
    """
    import matplotlib
    matplotlib.use('Agg')  # 不需要显示窗口
    import matplotlib.pyplot as plt

    # 只在有效区域计算范围
    if mask is not None:
        valid_depths = depth[mask]
    else:
        valid_depths = depth[depth > 0]

    # 归一化到[0, 1]
    if vmin is None:
        vmin = valid_depths.min() if len(valid_depths) > 0 else 0
    if vmax is None:
        vmax = valid_depths.max() if len(valid_depths) > 0 else 1

    depth_normalized = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0, 1)

    # 对无效区域使用特殊颜色（深紫色/黑色）
    if mask is not None:
        depth_normalized[~mask] = 0

    # 使用turbo colormap
    cmap = plt.get_cmap('turbo')
    colored = cmap(depth_normalized)

    # 转换为uint8 RGB
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return colored_rgb


def save_depth_comparison(rendered_depth, gt_depth, mask, save_path, view_name):
    """
    保存深度图：直接保存训练时使用的逆深度数据

    Args:
        rendered_depth: 渲染的**逆深度** [H, W]
        gt_depth: 真值**逆深度** [H, W]
        mask: 有效区域mask [H, W]
        save_path: 保存目录
        view_name: 视角名称
    """
    import cv2

    # 转换为numpy（保持逆深度格式，就像训练时一样）
    rendered_invdepth = rendered_depth.cpu().numpy()
    gt_invdepth = gt_depth.cpu().numpy()
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask

    # 保存原始逆深度数据为numpy文件
    np.save(os.path.join(save_path, f"{view_name}_rendered_invdepth.npy"), rendered_invdepth)
    np.save(os.path.join(save_path, f"{view_name}_gt_invdepth.npy"), gt_invdepth)
    np.save(os.path.join(save_path, f"{view_name}_mask.npy"), mask_np)

    # 为了可视化，创建归一化的灰度图（逆深度格式）
    # 只在mask有效区域归一化
    valid_mask = mask_np > 0.5

    # Rendered depth可视化
    if valid_mask.any() and rendered_invdepth[valid_mask].max() > 0:
        rendered_vis = rendered_invdepth.copy()
        vmin, vmax = np.percentile(rendered_invdepth[valid_mask], [1, 99])
        rendered_vis = np.clip((rendered_vis - vmin) / (vmax - vmin + 1e-6), 0, 1)
        rendered_vis[~valid_mask] = 0
        rendered_vis = (rendered_vis * 255).astype(np.uint8)
    else:
        rendered_vis = np.zeros_like(rendered_invdepth, dtype=np.uint8)

    # GT depth可视化
    if valid_mask.any() and gt_invdepth[valid_mask].max() > 0:
        gt_vis = gt_invdepth.copy()
        vmin, vmax = np.percentile(gt_invdepth[valid_mask], [1, 99])
        gt_vis = np.clip((gt_vis - vmin) / (vmax - vmin + 1e-6), 0, 1)
        gt_vis[~valid_mask] = 0
        gt_vis = (gt_vis * 255).astype(np.uint8)
    else:
        gt_vis = np.zeros_like(gt_invdepth, dtype=np.uint8)

    # 计算差异（逆深度空间）
    diff = np.abs(rendered_invdepth - gt_invdepth)
    diff[~valid_mask] = 0
    if valid_mask.any() and diff[valid_mask].max() > 0:
        diff_vis = np.clip(diff / (np.percentile(diff[valid_mask], 99) + 1e-6), 0, 1)
        diff_vis = (diff_vis * 255).astype(np.uint8)
    else:
        diff_vis = np.zeros_like(diff, dtype=np.uint8)

    # 保存可视化灰度图
    cv2.imwrite(os.path.join(save_path, f"{view_name}_rendered_invdepth.png"), rendered_vis)
    cv2.imwrite(os.path.join(save_path, f"{view_name}_gt_invdepth.png"), gt_vis)
    cv2.imwrite(os.path.join(save_path, f"{view_name}_diff.png"), diff_vis)

    # 创建对比图（横向拼接）
    h, w = rendered_invdepth.shape
    comparison = np.zeros((h, w * 3), dtype=np.uint8)
    comparison[:, :w] = rendered_vis
    comparison[:, w:2*w] = gt_vis
    comparison[:, 2*w:] = diff_vis

    # 添加标签
    comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison_rgb, "Rendered InvDepth", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison_rgb, "GT InvDepth", (w + 10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison_rgb, "Difference", (2*w + 10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(save_path, f"{view_name}_comparison.png"), comparison_rgb)

    # 计算统计信息（在逆深度空间）
    if valid_mask.any():
        mae = np.abs(rendered_invdepth[valid_mask] - gt_invdepth[valid_mask]).mean()
        rmse = np.sqrt(((rendered_invdepth[valid_mask] - gt_invdepth[valid_mask]) ** 2).mean())
    else:
        mae = 0.0
        rmse = 0.0

    return mae, rmse


def export_depth_maps(model_path, source_path, iteration=-1):
    """
    导出所有训练视角的深度图

    Args:
        model_path: 模型路径
        source_path: 数据集路径
        iteration: 要加载的迭代次数，-1表示最后一次
    """
    print("=" * 80)
    print("导出训练视角的渲染深度图")
    print("=" * 80)
    print(f"模型路径: {model_path}")
    print(f"数据集路径: {source_path}")
    print()

    # 创建输出目录
    output_dir = os.path.join(model_path, "depth_visualization")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    print()

    # 加载模型参数
    from argparse import Namespace

    # 创建命名空间参数
    args = Namespace(
        model_path=model_path,
        source_path=source_path,
        images='images',
        resolution=-1,
        white_background=False,
        data_device='cuda',
        eval=False,
        sh_degree=3,
        depths='depth',
        depth_mask_dir='mask',
        train_test_exp=False,
        undistorted=False,
        resolution_scales=[1.0],
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False
    )

    parser = ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)

    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)

    # 加载场景和高斯模型
    print("加载场景...")
    gaussians = GaussianModel(dataset.sh_degree, "default")
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # 加载训练好的模型
    if iteration == -1:
        # 找到最新的checkpoint
        checkpoints = [f for f in os.listdir(os.path.join(model_path, "point_cloud"))
                      if f.startswith("iteration_")]
        if checkpoints:
            iterations = [int(f.split("_")[1]) for f in checkpoints]
            iteration = max(iterations)

    print(f"加载迭代 {iteration} 的模型")
    gaussians.load_ply(os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))

    print(f"模型信息:")
    print(f"  点数: {gaussians.get_xyz.shape[0]}")
    print()

    # 设置渲染参数
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 获取训练相机
    train_cameras = scene.getTrainCameras()
    print(f"训练视角数量: {len(train_cameras)}")
    print()

    # 统计信息
    all_maes = []
    all_rmses = []

    # 渲染每个视角
    print("开始渲染...")
    print("-" * 80)

    with torch.no_grad():
        for idx, camera in enumerate(train_cameras):
            # 渲染
            render_pkg = render(camera, gaussians, pipeline, background)
            rendered_depth = render_pkg["depth"]  # [1, H, W] 逆深度

            # 获取GT深度和mask
            gt_invdepth = camera.invdepthmap.cuda()  # [1, H, W]
            depth_mask = camera.depth_mask.cuda()  # [1, H, W]

            # 保存对比图
            view_name = camera.image_name.replace('.png', '').replace('.jpg', '')
            mae, rmse = save_depth_comparison(
                rendered_depth.squeeze(0),
                gt_invdepth.squeeze(0),
                depth_mask.squeeze(0),
                output_dir,
                view_name
            )

            all_maes.append(mae)
            all_rmses.append(rmse)

            print(f"[{idx+1:2d}/{len(train_cameras)}] {view_name:20s} | "
                  f"MAE: {mae:.4f}m | RMSE: {rmse:.4f}m")

    print("-" * 80)
    print()

    # 输出统计信息
    print("=" * 80)
    print("统计结果")
    print("=" * 80)
    print(f"平均 MAE:  {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f} m")
    print(f"平均 RMSE: {np.mean(all_rmses):.4f} ± {np.std(all_rmses):.4f} m")
    print(f"最好视角 MAE:  {np.min(all_maes):.4f} m")
    print(f"最差视角 MAE:  {np.max(all_maes):.4f} m")
    print()
    print(f"✓ 所有深度图已保存到: {output_dir}")
    print("=" * 80)

    # 保存统计信息到文本文件
    stats_file = os.path.join(output_dir, "depth_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("Depth Rendering Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {source_path}\n")
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Number of points: {gaussians.get_xyz.shape[0]}\n")
        f.write(f"Number of views: {len(train_cameras)}\n\n")
        f.write("Per-view statistics:\n")
        f.write("-" * 80 + "\n")

        for idx, camera in enumerate(train_cameras):
            view_name = camera.image_name.replace('.png', '').replace('.jpg', '')
            f.write(f"{view_name:20s} | MAE: {all_maes[idx]:.4f}m | RMSE: {all_rmses[idx]:.4f}m\n")

        f.write("-" * 80 + "\n\n")
        f.write(f"Average MAE:  {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f} m\n")
        f.write(f"Average RMSE: {np.mean(all_rmses):.4f} ± {np.std(all_rmses):.4f} m\n")
        f.write(f"Best MAE:     {np.min(all_maes):.4f} m\n")
        f.write(f"Worst MAE:    {np.max(all_maes):.4f} m\n")

    print(f"✓ 统计信息已保存到: {stats_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="导出训练视角的渲染深度图")
    parser.add_argument("-m", "--model_path", type=str, required=True,
                       help="训练模型路径")
    parser.add_argument("-s", "--source_path", type=str, required=True,
                       help="数据集路径")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="要加载的迭代次数，-1表示最后一次 (默认: -1)")

    args = parser.parse_args()

    # 初始化CUDA
    safe_state(True)

    # 导出深度图
    export_depth_maps(args.model_path, args.source_path, args.iteration)
