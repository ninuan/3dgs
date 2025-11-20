#!/usr/bin/env python3
"""
诊断深度loss异常高的问题
检查GT depth和rendered depth的实际值
"""

import torch
import numpy as np
import cv2
from PIL import Image
import sys
import os

# 添加项目路径
sys.path.append('/home/wang/project/3dgs')

from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel

def diagnose_depth():
    """诊断深度loss计算"""

    # 加载场景
    parser = ArgumentParser(description="Depth Loss Diagnosis")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    args = parser.parse_args([
        "-s", "data_tum_cabinet",
        "-m", "output/tum_test_fixed",
        "--depths", "depth",
        "--depth_mask_dir", "mask"
    ])

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # 加载场景和gaussian
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # 随机选择几个视角
    train_cameras = scene.getTrainCameras()

    print("="*70)
    print("深度Loss诊断 - 检查GT深度和渲染深度的实际值")
    print("="*70)

    for i, viewpoint_cam in enumerate(train_cameras[:3]):  # 检查前3个视角
        if not viewpoint_cam.depth_reliable:
            continue

        print(f"\n视角 {i}: {viewpoint_cam.image_name}")

        # 渲染深度
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        invDepth_render = render_pkg["depth"]  # 渲染的逆深度

        # GT深度
        mono_invdepth = viewpoint_cam.invdepthmap.cuda()  # GT逆深度
        depth_mask = viewpoint_cam.depth_mask.cuda()

        # 转换为深度
        render_depth = 1.0 / (invDepth_render.squeeze(0) + 1e-6)  # [H, W]
        mono_depth = 1.0 / (mono_invdepth.squeeze(0) + 1e-6)  # [H, W]

        # 只在mask有效区域统计
        valid_mask = depth_mask.squeeze(0) > 0.5

        gt_depth_valid = mono_depth[valid_mask]
        render_depth_valid = render_depth[valid_mask]

        # 计算深度差异
        depth_diff = torch.abs(render_depth_valid - gt_depth_valid)

        print(f"  GT Depth:")
        print(f"    范围: [{gt_depth_valid.min().item():.3f}, {gt_depth_valid.max().item():.3f}] 米")
        print(f"    平均: {gt_depth_valid.mean().item():.3f} 米")

        print(f"  Rendered Depth:")
        print(f"    范围: [{render_depth_valid.min().item():.3f}, {render_depth_valid.max().item():.3f}] 米")
        print(f"    平均: {render_depth_valid.mean().item():.3f} 米")

        print(f"  Depth Difference:")
        print(f"    平均差异: {depth_diff.mean().item():.3f} 米")
        print(f"    最大差异: {depth_diff.max().item():.3f} 米")
        print(f"    L1 Loss (sum): {depth_diff.sum().item():.2f}")
        print(f"    有效像素数: {valid_mask.sum().item()}")

        # 计算如果应用base_multiplier=2.0后的loss
        loss_per_view = 2.0 * depth_diff.sum().item() / valid_mask.sum().item()
        loss_total = 2.0 * depth_diff.sum().item()

        print(f"  如果应用base_multiplier=2.0:")
        print(f"    Per-pixel loss: {loss_per_view:.6f}")
        print(f"    Total loss: {loss_total:.2f}")

    print("\n" + "="*70)
    print("诊断分析:")
    print("="*70)
    print("1. 检查GT depth范围是否合理 (室内应该2-6米)")
    print("2. 检查rendered depth是否与GT depth在相同尺度")
    print("3. 检查L1 Loss的数量级")
    print("   - 如果loss = 60,000+，每个像素平均贡献 ~0.35米")
    print("   - 对于室内场景，这个误差非常大")
    print("4. **关键问题**: depth_l1_weight可能没有被正确应用!")

if __name__ == "__main__":
    diagnose_depth()
