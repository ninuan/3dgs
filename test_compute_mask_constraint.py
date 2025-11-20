#!/usr/bin/env python3
"""
直接测试compute_mask_constraint，看它是否能正确识别mask外的点
"""
import torch
import sys
import os

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 加载必要的模块
from scene import Scene, GaussianModel
from argparse import Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from utils.mask_utils import compute_mask_constraint

print("="*80)
print("TESTING compute_mask_constraint FUNCTION")
print("="*80)

# 创建参数
class Args:
    def __init__(self):
        self.source_path = "data3"
        self.model_path = "output/data3"
        self.images = "images"
        self.depths = "depth"
        self.depth_mask_dir = "mask"
        self.resolution = -1
        self.data_device = "cuda"
        self.eval = False
        self.white_background = False
        self.sh_degree = 3
        self.train_test_exp = False

args = Args()

# 加载场景和高斯模型
print("\n[1/3] Loading scene and gaussians...")
gaussians = GaussianModel(args.sh_degree)
scene = Scene(args, gaussians, load_iteration=30000, shuffle=False)

print(f"  Loaded {gaussians.get_xyz.shape[0]} gaussians")

# 测试compute_mask_constraint
print("\n[2/3] Computing mask constraint...")

# 创建简单的pipe对象
class SimplePipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

pipe = SimplePipe()
background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

valid_region_mask = compute_mask_constraint(gaussians, scene, render, pipe, background)

if valid_region_mask is not None:
    num_valid = valid_region_mask.sum().item()
    num_invalid = (~valid_region_mask).sum().item()
    total = valid_region_mask.shape[0]

    print(f"\n[3/3] Mask constraint results:")
    print(f"  Total points: {total}")
    print(f"  Valid (in mask): {num_valid} ({100.0*num_valid/total:.1f}%)")
    print(f"  Invalid (outside mask): {num_invalid} ({100.0*num_invalid/total:.1f}%)")

    if num_invalid > 10000:
        print(f"\n⚠️  CRITICAL: {num_invalid} points should be pruned!")
        print(f"    This means mask constraint IS working correctly,")
        print(f"    but the pruning is NOT being applied during training!")
    elif num_invalid < 1000:
        print(f"\n✓ Mask constraint seems too strict or already clean")
else:
    print("\n❌ ERROR: compute_mask_constraint returned None!")
    print("   This means it's not being executed at all!")

print("\n" + "="*80)
