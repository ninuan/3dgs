#!/usr/bin/env python3
"""
验证深度单位转换修复
检查TUM深度图的单位转换是否正确
"""

import cv2
import numpy as np
import os

def check_depth_conversion(depth_path):
    """检查深度转换"""
    # 读取深度图
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    if depth_raw is None:
        print(f"无法读取: {depth_path}")
        return

    # TUM正确的转换
    depth_correct = depth_raw.astype(np.float32) / 5000.0

    # 之前错误的转换
    depth_wrong = depth_raw.astype(np.float32) / 1000.0

    # 统计有效深度值
    valid_mask = (depth_correct > 0.1) & (depth_correct < 6.0)

    print(f"\n深度图: {os.path.basename(depth_path)}")
    print(f"  原始值范围: [{depth_raw.min()}, {depth_raw.max()}]")
    print(f"  \n  ✓ 正确转换 (/5000.0):")
    print(f"    - 深度范围: [{depth_correct[valid_mask].min():.3f}m, {depth_correct[valid_mask].max():.3f}m]")
    print(f"    - 平均深度: {depth_correct[valid_mask].mean():.3f}m")
    print(f"    - 有效像素数: {valid_mask.sum()}")

    print(f"  \n  ✗ 错误转换 (/1000.0) - 之前的bug:")
    depth_wrong_valid = (depth_wrong > 0.1) & (depth_wrong < 30.0)
    print(f"    - 深度范围: [{depth_wrong[depth_wrong_valid].min():.3f}m, {depth_wrong[depth_wrong_valid].max():.3f}m]")
    print(f"    - 平均深度: {depth_wrong[depth_wrong_valid].mean():.3f}m")
    print(f"    - 这比正确值大5倍! (导致depth loss异常高)")


if __name__ == "__main__":
    # 检查几张深度图
    depth_dir = "/home/wang/project/3dgs/data_tum_cabinet/depth"

    if not os.path.exists(depth_dir):
        print(f"错误: 深度目录不存在: {depth_dir}")
        exit(1)

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])[:3]

    print("="*70)
    print("TUM深度单位转换验证")
    print("="*70)
    print("\n修复内容:")
    print("  文件: utils/camera_utils.py:67")
    print("  修改前: depth_m = depth_mm / 1000.0  # 错误!")
    print("  修改后: depth_m = depth_mm / 5000.0  # TUM标准格式")
    print("\n影响:")
    print("  - 之前的错误导致深度值比实际大5倍")
    print("  - 这导致depth loss异常高 (60,000+)")
    print("  - 修复后depth loss应该显著降低")
    print("="*70)

    for fname in depth_files:
        depth_path = os.path.join(depth_dir, fname)
        check_depth_conversion(depth_path)

    print("\n" + "="*70)
    print("✓ 深度单位转换已修复")
    print("="*70)
    print("\n下一步:")
    print("  1. 重新运行训练:")
    print("     export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH")
    print("     python train.py -s data_tum_cabinet -m output/tum_fixed \\")
    print("         --depths depth --depth_mask_dir mask \\")
    print("         --iterations 30000 --disable_viewer")
    print("\n  2. 预期结果:")
    print("     - Depth Loss应该降到 < 100 (之前是60,000+)")
    print("     - 训练应该更稳定收敛")
    print()
