#!/usr/bin/env python3
"""
验证data目录中的数据集是否完整，用于深度图和mask训练
"""
import os
import sys
import numpy as np
from PIL import Image

def verify_data_directory(data_path):
    """验证数据集完整性"""
    print("=" * 70)
    print("验证深度图+Mask训练数据集")
    print("=" * 70)

    errors = []
    warnings = []

    # 1. 检查sparse目录
    sparse_path = os.path.join(data_path, 'sparse/0')
    if not os.path.exists(sparse_path):
        errors.append(f"缺少 sparse/0 目录")
        return errors, warnings

    required_files = {
        'cameras.bin': '相机参数（二进制）',
        'cameras.txt': '相机参数（文本）',
        'images.txt': '图像位姿信息',
        'points3D.bin': '3D点云（二进制）',
        'point.ply': '3D点云（PLY格式）'
    }

    print("\n1️⃣  检查COLMAP文件:")
    for filename, desc in required_files.items():
        filepath = os.path.join(sparse_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024
            print(f"   ✅ {filename:20s} - {desc:20s} ({size:.1f} KB)")
        else:
            print(f"   ❌ {filename:20s} - 缺失")
            errors.append(f"缺少文件: {filename}")

    # 2. 读取images.txt获取图像列表
    images_txt = os.path.join(sparse_path, 'images.txt')
    image_names = []

    if os.path.exists(images_txt):
        with open(images_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 10:
                        image_names.append(parts[9])

    print(f"\n2️⃣  从images.txt读取到 {len(image_names)} 张图像:")
    for name in image_names:
        print(f"   - {name}")

    # 3. 检查depth目录
    print("\n3️⃣  检查深度图 (depth/):")
    depth_dir = os.path.join(data_path, 'depth')

    if not os.path.exists(depth_dir):
        errors.append("缺少 depth/ 目录")
        print(f"   ❌ depth/ 目录不存在")
    else:
        depth_files = set(os.listdir(depth_dir))
        for img_name in image_names:
            if img_name in depth_files:
                depth_path = os.path.join(depth_dir, img_name)
                size = os.path.getsize(depth_path) / 1024
                print(f"   ✅ {img_name:20s} ({size:.1f} KB)")
            else:
                print(f"   ❌ {img_name:20s} - 缺失")
                errors.append(f"缺少深度图: {img_name}")

    # 4. 检查mask目录
    print("\n4️⃣  检查Mask (mask/):")
    mask_dir = os.path.join(data_path, 'mask')

    if not os.path.exists(mask_dir):
        warnings.append("缺少 mask/ 目录（如需使用mask，请添加）")
        print(f"   ⚠️  mask/ 目录不存在（可选）")
    else:
        mask_files = set(os.listdir(mask_dir))
        for img_name in image_names:
            if img_name in mask_files:
                mask_path = os.path.join(mask_dir, img_name)
                size = os.path.getsize(mask_path) / 1024
                print(f"   ✅ {img_name:20s} ({size:.1f} KB)")
            else:
                print(f"   ⚠️  {img_name:20s} - 缺失（可选）")
                warnings.append(f"缺少mask: {img_name}")

    # 5. 检查images目录（可选，因为您使用depth训练）
    print("\n5️⃣  检查RGB图像 (images/):")
    images_dir = os.path.join(data_path, 'images')

    if not os.path.exists(images_dir):
        warnings.append("缺少 images/ 目录（如果仅使用深度图训练，此目录可选）")
        print(f"   ⚠️  images/ 目录不存在（深度训练时可选）")
    else:
        rgb_files = set(os.listdir(images_dir))
        if len(rgb_files) == 0:
            print(f"   ⚠️  images/ 目录为空（深度训练时可选）")
        else:
            for img_name in image_names:
                if img_name in rgb_files:
                    img_path = os.path.join(images_dir, img_name)
                    size = os.path.getsize(img_path) / 1024
                    print(f"   ✅ {img_name:20s} ({size:.1f} KB)")
                else:
                    print(f"   ⚠️  {img_name:20s} - 缺失（可选）")

    return errors, warnings, image_names

def main():
    data_path = '/home/wang/project/gaussian-splatting-gai/data'

    errors, warnings, image_names = verify_data_directory(data_path)

    # 总结
    print("\n" + "=" * 70)
    print("验证结果:")
    print("=" * 70)

    if errors:
        print(f"\n❌ 发现 {len(errors)} 个错误:")
        for err in errors:
            print(f"   - {err}")
    else:
        print("\n✅ 所有必需文件检查通过!")

    if warnings:
        print(f"\n⚠️  {len(warnings)} 个警告:")
        for warn in warnings:
            print(f"   - {warn}")

    # 数据集摘要
    print("\n" + "=" * 70)
    print("数据集摘要:")
    print("=" * 70)
    print(f"📊 视图数量: {len(image_names)}")
    print(f"📁 数据路径: {data_path}")

    print("\n📋 推荐的训练命令:")
    print("=" * 70)
    print(f"# 基础训练（使用深度图）")
    print(f"python train.py -s data/ -d depth")
    print(f"\n# 使用深度图 + mask")
    print(f"python train.py -s data/ -d depth --depth_mask_dir mask")
    print(f"\n# 完整训练命令示例")
    print(f"python train.py -s data/ -d depth --depth_mask_dir mask -m output/my_scene")

    print("\n" + "=" * 70)

    return len(errors) == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
