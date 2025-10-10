#!/usr/bin/env python3
"""
处理data目录中的COLMAP数据，生成完整的数据集结构
"""
import os
import sys
import struct
import numpy as np
from pathlib import Path
from plyfile import PlyData

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_text

def convert_cameras_bin_to_txt(bin_path, txt_path):
    """将cameras.bin转换为cameras.txt"""
    cameras = read_intrinsics_binary(bin_path)

    with open(txt_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")

        for camera_id, cam in cameras.items():
            params_str = ' '.join(map(str, cam.params))
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params_str}\n")

    print(f"✓ 生成 cameras.txt: {len(cameras)} 个相机")

def convert_ply_to_points3d_bin(ply_path, bin_path):
    """将point.ply转换为points3D.bin"""
    # 读取PLY文件
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']

    num_points = len(vertices)
    print(f"读取 {num_points} 个3D点")

    with open(bin_path, 'wb') as f:
        # 写入点的数量
        f.write(struct.pack('Q', num_points))

        for i in range(num_points):
            # POINT3D_ID
            point_id = i + 1
            f.write(struct.pack('Q', point_id))

            # XYZ
            x = float(vertices['x'][i])
            y = float(vertices['y'][i])
            z = float(vertices['z'][i])
            f.write(struct.pack('ddd', x, y, z))

            # RGB
            r = int(vertices['red'][i]) if 'red' in vertices else 128
            g = int(vertices['green'][i]) if 'green' in vertices else 128
            b = int(vertices['blue'][i]) if 'blue' in vertices else 128
            f.write(struct.pack('BBB', r, g, b))

            # ERROR
            error = 1.0
            f.write(struct.pack('d', error))

            # TRACK_LENGTH
            track_length = 0
            f.write(struct.pack('Q', track_length))

    print(f"✓ 生成 points3D.bin: {num_points} 个点")

def create_images_directory(data_path, images_txt_path):
    """根据images.txt创建images目录结构"""
    # 读取images.txt获取需要的图像列表
    image_names = []
    with open(images_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 10:
                    image_name = parts[9]  # 最后一列是图像名称
                    image_names.append(image_name)

    print(f"\n需要的图像文件: {image_names}")

    # 创建images目录
    images_dir = os.path.join(data_path, 'images')
    os.makedirs(images_dir, exist_ok=True)

    return image_names

def main():
    # 设置路径
    data_path = '/home/wang/project/gaussian-splatting-gai/data'
    sparse_path = os.path.join(data_path, 'sparse/0')

    print("=" * 60)
    print("处理COLMAP数据集")
    print("=" * 60)

    # 1. 转换cameras.bin到cameras.txt
    print("\n1. 转换相机参数文件...")
    cameras_bin = os.path.join(sparse_path, 'cameras.bin')
    cameras_txt = os.path.join(sparse_path, 'cameras.txt')

    if os.path.exists(cameras_bin):
        convert_cameras_bin_to_txt(cameras_bin, cameras_txt)
    else:
        print(f"✗ 找不到 cameras.bin")

    # 2. 检查images.txt
    print("\n2. 检查图像列表...")
    images_txt = os.path.join(sparse_path, 'images.txt')
    if os.path.exists(images_txt):
        image_names = create_images_directory(data_path, images_txt)
        print(f"✓ images.txt 存在，包含 {len(image_names)} 张图像")
    else:
        print(f"✗ 找不到 images.txt")

    # 3. 转换point.ply到points3D.bin
    print("\n3. 转换3D点云文件...")
    ply_path = os.path.join(sparse_path, 'point.ply')
    points3d_bin = os.path.join(sparse_path, 'points3D.bin')

    if os.path.exists(ply_path):
        convert_ply_to_points3d_bin(ply_path, points3d_bin)
    else:
        print(f"✗ 找不到 point.ply")

    # 4. 显示最终的目录结构
    print("\n" + "=" * 60)
    print("完成！期望的数据集结构:")
    print("=" * 60)
    print(f"""
{data_path}/
├── images/
│   ├── {image_names[0] if image_names else '*.png'}
│   ├── {image_names[1] if len(image_names) > 1 else '...'}
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin      (原始文件)
        ├── cameras.txt      (✓ 已生成)
        ├── images.txt       (原始文件)
        ├── points3D.bin     (✓ 已生成)
        └── point.ply        (原始文件)

⚠️  注意: 您需要将对应的图像文件放入 images/ 目录:
    """)
    for img in image_names:
        print(f"    - {img}")

if __name__ == '__main__':
    main()
