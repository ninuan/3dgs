"""
将extern.txt (W2C格式) 转换为标准COLMAP images.txt格式

extern.txt格式：IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
  - qvec和tvec已经是W2C格式，不需要转换
"""
import numpy as np

# 读取extern.txt (已经是W2C格式)
cameras = []
input_file = 'data1/extern.txt'  # 修改这里指定输入文件
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or len(line) == 0:
            continue
        parts = line.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]

        # extern.txt中已经是W2C格式，直接使用
        qvec = np.array([qw, qx, qy, qz])
        tvec = np.array([tx, ty, tz])

        cameras.append((img_id, qvec, tvec, camera_id, name))

# 写入标准COLMAP格式的images.txt
output_file = 'data1/sparse/0/images.txt'
with open(output_file, 'w') as f:
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    f.write(f"# Number of images: {len(cameras)}\n")

    for img_id, qvec, tvec, camera_id, name in cameras:
        # Write pose line
        f.write(f"{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} ")
        f.write(f"{tvec[0]} {tvec[1]} {tvec[2]} {camera_id} {name}\n")
        # Write empty 2D points line
        f.write("\n")

print("✅ 已生成修复后的COLMAP格式 images.txt")
print(f"   输入文件: {input_file}")
print(f"   输出文件: {output_file}")
print(f"   包含 {len(cameras)} 个相机")
print()
print("⚠️  重要：extern.txt中的qvec和tvec已经是W2C格式，无需转换")
