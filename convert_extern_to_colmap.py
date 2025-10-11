"""
将非标准格式的extern.txt转换为标准COLMAP images.txt格式
"""
import numpy as np
from scene.colmap_loader import qvec2rotmat

# 读取extern.txt (T是相机中心)
cameras = []
with open('data/extern.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or len(line) == 0:
            continue
        parts = line.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        cx, cy, cz = map(float, parts[5:8])  # Camera center
        camera_id = int(parts[8])
        name = parts[9]

        # 转换为标准COLMAP格式: T = -R * C
        qvec = np.array([qw, qx, qy, qz])
        R = qvec2rotmat(qvec)
        C = np.array([cx, cy, cz])
        tvec = -R @ C

        cameras.append((img_id, qvec, tvec, camera_id, name))

# 写入标准COLMAP格式的images.txt
with open('data/sparse/0/images.txt', 'w') as f:
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

print("✅ 已生成标准COLMAP格式的 data/sparse/0/images.txt")
print(f"   包含 {len(cameras)} 个相机")
