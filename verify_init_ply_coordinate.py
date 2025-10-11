"""
问题分析：
extern.txt 中 T = 相机中心C，而标准COLMAP中 T = tvec = -R*C
我们已经转换了 images.txt 为标准格式，但是 init.ply 仍然在原坐标系中

关键洞察：
如果 extern.txt 用的是相机中心C，那么init.ply很可能也在同一个坐标系中！
我们只需要验证：用相机中心C的公式投影init.ply，是否能看到点云？
"""
import numpy as np
from plyfile import PlyData
import cv2

def quat_to_matrix(qw, qx, qy, qz):
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

# 读取init.ply
ply = PlyData.read('data/init.ply')
xyz = np.stack([
    np.asarray(ply.elements[0]['x']),
    np.asarray(ply.elements[0]['y']),
    np.asarray(ply.elements[0]['z'])
], axis=1)

K = np.loadtxt('data/intrinsic_ir.txt')

# 读取extern.txt（原始格式，T=相机中心）
cameras = []
with open('data/extern.txt', 'r') as f:
    for line in f:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        parts = line.split()
        name = parts[-1]
        qw, qx, qy, qz = map(float, parts[1:5])
        cx, cy, cz = map(float, parts[5:8])
        cameras.append((name, qw, qx, qy, qz, cx, cy, cz))

print('使用相机中心C的公式投影init.ply:\n')
print(f"{'相机名':<15} {'在前方':<8} {'在图像内':<10} {'在mask内':<10} {'V范围':<20}")
print('='*70)

total_in_mask = 0
for name, qw, qx, qy, qz, cx, cy, cz in cameras:
    R = quat_to_matrix(qw, qx, qy, qz)
    C = np.array([cx, cy, cz])

    # T是相机中心：X_cam = R * (X_world - C)
    RT = np.hstack([R, (-R @ C).reshape(3, 1)])

    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyz_cam = RT @ xyz_h.T

    in_front = (xyz_cam[2, :] > 0).sum()

    if in_front > 0:
        xyz_img = K @ xyz_cam
        uv = xyz_img[:2, :] / xyz_img[2, :]
        in_image = ((uv[0, :] >= 0) & (uv[0, :] < 512) & (uv[1, :] >= 0) & (uv[1, :] < 424)).sum()

        if in_image > 0:
            in_img_mask = (uv[0, :] >= 0) & (uv[0, :] < 512) & (uv[1, :] >= 0) & (uv[1, :] < 424)
            uv_valid = uv[:, in_img_mask].astype(int)
            v_range = f'[{uv_valid[1, :].min()}, {uv_valid[1, :].max()}]'

            mask = cv2.imread(f'data/mask/{name}', cv2.IMREAD_GRAYSCALE)
            in_mask = sum(1 for i in range(uv_valid.shape[1]) if mask[uv_valid[1, i], uv_valid[0, i]] > 128)
            total_in_mask += in_mask
        else:
            in_mask = 0
            v_range = 'N/A'
    else:
        in_image = 0
        in_mask = 0
        v_range = 'N/A'

    print(f'{name:<15} {in_front:<8} {in_image:<10} {in_mask:<10} {v_range:<20}')

print(f'\n总结: 总共 {total_in_mask} 个点在某个相机的mask内')

if total_in_mask > 10000:
    print('\n✅ 结论：init.ply已经在正确的坐标系中！')
    print('   extern.txt的T确实是相机中心C')
    print('   我们应该直接使用init.ply，但要把images.txt转换回相机中心格式')
else:
    print('\n❌ 结论：init.ply与extern.txt不匹配')
