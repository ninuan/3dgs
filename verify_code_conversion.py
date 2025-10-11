"""
验证代码转换是否正确：读取相机并投影init.ply
"""
import numpy as np
from plyfile import PlyData
import cv2
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat

# 读取init.ply
ply = PlyData.read('data/sparse/0/points3D.ply')
xyz = np.stack([
    np.asarray(ply.elements[0]['x']),
    np.asarray(ply.elements[0]['y']),
    np.asarray(ply.elements[0]['z'])
], axis=1)

# 读取相机参数
cam_extrinsics = read_extrinsics_text('data/sparse/0/images.txt')
cam_intrinsics = read_intrinsics_text('data/sparse/0/cameras.txt')
cam_intr = list(cam_intrinsics.values())[0]
fx, fy, cx, cy = cam_intr.params
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

print("测试代码中的转换逻辑:\n")
print(f"{'相机名':<15} {'在前方':<8} {'在图像内':<10} {'在mask内':<10}")
print('='*60)

total_in_mask = 0
for img_id, extr in cam_extrinsics.items():
    # 模拟dataset_readers.py中的转换
    R_transposed = np.transpose(qvec2rotmat(extr.qvec))  # R = R_original^T
    C = np.array(extr.tvec)  # 相机中心

    USE_CAMERA_CENTER_FORMAT = True
    if USE_CAMERA_CENTER_FORMAT:
        R_original = qvec2rotmat(extr.qvec)
        T = -R_original @ C  # 标准tvec
    else:
        T = C

    # 现在构建world-to-camera矩阵
    # getWorld2View2(R_transposed, T) 会构建:
    # Rt = [R_transposed^T | T] = [R_original | T]
    #
    # 等等，这里有问题！getWorld2View2期望R是R^T，然后它再次转置
    # 让我重新看看getWorld2View2的代码...

    # 实际上getWorld2View2做的是：
    # Rt[:3,:3] = R.transpose()  # 如果输入是R^T，这里变成R
    # Rt[:3,3] = t
    # 所以最终Rt = [R | t]

    # 标准COLMAP: X_cam = R * X_world + T
    # 所以直接用R和T构建矩阵
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R_original
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0

    # 投影
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyz_cam = Rt @ xyz_h.T

    in_front = (xyz_cam[2, :] > 0).sum()

    if in_front > 0:
        xyz_img = K @ xyz_cam[:3, :]
        uv = xyz_img[:2, :] / xyz_img[2, :]
        in_image = ((uv[0, :] >= 0) & (uv[0, :] < 512) & (uv[1, :] >= 0) & (uv[1, :] < 424)).sum()

        if in_image > 0:
            in_img_mask = (uv[0, :] >= 0) & (uv[0, :] < 512) & (uv[1, :] >= 0) & (uv[1, :] < 424)
            uv_valid = uv[:, in_img_mask].astype(int)

            mask = cv2.imread(f'data/mask/{extr.name}', cv2.IMREAD_GRAYSCALE)
            in_mask = sum(1 for i in range(uv_valid.shape[1]) if mask[uv_valid[1, i], uv_valid[0, i]] > 128)
            total_in_mask += in_mask
        else:
            in_mask = 0
    else:
        in_image = 0
        in_mask = 0

    print(f'{extr.name:<15} {in_front:<8} {in_image:<10} {in_mask:<10}')

print(f'\n总结: 总共 {total_in_mask} 个点在某个相机的mask内')

if total_in_mask > 10000:
    print('✅ 转换正确！')
else:
    print('❌ 转换仍有问题')
