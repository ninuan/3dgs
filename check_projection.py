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

# 读取点云
ply = PlyData.read('data/sparse/0/points3D.ply')
xyz = np.stack([
    np.asarray(ply.elements[0]['x']),
    np.asarray(ply.elements[0]['y']),
    np.asarray(ply.elements[0]['z'])
], axis=1)

K = np.loadtxt('data/intrinsic_ir.txt')

# 所有相机
cameras = [
    ('000009.png', -0.36195661924278694, -0.5691579652359762, 0.6296720645271051, -0.3854344400140554, 0.5667832494575705, 0.14080599555364104, -0.3085890812347516),
    ('000015.png', -0.36553145265845416, -0.5749990191209956, 0.6243427153793635, -0.3820458858305068, 1.1094795073127535, 0.22884040913437131, -0.5088217962205973),
    ('000174.png', 0.3476886617610083, 0.5702027624446022, 0.6287260992975623, -0.39835272842727637, 0.9865273633394068, -1.8813561782233548, 3.7915078845473706),
    ('000194.png', 0.3162594180636008, 0.5205349309506726, 0.6704260465536539, -0.42373609977046905, 0.5132390385479935, -1.899836453118437, 3.8474652212070897),
    ('000291.png', 0.422444091790374, 0.6682997402308855, -0.5232620332536221, 0.31798316162026935, -0.15047240351895766, 0.3668881625098841, -0.7410158094352969),
    ('000299.png', 0.4327915231074722, 0.6853419887079405, -0.5007335187015434, 0.30374956673044096, -0.8261591143472012, 0.5034171148856903, -0.992756581300161),
]

print('检查所有相机的投影情况:\n')
print(f"{'相机名':<15} {'在前方':<8} {'在图像内':<10} {'在mask内':<10} {'V范围':<20}")
print('='*70)

total_in_mask = 0
for name, qw, qx, qy, qz, tx, ty, tz in cameras:
    R = quat_to_matrix(qw, qx, qy, qz)
    T = np.array([tx, ty, tz])

    # T is camera center (non-standard!)
    # X_cam = R * (X_world - T) = R * X_world - R * T
    RT = np.hstack([R, (-R @ T).reshape(3, 1)])

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

if total_in_mask == 0:
    print('\n❌ 问题：点云需要向下平移（Y方向）才能与mask对齐')
    print('建议：将点云Y坐标整体增加约0.2-0.3')
