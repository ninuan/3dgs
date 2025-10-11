"""
方案：恢复images.txt为相机中心格式（与extern.txt一致）
然后修改代码以支持这种格式
"""
import numpy as np
import shutil

# 1. 备份当前的标准格式images.txt
shutil.copy('data/sparse/0/images.txt', 'data/sparse/0/images_standard_colmap.txt.bak')

# 2. 恢复为相机中心格式（直接复制extern.txt）
shutil.copy('data/extern.txt', 'data/sparse/0/images.txt')

# 3. 恢复init.ply为points3D.ply
shutil.copy('data/init.ply', 'data/sparse/0/points3D_init_backup.ply')

# 复制init.ply但添加法向量
from plyfile import PlyData, PlyElement

ply = PlyData.read('data/init.ply')
xyz = np.stack([
    np.asarray(ply.elements[0]['x']),
    np.asarray(ply.elements[0]['y']),
    np.asarray(ply.elements[0]['z'])
], axis=1)
rgb = np.stack([
    np.asarray(ply.elements[0]['red']),
    np.asarray(ply.elements[0]['green']),
    np.asarray(ply.elements[0]['blue'])
], axis=1)

normals = np.zeros_like(xyz)

dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
         ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

elements = np.empty(xyz.shape[0], dtype=dtype)
elements['x'] = xyz[:, 0].astype('f4')
elements['y'] = xyz[:, 1].astype('f4')
elements['z'] = xyz[:, 2].astype('f4')
elements['nx'] = normals[:, 0].astype('f4')
elements['ny'] = normals[:, 1].astype('f4')
elements['nz'] = normals[:, 2].astype('f4')
elements['red'] = rgb[:, 0]
elements['green'] = rgb[:, 1]
elements['blue'] = rgb[:, 2]

vertex_element = PlyElement.describe(elements, 'vertex')
ply_data = PlyData([vertex_element])
ply_data.write('data/sparse/0/points3D.ply')

print("✅ 已恢复原始数据格式（相机中心格式）:")
print("   - images.txt = extern.txt (T是相机中心)")
print("   - points3D.ply = init.ply (带法向量)")
print("\n下一步：修改代码以支持相机中心格式的T向量")
