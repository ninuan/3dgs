import torch
import numpy as np
from plyfile import PlyData

# 读取训练后的点云
ply_path = "output/data_depth_only/point_cloud/iteration_30000/point_cloud.ply"

try:
    plydata = PlyData.read(ply_path)
    xyz = np.stack([
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ], axis=1)

    opacity = np.asarray(plydata.elements[0]["opacity"])
    scales = np.stack([
        np.asarray(plydata.elements[0]["scale_0"]),
        np.asarray(plydata.elements[0]["scale_1"]),
        np.asarray(plydata.elements[0]["scale_2"])
    ], axis=1)

    print("=" * 70)
    print("点云统计分析")
    print("=" * 70)

    print(f"\n总点数: {len(xyz)}")

    print(f"\n坐标范围:")
    print(f"  X: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}] (range: {xyz[:, 0].max() - xyz[:, 0].min():.3f})")
    print(f"  Y: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}] (range: {xyz[:, 1].max() - xyz[:, 1].min():.3f})")
    print(f"  Z: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}] (range: {xyz[:, 2].max() - xyz[:, 2].min():.3f})")

    # 计算点云的中心和半径
    center = xyz.mean(axis=0)
    distances = np.linalg.norm(xyz - center, axis=1)
    print(f"\n点云中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"点云半径: {distances.max():.3f}")
    print(f"90%点云半径: {np.percentile(distances, 90):.3f}")

    # Opacity统计
    opacity_sigmoid = 1 / (1 + np.exp(-opacity))
    print(f"\nOpacity统计:")
    print(f"  激活前: [{opacity.min():.3f}, {opacity.max():.3f}]")
    print(f"  激活后: [{opacity_sigmoid.min():.6f}, {opacity_sigmoid.max():.6f}]")
    print(f"  高opacity点 (>0.5): {(opacity_sigmoid > 0.5).sum()} ({(opacity_sigmoid > 0.5).sum()/len(xyz)*100:.1f}%)")
    print(f"  低opacity点 (<0.1): {(opacity_sigmoid < 0.1).sum()} ({(opacity_sigmoid < 0.1).sum()/len(xyz)*100:.1f}%)")

    # Scale统计
    scale_activated = np.exp(scales)
    max_scale = scale_activated.max(axis=1)
    print(f"\nScale统计:")
    print(f"  最大scale范围: [{max_scale.min():.6f}, {max_scale.max():.6f}]")
    print(f"  平均最大scale: {max_scale.mean():.6f}")
    print(f"  90百分位: {np.percentile(max_scale, 90):.6f}")
    print(f"  大scale点 (>0.1): {(max_scale > 0.1).sum()} ({(max_scale > 0.1).sum()/len(xyz)*100:.1f}%)")
    print(f"  大scale点 (>0.5): {(max_scale > 0.5).sum()} ({(max_scale > 0.5).sum()/len(xyz)*100:.1f}%)")

    # 比较初始点云
    init_ply = PlyData.read("data/sparse/0/points3D.ply")
    init_xyz = np.stack([
        np.asarray(init_ply['vertex']['x']),
        np.asarray(init_ply['vertex']['y']),
        np.asarray(init_ply['vertex']['z'])
    ], axis=1)

    init_center = init_xyz.mean(axis=0)
    init_distances = np.linalg.norm(init_xyz - init_center, axis=1)

    print(f"\n初始点云对比:")
    print(f"  初始点数: {len(init_xyz)}")
    print(f"  初始中心: [{init_center[0]:.3f}, {init_center[1]:.3f}, {init_center[2]:.3f}]")
    print(f"  初始半径: {init_distances.max():.3f}")
    print(f"  中心漂移: {np.linalg.norm(center - init_center):.3f}")
    print(f"  半径变化: {distances.max() - init_distances.max():.3f}")

    if distances.max() > init_distances.max() * 2:
        print(f"\n⚠️  警告: 点云半径扩大了 {distances.max() / init_distances.max():.1f}x，存在发散！")

except FileNotFoundError:
    print(f"文件未找到: {ply_path}")
    print("请先运行训练生成点云")
