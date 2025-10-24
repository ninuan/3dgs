import torch
import numpy as np
from plyfile import PlyData

print("=" * 70)
print("Data2 Training Results - Fixed Isolation Pruning")
print("=" * 70)

# Read final point cloud
ply_path = "output/data2_fixed/point_cloud/iteration_30000/point_cloud.ply"
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

print(f"\nFinal Point Cloud Statistics:")
print(f"  Total points: {len(xyz)}")

# Compute point cloud geometry
center = xyz.mean(axis=0)
distances = np.linalg.norm(xyz - center, axis=1)
print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
print(f"  Max radius: {distances.max():.3f}")
print(f"  90% radius: {np.percentile(distances, 90):.3f}")
print(f"  Median radius: {np.median(distances):.3f}")

# Opacity statistics
opacity_sigmoid = 1 / (1 + np.exp(-opacity))
print(f"\nOpacity Statistics:")
print(f"  Mean opacity: {opacity_sigmoid.mean():.4f}")
print(f"  High opacity (>0.5): {(opacity_sigmoid > 0.5).sum()} ({(opacity_sigmoid > 0.5).sum()/len(xyz)*100:.1f}%)")
print(f"  Low opacity (<0.1): {(opacity_sigmoid < 0.1).sum()} ({(opacity_sigmoid < 0.1).sum()/len(xyz)*100:.1f}%)")

# Scale statistics
scale_activated = np.exp(scales)
max_scale = scale_activated.max(axis=1)
print(f"\nScale Statistics:")
print(f"  Mean max scale: {max_scale.mean():.4f}")
print(f"  90th percentile: {np.percentile(max_scale, 90):.4f}")

# Compare with initial point cloud
init_ply = PlyData.read("data2/sparse/0/points3D.ply")
init_xyz = np.stack([
    np.asarray(init_ply['vertex']['x']),
    np.asarray(init_ply['vertex']['y']),
    np.asarray(init_ply['vertex']['z'])
], axis=1)

init_center = init_xyz.mean(axis=0)
init_distances = np.linalg.norm(init_xyz - init_center, axis=1)

print(f"\nComparison with Initial Point Cloud:")
print(f"  Initial points: {len(init_xyz)}")
print(f"  Final points: {len(xyz)}")
print(f"  Point ratio: {len(xyz)/len(init_xyz):.2f}x")
print(f"  Initial radius: {init_distances.max():.3f}")
print(f"  Final radius: {distances.max():.3f}")
print(f"  Radius change: {(distances.max() - init_distances.max()) / init_distances.max() * 100:.1f}%")
print(f"  Center drift: {np.linalg.norm(center - init_center):.3f}")

if distances.max() > init_distances.max() * 2:
    print(f"\n⚠️  WARNING: Point cloud radius increased by >100% - possible divergence!")
else:
    print(f"\n✅ Point cloud radius is stable - no catastrophic divergence!")

