#!/usr/bin/env python3
"""
Diagnostic script to check if points in the initial point cloud respect mask constraints.
"""

import numpy as np
from plyfile import PlyData
import cv2
import os
from pathlib import Path

# Configuration
DATA_DIR = "data3"
SPARSE_DIR = os.path.join(DATA_DIR, "sparse/0")
MASK_DIR = os.path.join(DATA_DIR, "mask")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")

# Camera intrinsics (depth camera)
fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5
width, height = 640, 480

def load_point_cloud(ply_path):
    """Load point cloud from PLY file."""
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return xyz

def project_point_to_image(point_3d, extrinsic):
    """
    Project 3D point to 2D image coordinates.

    Args:
        point_3d: (3,) array in world coordinates
        extrinsic: (4, 4) world-to-camera transform

    Returns:
        (u, v) pixel coordinates, or None if behind camera
    """
    # Transform to camera coordinates
    point_cam = extrinsic @ np.append(point_3d, 1.0)
    x, y, z = point_cam[:3]

    # Check if point is in front of camera
    if z <= 0:
        return None

    # Project to image
    u = fx * x / z + cx
    v = fy * y / z + cy

    return (u, v)

def load_camera_poses(images_txt_path):
    """Load camera poses from images.txt."""
    poses = {}

    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    # Skip header comments
    line_idx = 0
    while line_idx < len(lines) and lines[line_idx].startswith('#'):
        line_idx += 1

    # Parse camera poses (every 2 lines: pose info + points info)
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if not line:
            line_idx += 1
            continue

        parts = line.split()
        if len(parts) < 10:
            line_idx += 1
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        image_name = parts[9]

        # Convert quaternion to rotation matrix
        R = quat_to_rotation_matrix(qw, qx, qy, qz)

        # Build world-to-camera extrinsic matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = [tx, ty, tz]

        poses[image_name] = extrinsic

        # Skip next line (points line)
        line_idx += 2

    return poses

def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def main():
    print("=" * 80)
    print("MASK CONSTRAINT DIAGNOSTIC")
    print("=" * 80)

    # 1. Load initial point cloud
    print("\n[1] Loading initial point cloud...")
    ply_path = os.path.join(SPARSE_DIR, "points3D.ply")
    xyz = load_point_cloud(ply_path)
    print(f"    Total points: {len(xyz)}")
    print(f"    XYZ range:")
    print(f"      X: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
    print(f"      Y: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
    print(f"      Z: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")

    # 2. Load camera poses
    print("\n[2] Loading camera poses...")
    images_txt_path = os.path.join(SPARSE_DIR, "images.txt")
    poses = load_camera_poses(images_txt_path)
    print(f"    Loaded {len(poses)} camera poses")

    # 3. Load masks
    print("\n[3] Loading masks...")
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])
    print(f"    Found {len(mask_files)} mask files")

    # 4. Check a sample frame
    print("\n[4] Checking point-mask alignment for first frame...")
    first_mask_file = mask_files[0]
    mask_path = os.path.join(MASK_DIR, first_mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"    Mask file: {first_mask_file}")
    print(f"    Mask shape: {mask.shape}")
    print(f"    Mask coverage: {(mask > 0).sum()} / {mask.size} pixels ({100.0 * (mask > 0).sum() / mask.size:.1f}%)")

    # Find corresponding image
    frame_id = first_mask_file.replace('.png', '')
    corresponding_image = f"{frame_id}.png"

    if corresponding_image in poses:
        print(f"    Found corresponding pose for {corresponding_image}")
        extrinsic = poses[corresponding_image]

        # Project all points to this view
        points_in_mask = 0
        points_out_mask = 0
        points_behind_camera = 0
        points_out_of_bounds = 0

        for point in xyz:
            uv = project_point_to_image(point, extrinsic)

            if uv is None:
                points_behind_camera += 1
                continue

            u, v = uv

            # Check if in image bounds
            if u < 0 or u >= width or v < 0 or v >= height:
                points_out_of_bounds += 1
                continue

            # Check mask value
            u_int, v_int = int(u), int(v)
            if mask[v_int, u_int] > 0:
                points_in_mask += 1
            else:
                points_out_mask += 1

        print(f"\n    Projection results:")
        print(f"      Points in mask region: {points_in_mask}")
        print(f"      Points OUTSIDE mask region: {points_out_mask}")
        print(f"      Points behind camera: {points_behind_camera}")
        print(f"      Points out of image bounds: {points_out_of_bounds}")

        if points_out_mask > 0:
            print(f"\n    ⚠️  WARNING: {points_out_mask} points are OUTSIDE the mask region!")
            print(f"               This means the initial point cloud contains points")
            print(f"               that should have been filtered during generation.")
    else:
        print(f"    ⚠️  Could not find pose for {corresponding_image}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
