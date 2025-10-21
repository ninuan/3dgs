#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from configparser import Interpolation
from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
import os

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    # 如果图像文件不存在，创建一个虚拟图像（深度训练时不需要RGB）
    if os.path.exists(cam_info.image_path):
        image = Image.open(cam_info.image_path)
    else:
        # 从相机参数获取分辨率，创建空白图像
        from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text
        cameras_file = os.path.join(os.path.dirname(os.path.dirname(cam_info.image_path)), "sparse/0/cameras.bin")
        if not os.path.exists(cameras_file):
            cameras_file = os.path.join(os.path.dirname(os.path.dirname(cam_info.image_path)), "sparse/0/cameras.txt")

        try:
            if cameras_file.endswith('.bin'):
                cameras = read_intrinsics_binary(cameras_file)
            else:
                cameras = read_intrinsics_text(cameras_file)
            cam = list(cameras.values())[0]
            width, height = cam.width, cam.height
        except:
            width, height = 1920, 1080  # 默认分辨率

        print(f"[Info] RGB image not found: {cam_info.image_path}, using dummy image ({width}x{height})")
        image = Image.new('RGB', (width, height), color=(0, 0, 0))

    if cam_info.depth_path != "":
        try:
            ext = os.path.splitext(cam_info.depth_path)[1].lower()
            if ext == ".npy":
                arr = np.load(cam_info.depth_path)
                if arr is None:
                    print(f"[Warn] invdepth npy unreadable: '{cam_info.depth_path}'. Skip this view's depth.")
                    invdepthmap = None
                else:
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    invdepthmap = arr.astype(np.float32)
            else:
                raw = cv2.imread(cam_info.depth_path, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    print(f"[Warn] invdepth not found or unreadable: '{cam_info.depth_path}'. Skip this view's depth.")
                    invdepthmap = None
                else:
                    # 深度图存储的是真实深度值（单位：毫米）
                    # 需要转换为逆深度（单位：1/米）
                    depth_mm = raw.astype(np.float32)
                    depth_m = depth_mm / 1000.0  # 毫米转米

                    # 转换为逆深度（invdepth = 1 / depth）
                    # 注意：需要处理depth=0的情况
                    invdepthmap = np.zeros_like(depth_m)
                    valid_mask = depth_m > 0
                    invdepthmap[valid_mask] = 1.0 / depth_m[valid_mask]

                    if is_nerf_synthetic:
                        # NeRF synthetic数据集的特殊处理（如果需要）
                        pass
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            invdepthmap = None
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    depth_mask = None
    try:
        if getattr(args, "depth_mask_dir", "") != "":
            base = os.path.splitext(cam_info.image_name)[0]
            mask_path = os.path.join(args.source_path, args.depth_mask_dir, f"{base}.png")
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if m is not None:
                    if m.ndim == 3:
                        m = m[..., 0]
                    m = m.astype(np.float32)
                    if m.max() > 1.0:
                        m = m / 255.0 if m.max() <= 255.0 else m / 65535.0
                    m = cv2.resize(m, (resolution[0], resolution[1]),interpolation=cv2.INTER_NEAREST)
                    depth_mask = m[None]
    except Exception as e:
        print(f"[Warn] depth_mask load failed for {cam_info.image_name}: {e}")

    depth_only = getattr(args, "depth_only", False)

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, prcppoint=cam_info.prcppoint, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,depth_only=depth_only,depth_mask=depth_mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry