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

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def depth2normal(depth, mask, camera):
    '''
    从深度图计算法向
    '''
    camD = depth.permute([1,2,0])
    mask = mask.permute([1,2,0])
    shape = camD.shape
    device = camD.device

    h, w, _ = torch.meshgrid(
        torch.arange(0, shape[0]),
        torch.arange(0, shape[1]),
        torch.arange(0, shape[2]),
        indexing='ij'
    )
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis = -1)

    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD

    from utils.graphics_utils import fov2focal
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([[K00, 0], [0, K11]]).to(device)
    Kinv = torch.inverse(K)

    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    p = torch.nn.functional.pad(camPos[None], [0,0,1,1,1,1],mode="replicate")
    mask = torch.nn.functional.pad(mask[None].to(torch.float32), [0,0,1,1,1,1],mode="replicate").to(torch.bool)

    p_c = (p[:, 1:-1, 1:-1, :]) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:,  :-2, 1:-1, :] - p_c) * mask[:,  :-2, 1:-1, :]  # 上
    p_l = (p[:, 1:-1,  :-2, :] - p_c) * mask[:, 1:-1,  :-2, :]  # 左
    p_b = (p[:, 2:  , 1:-1, :] - p_c) * mask[:, 2:  , 1:-1, :]  # 下
    p_r = (p[:, 1:-1, 2:  , :] - p_c) * mask[:, 1:-1, 2:  , :]  # 右

    n_ul = torch.cross(p_u, p_l, dim=-1)
    n_ur = torch.cross(p_r,p_u,dim=-1)
    n_br = torch.cross(p_b,p_r,dim=-1)
    n_bl = torch.cross(p_l,p_b,dim=-1)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]

    mask = mask[0, 1:-1, 1:-1, :]
    n = torch.nn.functional.normalize(n, dim=-1)
    n = (n * mask).permute([2,0,1])

    return n