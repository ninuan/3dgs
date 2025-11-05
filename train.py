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

import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.mask_utils import compute_mask_constraint
import uuid
from tqdm import tqdm
from utils.image_utils import psnr,depth2normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # if depth_l1_weight(iteration) > 0 and not viewpoint_cam.depth_reliable:
        #     continue

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        # RGB Loss (已注释，仅使用深度监督)
        # gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # if FUSED_SSIM_AVAILABLE:
        #     ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        # else:
        #     ssim_value = ssim(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # 初始化loss组件
        loss = torch.tensor(0.0, device="cuda", requires_grad=True)
        loss_depth_l1 = 0.0
        loss_normal = 0.0
        loss_smooth = 0.0

        # Depth regularization with normal consistency
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            # 渲染器返回的是 inverse depth（逆深度），不是 depth
            # CUDA代码中：expected_invdepth += (1 / depths[j]) * alpha * T
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda().clamp(0.0, 1.0)

            alpha_mask = getattr(viewpoint_cam, "alpha_mask", None)

            # if alpha_mask is not None:
            #     depth_mask = depth_mask * alpha_mask.cuda().clamp(0.0, 1.0)

            # === 深度尺度对齐：使用跨视角一致性对齐 ===
            # 第一次运行时计算全局对齐参数（只计算一次）
            # if not hasattr(scene, 'depth_alignment_params'):
            #     from utils.depth_alignment import align_depth_cross_view

            #     # 使用所有训练相机进行跨视角对齐
            #     all_cameras = scene.getTrainCameras()
            #     scene.depth_alignment_params = align_depth_cross_view(all_cameras, max_views=10)

            #     if len(scene.depth_alignment_params) == 0:li
            #         print("[Warning] Cross-view depth alignment failed, using identity scale")
            #         scene.depth_alignment_params = None

            # # 应用对齐参数
            # if scene.depth_alignment_params is not None and viewpoint_cam.image_name in scene.depth_alignment_params:
            #     scale, shift = scene.depth_alignment_params[viewpoint_cam.image_name]

            #     # 转换逆深度为深度
            #     mono_depth = 1.0 / (mono_invdepth.squeeze(0) + 1e-6)

            #     # 应用对齐
            #     aligned_depth = scale * mono_depth + shift

            #     # 转回逆深度
            #     mono_invdepth = 1.0 / (aligned_depth.unsqueeze(0) + 1e-6)

            #     if iteration % 1000 == 0:
            #         print(f"[Iter {iteration}] View {viewpoint_cam.image_name}: "
            #               f"depth alignment scale={scale:.4f}")
            # === 深度对齐结束 ===

            valid = (depth_mask > 0.5).float()
            denom = valid.sum().clamp(min=1.0)

            # 1. 深度L1损失
            depth_l1_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).sum() / denom
            loss_depth_l1 = depth_l1_weight(iteration) * depth_l1_pure

            # 2. 法向一致性loss：比较渲染深度和单目深度各自计算出的法向
            # 从渲染深度计算法向
            render_normal = depth2normal(invDepth, depth_mask, viewpoint_cam)
            # 从单目深度计算法向
            mono_normal = depth2normal(mono_invdepth, depth_mask, viewpoint_cam)

            # 计算法向余弦相似度 (越接近1越好)
            cos_sim = (render_normal * mono_normal).sum(dim=0)  # [H, W]
            # 转换为loss：1 - similarity
            normal_loss_pure = ((1.0 - cos_sim) * depth_mask).sum() / denom

            # 法向loss权重
            normal_weight = 0.2
            # loss_normal = depth_l1_weight(iteration) * normal_weight * normal_loss_pure
            loss_normal = normal_weight * normal_loss_pure

            # 3. 深度平滑loss：惩罚深度不连续，去除离散点云
            # 在迭代5000后启用，给足够时间让模型先收敛
            lambda_smooth = 0.5 if iteration > 5000 else 0.0
            loss_smooth = 0.0

            if lambda_smooth > 0:
                # 转换逆深度为深度
                depth_render = 1.0 / (invDepth.squeeze(0) + 1e-6)

                # 计算深度梯度（水平和垂直方向）
                # 如果存在离散点云，深度会有突变，梯度会很大
                grad_x = torch.abs(depth_render[:, 1:] - depth_render[:, :-1])  # [H, W-1]
                grad_y = torch.abs(depth_render[1:, :] - depth_render[:-1, :])  # [H-1, W]

                # 只在mask有效的相邻区域计算
                mask_2d = depth_mask.squeeze(0)
                mask_x = mask_2d[:, 1:] * mask_2d[:, :-1]  # 相邻像素都有效
                mask_y = mask_2d[1:, :] * mask_2d[:-1, :]

                # 计算有效区域的梯度均值
                denom_x = mask_x.sum().clamp(min=1.0)
                denom_y = mask_y.sum().clamp(min=1.0)

                loss_smooth = lambda_smooth * (
                    (grad_x * mask_x).sum() / denom_x +
                    (grad_y * mask_y).sum() / denom_y
                )

            # 总loss = 深度loss + 法向loss + 平滑loss
            loss = loss + loss_depth_l1 + loss_normal + loss_smooth

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            # 处理 loss_depth_l1 可能是 float 或 tensor 的情况
            depth_loss_value = loss_depth_l1.item() if isinstance(loss_depth_l1, torch.Tensor) else loss_depth_l1
            ema_Ll1depth_for_log = 0.4 * depth_loss_value + 0.6 * ema_Ll1depth_for_log

            # 处理平滑loss
            smooth_loss_value = loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else loss_smooth

            if iteration % 10 == 0:
                postfix_dict = {
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Depth": f"{ema_Ll1depth_for_log:.{7}f}",
                    "Smooth": f"{smooth_loss_value:.{5}f}"
                }
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, loss_depth_l1 if isinstance(loss_depth_l1, torch.Tensor) else torch.tensor(0.0), loss, torch.tensor(0.0), iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # 计算mask约束：判断每个Gaussian点是否在至少一个视图的mask内
                    valid_region_mask = None
                    if dataset.depth_mask_dir != "":
                        valid_region_mask = compute_mask_constraint(gaussians, scene, render, pipe, background)

                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, valid_region_mask)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)

                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss_value, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
