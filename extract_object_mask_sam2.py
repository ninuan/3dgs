#!/usr/bin/env python3
"""
使用SAM 2 (Segment Anything Model 2) 提取视频物体mask
SAM 2专门为视频设计,能够跨帧传播mask,更好地处理视角变化
需要安装: pip install git+https://github.com/facebookresearch/segment-anything-2.git
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import torch

try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("警告: SAM 2未安装")
    print("安装方法: pip install git+https://github.com/facebookresearch/sam2.git")


class VideoMaskExtractor:
    """SAM 2视频mask提取器"""
    
    def __init__(self, checkpoint="checkpoints/sam2.1_hiera_large.pt", 
                 model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
        if not SAM2_AVAILABLE:
            raise ImportError("请先安装 SAM 2")
        
        print("加载SAM 2.1视频模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        self.device = device
        
        self.points = []
        self.labels = []
        self.current_frame = None
        
    def click_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        image, window_name = param
        
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键：添加前景点
            self.points.append([x, y])
            self.labels.append(1)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 绿色
            print(f"添加前景点: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键：添加背景点
            self.points.append([x, y])
            self.labels.append(0)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 红色
            print(f"添加背景点: ({x}, {y})")
            
        cv2.imshow(window_name, image)
    
    def get_user_annotation(self, rgb_image, frame_idx=0):
        """让用户在指定帧上标注物体"""
        display = rgb_image.copy()
        window_name = f"帧{frame_idx} | 左键=前景, 右键=背景, q=完成"
        
        self.points = []
        self.labels = []
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.click_callback, (display, window_name))
        
        print(f"\n=== 在帧 {frame_idx} 上标注物体 ===")
        print("1. 左键点击：标记目标物体上的点（绿色）")
        print("2. 右键点击：标记背景点（红色）")
        print("3. 按 'q' 键：完成选择")
        print("4. 按 'r' 键：重置所有点")
        
        while True:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') and len(self.points) > 0:
                break
            elif key == ord('r'):
                self.points = []
                self.labels = []
                display = rgb_image.copy()
                print("已重置所有点")
        
        cv2.destroyAllWindows()
        
        return np.array(self.points), np.array(self.labels)


def extract_masks_sam2(rgb_dir, output_dir, checkpoint, model_cfg):
    """使用SAM 2提取视频序列的mask"""
    
    # 获取所有RGB图像
    rgb_files = sorted(Path(rgb_dir).glob('*.png'))
    print(f"\n找到 {len(rgb_files)} 张RGB图像")
    
    if len(rgb_files) == 0:
        print("错误: 未找到图像文件")
        return
    
    # 创建临时视频目录结构 (SAM 2需要特定的目录结构)
    video_dir = Path(output_dir).parent / "temp_video"
    video_dir.mkdir(exist_ok=True)
    
    # 将图像复制到临时目录并转换为JPG (SAM 2推荐格式)
    print("\n准备视频帧...")
    frame_names = []
    for idx, rgb_file in enumerate(rgb_files):
        img = cv2.imread(str(rgb_file))
        frame_name = f"{idx:06d}.jpg"
        frame_path = video_dir / frame_name
        cv2.imwrite(str(frame_path), img)
        frame_names.append(frame_name)
    
    # 初始化提取器
    extractor = VideoMaskExtractor(checkpoint, model_cfg)
    
    # 初始化SAM 2的视频预测器
    print("\n初始化SAM 2视频预测器...")
    inference_state = extractor.predictor.init_state(video_path=str(video_dir))
    
    # 在第一帧上让用户标注
    first_frame = cv2.imread(str(rgb_files[0]))
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    print("\n请在第一帧上标注目标物体...")
    points, labels = extractor.get_user_annotation(first_frame_rgb, frame_idx=0)
    
    # 将点击添加到第一帧
    ann_frame_idx = 0
    ann_obj_id = 1  # 物体ID
    
    _, out_obj_ids, out_mask_logits = extractor.predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    
    # 在第一帧上预览结果
    print("\n预览第一帧的分割结果...")
    first_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
    preview = first_frame_rgb.copy()
    preview[first_mask] = preview[first_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.imshow("第一帧分割结果 (按任意键继续)", preview.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 传播到整个视频
    print("\n开始传播分割到所有帧...")
    os.makedirs(output_dir, exist_ok=True)
    
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in extractor.predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
        if (out_frame_idx + 1) % 20 == 0:
            print(f"  进度: {out_frame_idx + 1}/{len(rgb_files)}")
    
    # 保存所有mask
    print("\n保存mask...")
    for frame_idx in range(len(rgb_files)):
        if frame_idx in video_segments and ann_obj_id in video_segments[frame_idx]:
            mask = video_segments[frame_idx][ann_obj_id].squeeze()
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            # 如果某帧没有分割结果,创建空mask
            h, w = cv2.imread(str(rgb_files[frame_idx])).shape[:2]
            mask_uint8 = np.zeros((h, w), dtype=np.uint8)
        
        output_path = os.path.join(output_dir, rgb_files[frame_idx].name)
        cv2.imwrite(output_path, mask_uint8)
    
    # 清理临时目录
    print("\n清理临时文件...")
    import shutil
    shutil.rmtree(video_dir)
    
    print(f"\n✓ 完成！Mask保存到: {output_dir}")
    print(f"✓ 共处理 {len(rgb_files)} 帧")
    
    # 可视化一些结果
    print("\n显示部分结果预览...")
    preview_frames = [0, len(rgb_files)//4, len(rgb_files)//2, 3*len(rgb_files)//4, len(rgb_files)-1]
    
    for idx in preview_frames:
        if idx < len(rgb_files):
            rgb = cv2.imread(str(rgb_files[idx]))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            mask_path = os.path.join(output_dir, rgb_files[idx].name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            result = rgb.copy()
            result[mask > 128] = result[mask > 128] * 0.5 + np.array([0, 255, 0]) * 0.5
            
            cv2.imshow(f"帧 {idx} (按任意键继续)", result.astype(np.uint8))
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用SAM 2提取视频物体mask')
    parser.add_argument('--rgb_dir', required=True, help='RGB图像目录')
    parser.add_argument('--output_dir', required=True, help='输出mask目录')
    parser.add_argument('--checkpoint', default='checkpoints/sam2.1_hiera_large.pt',
                       help='SAM 2模型权重路径')
    parser.add_argument('--model_cfg', default='configs/sam2.1/sam2.1_hiera_l.yaml',
                       help='SAM 2模型配置文件')
    
    args = parser.parse_args()
    
    if not SAM2_AVAILABLE:
        print("\n错误: 未安装SAM 2库")
        print("请运行: pip install git+https://github.com/facebookresearch/sam2.git")
        exit(1)
    
    extract_masks_sam2(args.rgb_dir, args.output_dir, args.checkpoint, args.model_cfg)
