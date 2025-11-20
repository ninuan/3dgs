#!/usr/bin/env python3
"""
使用SAM (Segment Anything Model) 提取物体mask
需要安装: pip install segment-anything opencv-python torch torchvision
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("警告: segment-anything未安装")
    print("安装方法: pip install git+https://github.com/facebookresearch/segment-anything.git")


class InteractiveMaskExtractor:
    """交互式mask提取器"""
    
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth"):
        if not SAM_AVAILABLE:
            raise ImportError("请先安装 segment-anything")
        
        # 加载SAM模型
        print("加载SAM模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.device = device
        
        self.points = []
        self.labels = []  # 1=前景点，0=背景点
        
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
    
    def get_mask_from_user_input(self, rgb_image):
        """
        让用户交互式选择物体
        左键：点击物体（前景）
        右键：点击背景
        按'q'完成选择
        """
        display = rgb_image.copy()
        window_name = "选择目标物体 | 左键=前景, 右键=背景, q=完成"
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.click_callback, (display, window_name))
        
        print("\n=== 交互式物体选择 ===")
        print("1. 左键点击：标记目标物体上的点（绿色）")
        print("2. 右键点击：标记背景点（红色）")
        print("3. 按 'q' 键：完成选择")
        print("4. 按 'r' 键：重置所有点")
        
        while True:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') and len(self.points) > 0:  # 完成
                break
            elif key == ord('r'):  # 重置
                self.points = []
                self.labels = []
                display = rgb_image.copy()
                print("已重置所有点")
        
        cv2.destroyAllWindows()
        
        # 使用SAM生成mask
        self.predictor.set_image(rgb_image)
        input_points = np.array(self.points)
        input_labels = np.array(self.labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # 选择得分最高的mask
        best_mask = masks[scores.argmax()]
        
        # 可视化结果
        result = rgb_image.copy()
        result[best_mask] = result[best_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.imshow("分割结果 (按任意键继续)", result.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return best_mask


def extract_masks_sam(rgb_dir, depth_dir, output_dir, sam_checkpoint):
    """使用SAM提取所有帧的mask"""
    
    extractor = InteractiveMaskExtractor(sam_checkpoint)
    
    rgb_files = sorted(Path(rgb_dir).glob('*.png'))
    print(f"\n找到 {len(rgb_files)} 张RGB图像")
    
    # 在第一帧上交互式选择物体
    first_rgb = cv2.imread(str(rgb_files[0]))
    first_rgb = cv2.cvtColor(first_rgb, cv2.COLOR_BGR2RGB)
    
    print("\n在第一帧上选择目标物体...")
    reference_mask = extractor.get_mask_from_user_input(first_rgb)
    
    # 为所有帧生成mask
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n开始处理所有帧...")
    
    for idx, rgb_file in enumerate(rgb_files):
        rgb = cv2.imread(str(rgb_file))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # 使用相同的点在每一帧上预测
        extractor.predictor.set_image(rgb)
        input_points = np.array(extractor.points)
        input_labels = np.array(extractor.labels)
        
        masks, scores, _ = extractor.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        
        mask = (masks[0] * 255).astype(np.uint8)
        
        # 保存mask
        output_path = os.path.join(output_dir, rgb_file.name)
        cv2.imwrite(output_path, mask)
        
        if (idx + 1) % 20 == 0:
            print(f"  进度: {idx+1}/{len(rgb_files)}")
    
    print(f"\n✓ 完成！Mask保存到: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用SAM提取物体mask')
    parser.add_argument('--rgb_dir', required=True, help='RGB图像目录')
    parser.add_argument('--depth_dir', required=True, help='深度图目录')
    parser.add_argument('--output_dir', required=True, help='输出mask目录')
    parser.add_argument('--sam_checkpoint', default='sam_vit_h_4b8939.pth',
                       help='SAM模型权重路径')
    
    args = parser.parse_args()
    
    if not SAM_AVAILABLE:
        print("\n错误: 未安装segment-anything库")
        print("请运行: pip install git+https://github.com/facebookresearch/segment-anything.git")
        exit(1)
    
    extract_masks_sam(args.rgb_dir, args.depth_dir, args.output_dir, args.sam_checkpoint)
