#!/usr/bin/env python3
"""
智能物体mask提取：结合RGB和深度信息
使用GrabCut + 深度聚类，无需额外依赖
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


class SmartMaskExtractor:
    """智能mask提取器"""
    
    def __init__(self):
        self.rect = None
        self.drawing = False
        self.ix, self.iy = -1, -1
        
    def draw_rectangle(self, event, x, y, flags, param):
        """鼠标绘制矩形回调"""
        image, window_name = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = image.copy()
                cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow(window_name, img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.rectangle(image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            self.rect = (min(self.ix, x), min(self.iy, y),
                        abs(x - self.ix), abs(y - self.iy))
            cv2.imshow(window_name, image)
    
    def get_object_roi(self, rgb_image):
        """让用户框选目标物体"""
        display = rgb_image.copy()
        window_name = "框选目标物体 | 拖动鼠标框选，按q完成"
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.draw_rectangle, (display, window_name))
        
        print("\n=== 框选目标物体 ===")
        print("1. 按住鼠标左键拖动，框选目标物体")
        print("2. 按 'q' 键确认选择")
        print("3. 按 'r' 键重新框选")
        
        while True:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') and self.rect is not None:
                break
            elif key == ord('r'):
                display = rgb_image.copy()
                self.rect = None
                print("已重置，请重新框选")
        
        cv2.destroyAllWindows()
        return self.rect
    
    def extract_mask_grabcut(self, rgb_image, depth_image, rect):
        """
        使用GrabCut + 深度信息提取mask
        
        策略：
        1. 用户框选大致区域
        2. GrabCut精细分割
        3. 用深度信息过滤误检
        """
        # 初始化GrabCut的mask
        mask = np.zeros(rgb_image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # 运行GrabCut
        print("运行GrabCut分割...")
        cv2.grabCut(rgb_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # 生成初步mask（0,2=背景，1,3=前景）
        mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # 结合深度信息优化
        if depth_image is not None:
            print("结合深度信息优化...")
            depth_m = depth_image.astype(np.float32) / 5000.0
            
            # 计算前景的深度范围
            fg_depths = depth_m[mask_binary == 1]
            fg_depths = fg_depths[fg_depths > 0]
            
            if len(fg_depths) > 0:
                depth_mean = np.median(fg_depths)
                depth_std = np.std(fg_depths)
                
                # 保留深度接近的区域（3倍标准差）
                depth_mask = ((depth_m > depth_mean - 3*depth_std) & 
                             (depth_m < depth_mean + 3*depth_std) &
                             (depth_m > 0))
                
                # 与GrabCut结果取交集
                mask_binary = (mask_binary & depth_mask).astype('uint8')
                
                print(f"  前景深度: {depth_mean:.2f}m ± {depth_std:.2f}m")
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
        
        # 保留最大连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        if num_labels > 1:
            # 找最大区域（排除背景label=0）
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_binary = (labels == largest_label).astype('uint8')
        
        return mask_binary


def extract_masks_smart(rgb_dir, depth_dir, output_dir):
    """智能提取所有帧的mask"""
    
    extractor = SmartMaskExtractor()
    
    rgb_files = sorted(Path(rgb_dir).glob('*.png'))
    depth_files = sorted(Path(depth_dir).glob('*.png'))
    
    print(f"\n找到 {len(rgb_files)} 张RGB图像")
    print(f"找到 {len(depth_files)} 张深度图")
    
    # 在第一帧上框选物体
    first_rgb = cv2.imread(str(rgb_files[0]))
    first_depth = cv2.imread(str(depth_files[0]), cv2.IMREAD_ANYDEPTH)
    
    print("\n在第一帧上框选目标物体...")
    rect = extractor.get_object_roi(first_rgb)
    
    print(f"选择区域: {rect}")
    
    # 生成第一帧mask
    reference_mask = extractor.extract_mask_grabcut(first_rgb, first_depth, rect)
    
    # 可视化第一帧结果
    result = first_rgb.copy()
    result[reference_mask == 1] = result[reference_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.imshow("第一帧分割结果 (按任意键继续)", result.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 计算参考深度范围（用于后续帧）
    ref_depth_m = first_depth.astype(np.float32) / 5000.0
    ref_depths = ref_depth_m[reference_mask == 1]
    ref_depths = ref_depths[ref_depths > 0]
    depth_mean = np.median(ref_depths)
    depth_std = np.std(ref_depths)
    
    print(f"\n参考深度: {depth_mean:.2f}m ± {depth_std:.2f}m")
    print(f"将使用此深度范围处理其他帧...")
    
    # 处理所有帧
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        rgb = cv2.imread(str(rgb_file))
        depth = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
        
        if idx == 0:
            # 第一帧已经处理过
            mask = reference_mask
        else:
            # 后续帧：使用同样的rect + 深度约束
            mask = extractor.extract_mask_grabcut(rgb, depth, rect)
        
        # 保存mask
        output_path = os.path.join(output_dir, rgb_file.name)
        cv2.imwrite(output_path, mask * 255)
        
        if (idx + 1) % 20 == 0:
            print(f"  进度: {idx+1}/{len(rgb_files)}")
    
    print(f"\n✓ 完成！Mask保存到: {output_dir}")
    
    # 统计覆盖率
    coverage = (reference_mask > 0).sum() / reference_mask.size * 100
    print(f"第一帧覆盖率: {coverage:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='智能物体mask提取')
    parser.add_argument('--rgb_dir', required=True, help='RGB图像目录')
    parser.add_argument('--depth_dir', required=True, help='深度图目录')
    parser.add_argument('--output_dir', required=True, help='输出mask目录')
    
    args = parser.parse_args()
    
    extract_masks_smart(args.rgb_dir, args.depth_dir, args.output_dir)
