# 诊断脚本 - 检查深度畸变loss是否真的在工作

import torch

# 模拟检查
print("=== 深度畸变Loss诊断 ===\n")

# 问题1: depth_distortion是否在render_pkg中返回？
print("1. 检查render_pkg中是否有depth_distortion:")
print("   - 查看 gaussian_renderer/__init__.py")
print("   - 确认返回的dict中包含 'depth_distortion'")
print()

# 问题2: CUDA代码是否正确实现？
print("2. 检查CUDA实现:")
print("   - submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu")
print("   - 是否有两次遍历计算depth_distortion？")
print()

# 问题3: loss值是否正常？
print("3. 检查训练日志:")
print("   - Dist loss的值是多少？")
print("   - 如果是0.00000，说明depth_distortion根本没计算")
print("   - 如果很小（<0.001），可能需要调整scale")
print()

print("=== 建议的检查步骤 ===")
print("1. 在train.py中添加打印，看depth_distortion的值")
print("2. 确认CUDA代码确实被编译了")
print("3. 查看训练开始时的loss值")
