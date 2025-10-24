import cv2
import numpy as np

# 检查mask文件
mask_path = "data/mask/000009.png"
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

print(f"Mask形状: {mask.shape}")
print(f"Mask数据类型: {mask.dtype}")
print(f"Mask范围: min={mask.min()}, max={mask.max()}")
print(f"Mask非零像素数: {(mask > 0).sum()} / {mask.size}")
print(f"Mask非零比例: {(mask > 0).sum() / mask.size * 100:.2f}%")

# 可视化mask
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(mask, cmap='gray')
plt.title("Mask Visualization")
plt.colorbar()
plt.savefig("debug_mask.png", dpi=150, bbox_inches='tight')
print("\nMask可视化已保存到 debug_mask.png")
