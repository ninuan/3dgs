# 快速开始：Mask约束的点云优化

## 🎯 目标
使用深度图和mask训练，让点云只聚焦在目标物体上，不发散到周围。

## ✅ 数据准备（已完成）

您的data目录已包含：
- ✅ 6张深度图 (data/depth/)
- ✅ 6张mask (data/mask/)
- ✅ COLMAP数据 (data/sparse/0/)
- ✅ 初始点云 (33,404个点)

## 🚀 一键开始

### 推荐命令（保守策略）

```bash
# 激活环境
conda activate gaussian_splatting

# 开始训练
python train.py -s data/ -d depth --depth_mask_dir mask \
    --densify_grad_threshold 0.0004 \
    --densification_interval 150 \
    --densify_until_iter 10000 \
    --iterations 20000 \
    -m output/optimized_object \
    --disable_viewer
```

### 为什么使用这些参数？

| 参数 | 值 | 原因 |
|------|-----|------|
| `-d depth` | 深度图目录 | 使用深度监督 |
| `--depth_mask_dir mask` | Mask目录 | **启用mask约束** ✅ |
| `--densify_grad_threshold 0.0004` | 梯度阈值加倍 | 减少点云增长 |
| `--densification_interval 150` | 间隔增加50% | 降低densification频率 |
| `--densify_until_iter 10000` | 提前停止 | 防止后期发散 |
| `--iterations 20000` | 减少迭代 | 更快收敛 |
| `--disable_viewer` | 禁用viewer | 避免端口冲突 |

## 📊 监控训练

### 期望看到的输出

```bash
# 开始训练
Optimizing output/optimized_object
Loading Training Cameras
Number of points at initialisation: 33404

# Densification时的mask约束信息
[Mask Constraint] 28500/33404 points in valid region, 4904 outside

# 训练进度
Training progress: 50%|█████| 10000/20000 [00:55<00:55, 181it/s, Loss=0.0003, Depth Loss=0.0003]
```

### 关键指标

- **Loss**: 应该逐渐下降到 0.0001-0.001
- **Depth Loss**: 深度监督loss，主要优化指标
- **Points in valid region**: 在mask内的点数

## 🎨 三种训练策略

### 1. 保守策略（推荐）⭐

```bash
python train.py -s data/ -d depth --depth_mask_dir mask \
    --densify_grad_threshold 0.0004 \
    --densification_interval 150 \
    --densify_until_iter 10000 \
    --iterations 20000 \
    --disable_viewer
```

**适合**：单目标优化，想要最聚焦的点云

### 2. 平衡策略

```bash
python train.py -s data/ -d depth --depth_mask_dir mask \
    --densify_grad_threshold 0.0003 \
    --densification_interval 120 \
    --iterations 30000 \
    --disable_viewer
```

**适合**：需要更多细节，可以接受轻微发散

### 3. 快速测试

```bash
python train.py -s data/ -d depth --depth_mask_dir mask \
    --iterations 5000 \
    --disable_viewer
```

**适合**：快速验证效果（5分钟内完成）

## 📁 查看结果

```bash
# 查看生成的点云
ls -lh output/optimized_object/point_cloud/

# 最终点云位置
output/optimized_object/point_cloud/iteration_20000/point_cloud.ply
```

## ✨ 关键改进

### 修改前
```
❌ 点云发散到目标外
❌ 产生大量无关点
❌ 难以提取目标模型
```

### 修改后
```
✅ 点云严格限制在mask内
✅ 自动裁剪发散点
✅ 只在目标区域densify
✅ 保持目标形状完整
```

## 🔧 故障排除

### 问题1: 仍有少量发散点

**解决**：使用更保守的参数
```bash
# 进一步增大梯度阈值
--densify_grad_threshold 0.0005

# 更早停止densification
--densify_until_iter 8000
```

### 问题2: 目标细节不足

**解决**：适当放松约束
```bash
# 降低梯度阈值
--densify_grad_threshold 0.0003

# 延长训练
--iterations 30000
```

### 问题3: 训练太慢

**解决**：减少迭代或降低分辨率
```bash
# 快速训练
--iterations 10000

# 或降低分辨率
-r 2
```

## 📚 完整文档

- **OPTIMIZATION_SUMMARY.md** - 完整优化总结
- **MASK_OPTIMIZATION_GUIDE.md** - 详细使用指南
- **FINAL_TRAINING_GUIDE.md** - 基础训练指南

## 💡 核心原理

```
Mask约束机制:
1. 计算每个Gaussian点是否在mask内
2. Densification时只克隆mask内的点
3. 定期裁剪mask外的点
4. 保持点云聚焦在目标物体
```

## 🎉 开始训练！

```bash
# 复制这个命令直接运行
python train.py -s data/ -d depth --depth_mask_dir mask \
    --densify_grad_threshold 0.0004 \
    --densification_interval 150 \
    --densify_until_iter 10000 \
    --iterations 20000 \
    -m output/my_optimized_object \
    --disable_viewer
```

训练时间：约15-20分钟（取决于硬件）

---

**准备状态**: ✅ 就绪  
**预期效果**: 聚焦的目标点云，无发散  
**关键参数**: `--depth_mask_dir mask` (必须设置)
