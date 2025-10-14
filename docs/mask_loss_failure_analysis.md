# Mask-Based Loss 失败分析与建议

## 问题总结

尝试实现mask-based loss来约束mask外的发散点，但结果导致**目标点云消失，只剩下无效的高斯点**。

## 失败原因

### 实现的Mask-Based Loss

```python
weighted_mask = depth_mask + lambda_outside * (1.0 - depth_mask)
virtual_gt = mono_invdepth * depth_mask
Ll1depth_pure = torch.abs((invDepth - virtual_gt) * weighted_mask).sum() / weighted_mask.sum()
```

### 致命缺陷

1. **虚拟GT的问题**：
   ```
   virtual_gt = mono_invdepth * depth_mask

   - mask内(mask=1): virtual_gt = gt_depth ✓
   - mask外(mask=0): virtual_gt = 0 ✗
   ```

   问题：mask外的loss变成 `|rendered - 0| * weight`
   → **惩罚所有rendered depth，把所有点的深度贡献都拉向0**
   → 结果：所有Gaussian的opacity被降低，最终被prune掉

2. **权重不平衡**：
   ```
   weighted_mask = mask + 0.5 * (1-mask)

   - mask内权重: 1.0
   - mask外权重: 0.5
   ```

   即使mask外权重较小，但因为virtual_gt=0，**任何非零的rendered depth都会产生loss**
   → mask外的"错误loss"累积起来主导了整个优化过程

3. **梯度方向错误**：
   ```
   dL/d(opacity) 对于mask外的点:
   - Loss = |rendered| * 0.5 (因为virtual_gt=0)
   - 梯度总是正的 → 降低opacity
   - 但这也影响到了mask内的点（因为3D Gaussian是连续的）
   ```

### 为什么目标点云消失了

```
Training过程:

Iteration 1-1000:
  - mask外: Loss = |rendered| * 0.5 → 降低这些区域的opacity
  - mask内: Loss = |rendered - gt| → 对齐深度
  - 结果：mask内的点虽然在学习深度，但也受到mask边缘的负面影响

Iteration 1000-5000:
  - mask外的点opacity → 0，被prune掉 ✓ (这是预期的)
  - 但mask内靠近边缘的点也受影响 ✗
  - 坐标变换可能学习到错误的translation（因为loss信号混乱）

Iteration 5000+:
  - mask内的点也逐渐opacity → 0
  - 因为整个loss landscape被mask外的"拉向0"主导
  - 最终：所有有效点都被prune掉
```

## 为什么之前的方案更好

### Hybrid Gradients (100×/50×) + Coordinate Transform

**工作原理**：
```
1. 深度loss正常计算（只在mask内）
   L = |rendered - gt| * mask

2. 混合梯度放大
   - 3D gradient × 100: 提供深度对齐信号
   - 2D gradient × 50: 提供空间约束

3. 坐标变换
   - 全局刚体约束，保持点云结构
   - 自动对齐COLMAP和深度图坐标系
```

**为什么有效**：
- ✅ Loss只在mask内计算，不会误伤有效点
- ✅ 梯度放大只影响densification，不改变loss方向
- ✅ 坐标变换提供全局约束，防止整体发散
- ✅ Pruning机制自然移除低opacity的点

**为什么不会把mask外的点都pruning掉**：
- Pruning只基于opacity threshold（0.005）
- mask外的点如果opacity > 0.005，就不会被prune
- 但混合梯度会让它们的梯度累积较低，不触发densification
- 结果：mask外的发散点逐渐减少，但不会完全消失

## 正确的Mask约束方法

如果确实需要mask约束，应该：

### 方法1：只在Pruning阶段使用Mask

```python
# 在densify_and_prune时
if iteration % densification_interval == 0:
    # 计算哪些点在mask外
    outside_mask_points = compute_outside_mask_points(gaussians, cameras, masks)

    # 对mask外的点使用更aggressive的pruning threshold
    prune_mask = (opacity < 0.005)  # 常规pruning
    prune_mask_outside = (opacity < 0.05) & outside_mask_points  # mask外更严格
    final_prune_mask = prune_mask | prune_mask_outside

    gaussians.prune_points(final_prune_mask)
```

**优点**：
- 不影响loss计算和梯度流动
- 只在pruning阶段起作用
- 不会误伤mask内的点

### 方法2：Opacity Regularization

```python
# 在loss中添加opacity正则项
if mask is not None:
    outside = (1.0 - mask).bool()
    outside_gaussians = compute_gaussian_mask_overlap(gaussians, cameras[i], outside)
    opacity_reg = gaussians.get_opacity[outside_gaussians].mean()
    loss = loss + 0.01 * opacity_reg  # 鼓励mask外的点降低opacity
```

**优点**：
- 直接作用于opacity，不影响深度loss
- 权重较小，不会主导优化
- 梯度明确：只降低opacity，不影响xyz/scale/rotation

## 最终建议

**不要使用mask-based depth loss**，原因：
1. 很难正确实现（需要精确的mask语义和权重平衡）
2. 容易误伤有效点（mask边缘的模糊性）
3. 可能与坐标变换学习冲突

**继续使用原来的方案**：
- Hybrid gradients: 100× (3D) + 50× (2D)
- Coordinate transform learning
- 原始learning rates
- 如果轻微发散，增加pruning频率或降低opacity threshold

**可选增强**（如果确实需要更强约束）：
- 方法1：在pruning阶段对mask外的点使用更严格的threshold
- 方法2：添加小权重的opacity regularization（只针对mask外的点）

---

*文档生成日期: 2025-10-13*
*问题: mask-based loss导致点云消失*
*结论: 回退到hybrid gradients方案*
