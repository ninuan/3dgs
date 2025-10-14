# Method 1: Aggressive Pruning 实现说明

## 概述

实现了 `mask_loss_failure_analysis.md` 文档中提及的 Method 1，通过在 pruning 阶段对 mask 外的点使用更严格的 opacity threshold，快速移除发散点，同时保持 mask 内的有效点云。

## 实现位置

文件：`scene/gaussian_model.py`

函数：`densify_and_prune()` (lines 521-570)

## 核心实现

### 代码逻辑

```python
# Method 1: Aggressive pruning for points outside mask
if valid_region_mask is not None:
    # 识别 mask 外的点
    outside_mask = ~valid_region_mask

    # 更严格的 opacity threshold：0.05 (是常规 0.005 的 10 倍)
    aggressive_threshold = 0.05
    outside_low_opacity = (self.get_opacity[:num_points_before] < aggressive_threshold).squeeze() & outside_mask

    # 扩展 mask 以匹配当前点数（densification 可能增加了点）
    extended_outside_prune = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    extended_outside_prune[:num_points_before] = outside_low_opacity

    # 合并到总的 pruning mask
    prune_mask = torch.logical_or(prune_mask, extended_outside_prune)

    # Debug 信息
    if outside_low_opacity.sum().item() > 0:
        print(f"[Aggressive Pruning] Removing {outside_low_opacity.sum().item()} "
              f"low-opacity points outside mask (threshold={aggressive_threshold})")
```

### 关键设计

1. **只影响 pruning，不影响 loss**：
   - ✅ Loss 计算保持原样（只在 mask 内计算深度 loss）
   - ✅ 不会产生梯度方向错误
   - ✅ 不会误伤 mask 内的有效点

2. **分级 threshold**：
   - Mask 内的点：使用常规 threshold 0.005
   - Mask 外的点：使用严格 threshold 0.05（10x 严格）
   - 效果：快速移除低 opacity 的发散点

3. **只作用于原始点**：
   - 使用 `num_points_before` 来标记 densification 前的点数
   - 只对原始点应用 aggressive pruning
   - 新增的点（通过 clone/split）不受影响

4. **兼容性**：
   - 与现有的 hybrid gradients (175×/50×) 兼容
   - 与 coordinate transform learning 兼容
   - 不改变其他 pruning 规则（big points, low opacity）

## 与失败方案的对比

### 失败的 Mask-Based Loss

```python
# ❌ 错误：修改了 loss 计算
weighted_mask = depth_mask + lambda_outside * (1.0 - depth_mask)
virtual_gt = mono_invdepth * depth_mask  # mask 外 GT = 0
Ll1depth_pure = torch.abs((invDepth - virtual_gt) * weighted_mask).sum() / weighted_mask.sum()

# 问题：
# 1. mask 外的 loss = |rendered| * weight → 拉所有深度到 0
# 2. 误伤了 mask 内的点
# 3. 导致整个点云消失
```

### Method 1 (当前实现)

```python
# ✅ 正确：只在 pruning 阶段起作用
if valid_region_mask is not None:
    outside_mask = ~valid_region_mask
    aggressive_threshold = 0.05
    outside_low_opacity = (self.get_opacity[:num_points_before] < aggressive_threshold).squeeze() & outside_mask
    prune_mask = torch.logical_or(prune_mask, extended_outside_prune)

# 优点：
# 1. 不影响 loss 和梯度流动
# 2. 只移除 mask 外 + 低 opacity 的点
# 3. 保留 mask 内的所有点
# 4. 实现简单，不易出错
```

## 配置参数

### 当前设置

- **Gradient multipliers**:
  - 3D gradient: 175× (gaussian_model.py:593)
  - 2D gradient: 50× (gaussian_model.py:602)

- **Aggressive threshold**: 0.05 (gaussian_model.py:550)
  - 相比常规 threshold 0.005 严格 10 倍

- **Transform learning rates**:
  - Translation: 0.01
  - Rotation: 0.001
  - Scale: 0.001

### 可调参数

如果需要更强/更弱的约束，可以调整：

```python
# 更激进的 pruning (移除更多点)
aggressive_threshold = 0.08  # 或 0.1

# 更保守的 pruning (保留更多点)
aggressive_threshold = 0.03  # 或 0.02
```

## 预期效果

1. **发散点减少**：
   - Mask 外的低 opacity 点被快速移除
   - 每次 densification 后会看到 `[Aggressive Pruning]` 消息

2. **有效点保留**：
   - Mask 内的点不受影响
   - 目标点云结构完整

3. **训练稳定**：
   - 不会出现梯度消失
   - Translation 不会爆炸
   - Loss 正常下降

4. **Debug 输出示例**：
   ```
   [Densify] Grad stats: mean=0.005234, max=2.451234, >threshold=234/15000, threshold=0.0002
   [Aggressive Pruning] Removing 127 low-opacity points outside mask (threshold=0.05)
   ```

## 测试方法

使用提供的测试脚本：

```bash
bash /tmp/test_method1_aggressive_pruning.sh
```

训练 1500 iterations 快速验证效果。

检查要点：
1. 是否有 `[Aggressive Pruning]` 输出
2. 移除的点数是否逐渐减少（说明发散点被清理）
3. 梯度统计是否保持健康
4. Translation 是否收敛到合理值

## 文件修改记录

- `scene/gaussian_model.py` (lines 543-563): 添加 Method 1 实现
- 移除了旧的 mask constraint 代码 (原 lines 565-586)
- 保持 loss 计算不变 (train.py)

## 后续优化方向

如果 Method 1 效果良好但仍有少量发散点，可以考虑：

1. **逐步增强 aggressive threshold**：
   ```python
   # 随训练进度提高 threshold
   aggressive_threshold = 0.05 + (iteration / max_iterations) * 0.05
   ```
   

2. **结合 opacity regularization** (Method 2)：
   ```python
   # 在 loss 中添加小权重的正则项
   outside_gaussians = compute_gaussian_mask_overlap(gaussians, cameras[i], outside_mask)
   opacity_reg = gaussians.get_opacity[outside_gaussians].mean()
   loss = loss + 0.01 * opacity_reg
   ```

3. **调整 densification 频率**：
   - 当前：每 100 iterations
   - 可以尝试：每 50 iterations（更频繁的 pruning）

---

*文档生成日期: 2025-10-13*
*实现方案: Method 1 from mask_loss_failure_analysis.md*
*状态: 已实现，待测试*
