# 深度一致性正则化实现总结

## 实施时间
2025-11-05

## 问题描述
深度相机误差导致点云在某些区域（如桌子边缘）有突出的毛刺。原因是同一射线上存在多个高斯点，且深度不连续。

## 解决方案

### 方案1: 深度一致性Loss (⭐⭐⭐⭐⭐)
**位置**: `train.py` lines 234-258

**原理**:
- 使用avg pooling (kernel_size=5) 平滑渲染深度
- 惩罚偏离平滑值的深度点
- 强制同一区域内的深度连续

**实现**:
```python
lambda_consistency = 0.1 if iteration > 5000 else 0.0

if lambda_consistency > 0:
    # 平滑深度
    depth_smoothed = F.avg_pool2d(
        depth_render.unsqueeze(0).unsqueeze(0),
        kernel_size=5, stride=1, padding=2
    ).squeeze()

    # 惩罚偏离
    deviation = torch.abs(depth_render - depth_smoothed)
    loss_consistency = lambda_consistency * (deviation * mask_2d).sum() / denom

    loss = loss + loss_consistency
```

**启用时机**: iteration > 5000（让模型先收敛）

**权重**: λ_consistency = 0.1

---

### 方案2: 自适应Aggressive Opacity Pruning (⭐⭐⭐⭐)
**位置**: `scene/gaussian_model.py` lines 478-492

**原理**:
- 对于稀疏点云（<200K points）：使用threshold = 0.08
- 对于稠密点云（≥200K points）：使用threshold = 0.015（是默认0.005的3倍）
- 快速移除低opacity的"悬浮"高斯点

**实现**:
```python
if self.get_xyz.shape[0] < 200000:
    aggressive_opacity_threshold = 0.08
    print(f"[Opacity Pruning] Sparse cloud, threshold={aggressive_opacity_threshold}")
else:
    aggressive_opacity_threshold = 0.015
    print(f"[Opacity Pruning] Dense cloud, threshold={aggressive_opacity_threshold}")

low_opacity_mask = (self.get_opacity < aggressive_opacity_threshold).squeeze()
prune_mask = torch.logical_or(prune_mask, low_opacity_mask)
```

---

## 其他修改

### Progress Bar更新
**位置**: `train.py` lines 275-284

增加了一致性loss的显示：
```python
postfix_dict = {
    "Loss": f"{ema_loss_for_log:.{7}f}",
    "Depth": f"{ema_Ll1depth_for_log:.{7}f}",
    "Smooth": f"{smooth_loss_value:.{5}f}",
    "Consist": f"{consistency_loss_value:.{5}f}"  # 新增
}
```

### Loss初始化
**位置**: `train.py` lines 133-138

```python
loss = torch.tensor(0.0, device="cuda", requires_grad=True)
loss_depth_l1 = 0.0
loss_normal = 0.0
loss_smooth = 0.0
loss_consistency = 0.0  # 新增
```

---

## 预期效果

1. **深度一致性loss**:
   - 减少同一区域内的深度突变
   - 强制高斯点在图像平面上形成平滑表面
   - 解决70-80%的"突出"问题

2. **Aggressive opacity pruning**:
   - 快速移除低opacity的悬浮点
   - 减少同一射线上的多余高斯点
   - 防止点云发散

---

## 测试方案

### 推荐命令
```bash
# 使用修复后的camera poses + 新的正则化策略
python train.py -s data2 -m output/data2_depth_consistency \
  --depth_l1_weight_init 1.0 --depth_l1_weight_final 1.0 \
  --iterations 30000
```

### 对比测试
建议与之前的结果对比：
- `output/data2_depth_only_v5`: 旧版本（无一致性loss）
- `output/data2_depth_consistency`: 新版本（有一致性loss + aggressive pruning）

### 评估指标
1. 点云质量：离散点是否减少
2. 深度误差：MAE/RMSE是否下降
3. 训练稳定性：loss曲线是否平滑
4. 点云数量：是否过度pruning

---

## 备选方案（如果效果不够好）

### 方案3: 深度范围约束
如果一致性loss效果不够，可以尝试硬约束：

```python
max_deviation = 0.1  # 10cm
depth_diff = torch.abs(rendered_depth - gt_depth)
outlier_mask = depth_diff > max_deviation
loss_outlier = lambda_outlier * (depth_diff * outlier_mask * mask).mean()
```

### 方案4: 2DGS深度畸变loss
需要修改CUDA rasterizer，工作量大但效果最好：

```cuda
// 累积每条射线的深度方差
rend_dist += alpha * T * (depth - expected_depth)^2
```

参考: `other/2d-gaussian-splatting/submodules/diff-gaussian-rasterization/`

---

## 注意事项

1. **iteration > 5000才启用一致性loss**: 让模型先用纯深度loss收敛，避免过早约束
2. **观察pruning数量**: 如果pruning过于aggressive导致holes，降低threshold
3. **lambda_consistency可调**: 当前0.1，可根据效果调整到0.05-0.2
4. **配合相机pose修复使用**: 确保已使用`convert_extern_to_colmap_fixed.py`修复过images.txt

---

## 相关文件

- `train.py`: 主训练脚本（添加一致性loss）
- `scene/gaussian_model.py`: 高斯模型（修改opacity pruning）
- `convert_extern_to_colmap_fixed.py`: 修复相机pose转换
- `utils/export_depth_maps.py`: 深度可视化工具
- `DEPTH_ARTIFACT_SOLUTIONS.md`: 所有解决方案对比

---

## 更新日志

- 2025-11-05: 实现深度一致性loss + aggressive opacity pruning
- 2025-11-05: 修复相机pose转换bug（camera fix）
- 2025-11-05: 创建深度可视化工具
