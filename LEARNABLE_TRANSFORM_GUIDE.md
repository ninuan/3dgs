# 可学习坐标变换方案 - 使用指南

## 📋 方案说明

本方案通过**可学习的坐标变换参数**来对齐点云和相机坐标系，实现：
- ✅ **不修改原始点云文件** - init.ply保持完整
- ✅ **训练时自动对齐** - 通过深度loss学习变换参数
- ✅ **保持点云完整性** - 点数保持在原始数量附近
- ✅ **深度监督优化** - 深度loss提供有效的优化信号

## 🎯 适用场景

当您遇到以下情况时，应使用本方案：
1. **点云和相机位姿不匹配** - 坐标系统存在偏移/旋转
2. **有深度图监督** - 提供深度图和对应mask
3. **不想预处理点云** - 希望在训练中自动优化
4. **保持原始点云结构** - 不希望损失点云的完整性

## 🚀 使用方法

### 基本训练命令

```bash
python train.py -s data/ -d depth --depth_mask_dir mask \
    -m output/your_model \
    --iterations 30000 \
    --disable_viewer
```

### 参数说明

- `-s data/`: 数据目录
- `-d depth`: 使用深度监督
- `--depth_mask_dir mask`: mask目录名称
- `-m output/your_model`: 输出目录
- `--iterations 30000`: 训练迭代次数

## 📊 训练过程

### 1. 初始化阶段

```
[Coordinate Transform] Learnable transformation parameters initialized
  Rotation (quaternion): [1. 0. 0. 0]
  Translation: [0. 0. 0.]
  Scale (log): 0.0
Number of points at initialisation: 33404
```

- 变换参数初始化为单位变换（无旋转、无平移、无缩放）
- 点云保持原始状态

### 2. 训练阶段

训练过程中会显示：
```
Training progress: 10%|█| 300/3000 [Loss=0.0110, Depth Loss=0.0110, T=[0.084,0.117,-0.045]]
```

- **Loss/Depth Loss**: 深度误差，应逐渐降低
- **T=[x,y,z]**: 当前学习到的平移向量（每100次迭代显示）

### 3. 完成后输出

```
Learned Coordinate Transformation:
  Rotation (quaternion): [1.0153, -0.0211, -0.0291, 0.0123]
  Translation: [-0.5035, 0.0682, 0.3528]
  Scale: 0.9880 (log: -0.0121)
```

显示最终学习到的变换参数。

## 📈 效果监控

### 正常训练的标志

1. **点数保持稳定**:
   - 初始: ~33,000点
   - 最终: 30,000~40,000点之间

2. **深度loss下降**:
   - 初始: 0.02~0.03
   - 最终: 0.0001~0.0003

3. **变换参数收敛**:
   - 平移向量在前1000次迭代快速变化
   - 后续逐渐稳定

### 异常情况识别

❌ **点数骤降** (如降到个位数):
- 可能原因: mask constraint被错误启用
- 解决方法: 确认代码中 `gaussians.transform_optimizer is None` 的判断正确

❌ **深度loss不降低**:
- 可能原因: 学习率设置问题或坐标系差异过大
- 解决方法: 检查深度图单位，调整变换学习率

## ⚙️ 高级配置

### 调整变换学习率

在 `scene/gaussian_model.py` 的 `training_setup()` 方法中（第254-259行）：

```python
transform_params = [
    {'params': [self._transform_rotation], 'lr': 0.001, 'name': 'transform_rotation'},
    {'params': [self._transform_translation], 'lr': 0.01, 'name': 'transform_translation'},
]
if self._transform_scale is not None:
    transform_params.append({'params': [self._transform_scale], 'lr': 0.001, 'name': 'transform_scale'})
```

- **Rotation LR** (default: 0.001): 旋转学习率
  - 增大: 更快收敛，但可能不稳定
  - 减小: 更稳定，但收敛慢

- **Translation LR** (default: 0.01): 平移学习率
  - 通常需要比旋转更大的学习率

- **Scale LR** (default: 0.001): 缩放学习率
  - 如果点云尺度差异很大，可以增大

### 禁用某些变换

如果您确定只需要平移而不需要旋转/缩放，可以在 `create_from_pcd()` 中注释掉：

```python
# self._transform_rotation = None  # 禁用旋转
# self._transform_scale = None     # 禁用缩放
```

## 🔍 与Mask Constraint的关系

**重要**: 当使用可学习变换时，**mask constraint会自动禁用**。

### 原因

1. **变换学习需要时间**: 初期大部分点不在mask内是正常的
2. **原始点云的完整性**: 即使变换收敛，原始点云中仍有部分点不在mask内，但它们是物体的一部分
3. **深度loss足够**: 深度监督已经提供了足够的约束

### 代码逻辑 (train.py:201-204)

```python
valid_region_mask = None
if dataset.depth_mask_dir != "" and gaussians.transform_optimizer is None:
    # 只有在没有坐标变换时才使用mask约束
    valid_region_mask = compute_mask_constraint(gaussians, scene, render, pipe, background)
```

- 有变换: mask constraint = None (不裁剪点)
- 无变换: 使用mask constraint (裁剪mask外的点)

## 📝 输出文件

训练完成后，在输出目录会生成：

```
output/your_model/
├── point_cloud/
│   ├── iteration_7000/
│   │   └── point_cloud.ply      # 中间结果
│   └── iteration_30000/
│       └── point_cloud.ply      # 最终结果
├── cameras.json                 # 相机参数
└── cfg_args                     # 训练配置
```

**注意**: 保存的 `point_cloud.ply` 中的坐标是**原始坐标**（`_xyz`），不包含变换。渲染时会自动应用学习到的变换。

## 🐛 常见问题

### Q1: 为什么点云数量会略有增加？

**A**: 这是正常的densification过程：
- 原始: 33,404点
- 最终: 33,000~40,000点
- 增加的点来自split和clone操作

### Q2: 如何查看学习到的变换参数？

**A**: 训练结束时会自动打印：
```
Learned Coordinate Transformation:
  Rotation (quaternion): [...]
  Translation: [...]
  Scale: ...
```

也可以在checkpoint中查看这些参数。

### Q3: 变换参数是否会保存？

**A**: 是的，变换参数会随着模型一起保存在checkpoint中。加载模型时会自动恢复。

### Q4: 可以关闭深度loss吗？

**A**: 不建议。深度loss是变换学习的主要监督信号。如果关闭，变换参数无法正确学习。

## 💡 最佳实践

1. **第一次训练使用较少迭代** (如3000次) 来验证变换是否正确学习
2. **观察平移向量的变化趋势** - 应该在前1000次快速收敛
3. **确认最终点数** - 应该在原始点数的80%~120%范围内
4. **验证深度loss** - 应该降到0.001以下

## 📚 技术细节

### 变换公式

```python
xyz_transformed = scale * (R @ xyz) + translation
```

其中：
- `R`: 从四元数转换的3x3旋转矩阵
- `scale`: exp(scale_log)，确保为正
- `translation`: 3维平移向量

### 梯度流

```
深度图 → Depth Loss → backward() → 更新变换参数 → get_xyz应用变换 → 渲染 → 计算新的loss
```

变换参数通过独立的优化器更新，与点云位置、颜色等参数并行优化。

---

**版本**: 1.0
**最后更新**: 2025-10-11
**状态**: ✅ 已测试并验证
