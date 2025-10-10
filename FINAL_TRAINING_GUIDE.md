# 最终训练指南

## ✅ 数据集已就绪

您的data目录已经完全配置好，可以直接开始训练！

## 🚀 快速开始

```bash
# 1. 激活环境
conda activate gaussian_splatting

# 2. 开始训练（推荐命令）
python train.py -s data/ -d depth --depth_mask_dir mask -m output/my_scene --disable_viewer

# 3. 等待训练完成（默认30000次迭代）
# Loss会从初始值逐渐降低
```

## 📊 训练参数调整

### 基础参数
```bash
# 快速测试（1000次迭代）
python train.py -s data/ -d depth --depth_mask_dir mask --iterations 1000 --disable_viewer

# 完整训练（30000次迭代，默认）
python train.py -s data/ -d depth --depth_mask_dir mask --iterations 30000 --disable_viewer

# 使用eval模式（训练/测试分离）
python train.py -s data/ -d depth --depth_mask_dir mask --eval --disable_viewer
```

### 高级参数
```bash
# 调整深度loss权重
python train.py -s data/ -d depth --depth_mask_dir mask \
    --depth_l1_weight_init 1.0 \
    --depth_l1_weight_final 0.01 \
    --disable_viewer

# 指定输出路径
python train.py -s data/ -d depth --depth_mask_dir mask \
    -m output/experiment_001 \
    --disable_viewer

# 保存检查点
python train.py -s data/ -d depth --depth_mask_dir mask \
    --checkpoint_iterations 5000 10000 20000 \
    --disable_viewer
```

## 📁 输出文件

训练完成后，输出目录包含：

```
output/my_scene/
├── point_cloud/
│   ├── iteration_7000/
│   │   └── point_cloud.ply
│   └── iteration_30000/
│       └── point_cloud.ply
├── cameras.json
├── cfg_args
└── chkpnt*.pth (如果使用了checkpoint参数)
```

## 🔍 监控训练

训练过程中会显示：
```
Training progress:  50%|█████     | 15000/30000 [02:30<02:30, 100it/s, Loss=0.0003208, Depth Loss=0.0003208]
```

- **Loss**: 总损失
- **Depth Loss**: 深度损失（这是您主要关注的指标）
- **it/s**: 每秒迭代次数

## 🎯 训练成功标志

✅ Loss稳定下降  
✅ Depth Loss收敛到较小值（如0.0001-0.001）  
✅ 训练完成后保存模型文件  
✅ 无错误或警告信息

## 🛠️ 渲染和可视化

```bash
# 渲染训练结果
python render.py -m output/my_scene

# 计算评估指标
python metrics.py -m output/my_scene

# 查看渲染结果
ls output/my_scene/train/  # 训练视图渲染
ls output/my_scene/test/   # 测试视图渲染
```

## ⚠️ 常见问题

### Q: 端口被占用
```
OSError: [Errno 98] Address already in use
```
**解决**: 添加 `--disable_viewer` 参数

### Q: 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 
```bash
# 降低分辨率
python train.py -s data/ -d depth --depth_mask_dir mask -r 2 --disable_viewer

# 或减少densify频率
python train.py -s data/ -d depth --depth_mask_dir mask --densify_grad_threshold 0.0004 --disable_viewer
```

### Q: 训练太慢
**解决**:
```bash
# 减少迭代次数测试
python train.py -s data/ -d depth --depth_mask_dir mask --iterations 5000 --disable_viewer
```

## 📚 参考文档

- **数据集详情**: `data/TRAINING_GUIDE_zh.md`
- **修复记录**: `FIX_SUMMARY.md`
- **数据处理**: `DATASET_SUMMARY.md`

## 🎓 下一步

1. ✅ 数据已准备好
2. ✅ 代码已修复
3. ⏳ **开始训练** ← 您在这里
4. ⏳ 评估结果
5. ⏳ 调整参数优化

---

**准备状态**: ✅ 就绪  
**最后验证**: 2025-10-09 20:56  
**测试命令**: `python train.py -s data/ -d depth --depth_mask_dir mask --iterations 200 --disable_viewer`
