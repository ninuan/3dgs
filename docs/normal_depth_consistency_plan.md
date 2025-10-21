# 法向-深度一致性改进方案

## 目标
利用深度监督的同时,添加法向-深度一致性约束,减少点云发散

## 核心改进

### 1. 添加depth2normal计算 ✅ 高优先级

**位置**: `utils/image_utils.py` (新增)

```python
def depth2normal(depth, mask, camera):
    """
    从深度图计算几何法向

    Args:
        depth: [1, H, W] 深度图
        mask: [1, H, W] 有效区域mask
        camera: Camera对象

    Returns:
        normal: [3, H, W] 归一化法向图
    """
    # 1. 深度转相机空间3D位置
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device

    # 2. 像素坐标网格
    h, w, _ = torch.meshgrid(
        torch.arange(0, shape[0]),
        torch.arange(0, shape[1]),
        torch.arange(0, shape[2]),
        indexing='ij'
    )
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)

    # 3. 反投影到相机空间
    # p -= principal_point
    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD

    # 内参矩阵的逆
    from utils.graphics_utils import fov2focal
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([[K00, 0], [0, K11]]).to(device)
    Kinv = torch.inverse(K)

    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)  # [H, W, 3]

    # 4. 填充边界,用于计算梯度
    p = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask = torch.nn.functional.pad(mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)

    # 5. 计算4个方向的向量差分
    p_c = (p[:, 1:-1, 1:-1, :]) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:,  :-2, 1:-1, :] - p_c) * mask[:,  :-2, 1:-1, :]  # 上
    p_l = (p[:, 1:-1,  :-2, :] - p_c) * mask[:, 1:-1,  :-2, :]  # 左
    p_b = (p[:, 2:  , 1:-1, :] - p_c) * mask[:, 2:  , 1:-1, :]  # 下
    p_r = (p[:, 1:-1, 2:  , :] - p_c) * mask[:, 1:-1, 2:  , :]  # 右

    # 6. 叉乘计算4个法向,取平均
    n_ul = torch.cross(p_u, p_l, dim=-1)
    n_ur = torch.cross(p_r, p_u, dim=-1)
    n_br = torch.cross(p_b, p_r, dim=-1)
    n_bl = torch.cross(p_l, p_b, dim=-1)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]  # 去掉batch维度

    # 7. 归一化
    mask = mask[0, 1:-1, 1:-1, :]
    n = torch.nn.functional.normalize(n, dim=-1)
    n = (n * mask).permute([2, 0, 1])  # [3, H, W]

    return n
```

### 2. 在训练中添加法向-深度一致性loss ✅ 高优先级

**位置**: `train.py` 训练循环中

```python
# 在计算RGB loss后添加

# 1. 渲染深度和mask
render_depth = render_pkg["depth"]  # 你已经有了
mask_vis = (render_pkg["opac"] > 1e-5).float()

# 2. 从深度计算几何法向
from utils.image_utils import depth2normal
d2n = depth2normal(render_depth, mask_vis, viewpoint_cam)

# 3. 如果你的渲染器支持法向输出,计算一致性loss
# 如果没有法向输出,可以从深度梯度估计
if "normal" in render_pkg:
    render_normal = render_pkg["normal"]
    render_normal = torch.nn.functional.normalize(render_normal, dim=0) * mask_vis

    # 法向-深度一致性loss (余弦距离)
    cos_sim = torch.sum(render_normal * d2n, dim=0, keepdim=True)
    loss_normal_depth = (1 - cos_sim[mask_vis > 0]).mean()
else:
    # 如果没有法向,从渲染深度计算梯度作为隐式法向约束
    # 这是一个简化版本
    pass

# 4. 添加到总loss
loss += 0.01 * loss_normal_depth  # 权重从小开始(0.01-0.1)
```

### 3. loss权重调度 ✅ 推荐

```python
# 逐渐增加法向一致性权重,在train.py中
iteration_ratio = iteration / opt.iterations

# 从0.01逐渐增加到0.1
loss_weight_normal = 0.01 + 0.09 * min(iteration_ratio * 2, 1.0)
loss += loss_weight_normal * loss_normal_depth
```

### 4. 可选: 曲率平滑loss ⚠️ 谨慎使用

```python
def normal2curv(normal, mask):
    """计算法向变化率(曲率)"""
    n = normal.permute([1, 2, 0])
    m = mask.permute([1, 2, 0])
    n = torch.nn.functional.pad(n[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    m = torch.nn.functional.pad(m[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)

    n_c = n[:, 1:-1, 1:-1, :] * m[:, 1:-1, 1:-1, :]
    n_u = (n[:,  :-2, 1:-1, :] - n_c) * m[:,  :-2, 1:-1, :]
    n_l = (n[:, 1:-1,  :-2, :] - n_c) * m[:, 1:-1,  :-2, :]
    n_b = (n[:, 2:  , 1:-1, :] - n_c) * m[:, 2:  , 1:-1, :]
    n_r = (n[:, 1:-1, 2:  , :] - n_c) * m[:, 1:-1, 2:  , :]

    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1]) * mask
    curv = curv.norm(dim=0, keepdim=True)
    return curv

# 在训练中
loss_curv = (curv * mask_vis).mean()
loss += 0.005 * loss_curv  # 很小的权重
```

## 预期效果

1. **减少点云发散**: 法向-深度一致性提供额外的几何约束
2. **更平滑的表面**: 深度信息隐式约束点云形成连续表面
3. **更好的泛化**: 几何先验帮助模型在新视角表现更好

## 实施优先级

1. ✅ **第一步**: 实现`depth2normal`函数
2. ✅ **第二步**: 添加基本的法向-深度一致性loss
3. ⚠️ **第三步** (可选): 添加曲率平滑loss
4. ⏸️ **暂不考虑**: Monocular法向先验(需要额外模型)

## 注意事项

1. **权重调优**: 从小权重(0.01)开始,观察点云质量
2. **mask处理**: 确保只在有效深度区域计算loss
3. **法向输出**: 如果渲染器不支持法向,需要修改rasterizer
4. **调试**: 保存d2n可视化,确保计算正确

## 与现有深度loss的关系

- **深度loss**: 直接约束深度值 → 控制点云位置
- **法向-深度一致性**: 约束深度梯度 → 控制点云法向/方向
- **互补性**: 两者结合提供完整几何约束

建议loss配置:
```python
loss_depth = l1_loss(render_depth * mask, gt_depth * mask)
loss_normal_depth = (1 - cos(render_normal, d2n)) * mask

loss = loss_rgb + 0.1 * loss_depth + 0.05 * loss_normal_depth
```
