# VGGT Dense Spatial-Forcing Alignment Design

## 1. Goal

当前方案的目标是把 VGGT 的 3D 几何知识注入到现有 Qwen3VL action planning 框架中，但不再采用上一版的 `VGGT query token -> hidden state -> teacher feature` 对齐方式。

新的方案参考 Spatial-Forcing：

- 不新增 VGGT special query token。
- 不让 VGGT token 直接进入 action DiT context。
- 不预测 latent CoT。
- 使用 VGGT 未压缩 dense patch-level hidden feature 作为 teacher。
- 对齐 Qwen/VLM 中间层的 image token hidden states 与 VGGT dense teacher feature。
- action 仍由原有 action token hidden states 驱动 DiT 生成轨迹。

因此它是一个辅助表示学习目标：让 VLM 的视觉中间表征显式吸收 VGGT 的几何特征。

## 2. Relation To Spatial-Forcing

Spatial-Forcing 的核心训练逻辑是：

```python
vla_hidden = output.hidden_states[vla_layer]
vision_hidden = vla_hidden[:, vision_token_start:vision_token_end, :]

vggt_output = vggt(images)
agg_vggt_hidden = vggt_output["features"][vggt_layer]
vggt_hidden = agg_vggt_hidden[:, :, patch_start_idx:, :]

vggt_hidden = custom_pooling(
    vggt_hidden,
    patch_hw,
    image_hw,
    vision_hidden,
    pooling_func,
    use_vggt_pe,
)

align_loss = align_projector(vision_hidden, vggt_hidden)
```

当前仓库的移植版本保持同一范式：

```text
Qwen 中间层 image token hidden
        vs
VGGT dense patch hidden teacher
```

主要差异是：

- Spatial-Forcing 在线运行 VGGT；当前方案使用离线 `vggt_dense` cache。
- Spatial-Forcing 的 pooling 是按 token 数比例做 bilinear resize；当前方案利用 Qwen `image_grid_thw` 做 per-view 2D pooling，更适配 Qwen3VL 的多视角 token layout。
- 当前方案同时保留原 action DiT 规划路径，不直接把 VGGT feature 作为 DiT context。

## 3. Dense VGGT Cache

当前方案需要未压缩 VGGT dense feature cache，而不是之前的 `[195, 1024]` query/global 压缩 cache。

新增 cache component：

```text
vggt_dense
```

dataset 读取后会放入：

```python
sample["vggt_dense_feature_cache"]
```

每个样本 payload 约定：

```text
features:        bf16 [num_views, num_patches, 2048]
valid_mask:      bool [num_views, num_patches]
patch_hw:        long [2]
image_hw:        long [2]
patch_start_idx: long [1]
layer_index:     long [1]
```

其中 `2048 = 2 * VGGT embed_dim`，因为 VGGT aggregator 输出会 concat frame/global intermediates：

```python
concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
```

默认 VGGT 输入尺寸为 `518 x 518`，patch size 为 `14`，因此每个 view 有：

```text
37 * 37 = 1369 patches
```

三视角样本的 teacher feature 通常为：

```text
[3, 1369, 2048]
```

## 4. Cache Generation

新增脚本：

```bash
tools/precompute_vggt_dense_cache.py
```

权重路径通过环境变量或 CLI 指定：

```bash
export VGGT_MODEL=/path/to/VGGT-1B/model.pt
```

示例命令：

```bash
source load_env.sh
export VGGT_MODEL=/path/to/VGGT-1B/model.pt

torchrun --standalone --nnodes=1 --nproc-per-node=8   tools/precompute_vggt_dense_cache.py   --cache-root /mnt/workspace/VLA-Drive/navsim_feature_cache/vggt_dense_train_layer-1   --datalist /mnt/workspace/VLA-Drive/train_meta.json   --data-root /mnt/workspace/VLA-Drive/navsim_dataset   --split train   --layer-index -1   --batch-size 1   --map-size-gb 512   --overwrite
```

脚本行为：

- 使用 `Spatial-Forcing/openvla-SF` 中的 VGGT 实现。
- 初始化 VGGT 时关闭 camera/point/depth/track heads，仅保留 feature extractor。
- 输入三视角 PIL image，resize 到 `518 x 518`，转为 `[B, 3, 3, 518, 518]`。
- 取 `vggt_output["features"][layer_index]`。
- 去掉 camera/register special token，仅保存 patch tokens。
- 每个 rank 写一个 LMDB，rank 0 写 manifest。

## 5. Model-Side Alignment Path

### 5.1 Qwen Hidden States

本地 Qwen3VL text model 已补充 `output_hidden_states` 支持。

训练时如果启用：

```yaml
framework.vggt_dense_align.enabled: true
```

则 Qwen language forward 会返回：

```python
last_hidden, qwen_hidden_states
```

否则仍只返回 `last_hidden`，避免额外内存开销。

### 5.2 VLM Layer Selection

配置项：

```yaml
framework:
  vggt_dense_align:
    vlm_layer_index: -1
```

`-1` 表示最后一层 hidden state。也可以设置为中间层，例如 `12`、`16`、`20` 等。

实际选择逻辑：

```python
align_hidden = qwen_hidden_states[vlm_layer_index]
```

### 5.3 Image Token Extraction

从指定层 hidden state 中，根据 Qwen image token id 抽取视觉 token：

```python
image_mask = input_ids.eq(image_token_id)
visual_hidden = hidden_state[image_mask].view(B, N, H)
```

其中：

```text
visual_hidden: [B, num_qwen_image_tokens, 2048]
```

这些 token 是 Qwen 中间层的视觉位置 hidden states，已经处在 VLM decoder 表征空间里。

## 6. VGGT Teacher Resampling

VGGT cache 中的 teacher feature 是 per-view patch grid：

```text
features: [V, P, D]
patch_hw: [patch_h, patch_w]
```

当前实现会根据 Qwen `image_grid_thw` 和 Qwen vision `spatial_merge_size` 计算每个 view 对应的 Qwen image-token grid：

```python
target_h = grid_h // spatial_merge_size
target_w = grid_w // spatial_merge_size
```

然后对每个 view 执行 2D pooling：

```text
VGGT [D, patch_h, patch_w]
    -> adaptive_avg_pool2d(target_h, target_w)
    -> flatten 成 token sequence
```

所有 view 拼接后得到：

```text
teacher_hidden: [num_qwen_image_tokens, 2048]
```

如果由于图像尺寸或 processor 细节导致 token 数仍不一致，会 fallback 到 1D interpolation 对齐长度。

## 7. VGGT Positional Encoding

当前已实现 Spatial-Forcing 中的 VGGT positional embedding 分支。

配置：

```yaml
framework:
  vggt_dense_align:
    use_vggt_pe: true
    vggt_pe_ratio: 0.1
```

逻辑：

1. 为 VGGT patch grid 构造归一化 2D UV grid。
2. 将 UV grid 转换为 sin-cos positional embedding。
3. 乘以 `vggt_pe_ratio`。
4. 在 pooling 之前加到 VGGT teacher feature map 上。

形式：

```python
view_feat = view_feat + PE_2D * 0.1
```

这和 Spatial-Forcing 的 `_apply_pos_embed` 设计一致。

## 8. Alignment Projector

新增 projector：

```python
SpatialForcingAlignProjector
```

结构：

```text
LayerNorm 可选
Linear(vlm_dim, teacher_dim)
GELU
Linear(teacher_dim, teacher_dim)
```

默认：

```text
vlm_dim = 2048
teacher_dim = 2048
use_vlm_norm = false
```

这与 Spatial-Forcing 的 projector 等价：

```python
fc1: llm_dim -> 2 * vggt_dim
fc2: 2 * vggt_dim -> 2 * vggt_dim
```

因为 VGGT `embed_dim=1024`，而实际 teacher hidden dim 是 `2 * embed_dim = 2048`。

## 9. Alignment Loss

对齐 loss 使用 cosine loss：

```text
L_raw = mean(1 - cosine(projector(visual_hidden), vggt_teacher_hidden))
L_vggt_dense = λ * L_raw
```

其中：

```text
λ = framework.vggt_dense_align.loss_weight
```

默认：

```yaml
loss_weight: 0.1
```

训练总 loss：

```text
total_loss = action_loss + optional_agent_losses + vggt_dense_loss
```

如果当前脚本使用 `minimal_agent`，则仍包含 32 agent query 的 DINO/bbox/visibility 监督。若要严格 action-only，应把 prompt mode 改为：

```bash
--framework.action_prompt_mode minimal
```

## 10. Gradient Diagnostics

为了判断 VGGT 对齐是否真的服务规划，当前记录了三项梯度指标。

定义：

```text
G = Qwen 指定中间层 image token hidden states

g_A = ∂L_action / ∂G
g_V = ∂(λ L_vggt_dense) / ∂G
```

日志指标：

```text
vggt_dense_action_grad_norm
vggt_dense_align_grad_norm
vggt_dense_action_align_grad_cosine
```

含义：

```text
vggt_dense_action_grad_norm = ||g_A||_2
vggt_dense_align_grad_norm  = ||g_V||_2
vggt_dense_action_align_grad_cosine = (g_A · g_V) / (||g_A||_2 ||g_V||_2)
```

解释：

- cosine 接近 `+1`：动作目标和 VGGT 对齐目标在该视觉表征上的优化方向一致。
- cosine 接近 `0`：两者基本正交，对齐 loss 更像独立辅助监督。
- cosine 小于 `0`：两者存在冲突，对齐 loss 可能在抑制规划目标需要的视觉表征。

注意：`g_V` 基于 weighted loss，因此量级对应真实加入总 loss 的梯度。

## 11. Training Script

新增训练脚本：

```bash
8-train_vggt_dense_align.sh
```

默认启用：

```bash
--framework.vggt_dense_align.enabled true
--framework.vggt_dense_align.loss_weight 0.1
--framework.vggt_dense_align.vlm_layer_index -1
--framework.vggt_dense_align.teacher_dim 2048
--framework.vggt_dense_align.use_vlm_norm false
--framework.vggt_dense_align.use_vggt_pe true
--framework.vggt_dense_align.vggt_pe_ratio 0.1
```

cache 相关环境变量：

```bash
export NAVSIM_FEATURE_CACHE_ROOT="$NAVSIM_VGGT_DENSE_CACHE_ROOT"
export NAVSIM_CACHE_COMPONENTS="vggt_dense"
export NAVSIM_CACHE_STRICT=1
```

启动示例：

```bash
export NAVSIM_VGGT_DENSE_CACHE_ROOT=/mnt/workspace/VLA-Drive/navsim_feature_cache/vggt_dense_train_layer-1
bash 8-train_vggt_dense_align.sh
```

## 12. Expected Behavior

训练初期应重点观察：

```text
action_dit_loss
vggt_dense_loss
vggt_dense_loss_raw
vggt_dense_valid_tokens
vggt_dense_action_grad_norm
vggt_dense_align_grad_norm
vggt_dense_action_align_grad_cosine
```

合理现象：

- `vggt_dense_valid_tokens` 应稳定接近 Qwen image token 数。
- `vggt_dense_loss_raw` 初期通常不会非常低，因为是未压缩 dense feature 对齐。
- `vggt_dense_align_grad_norm` 不应长期远大于 `action_grad_norm`，否则可能压制规划学习。
- `vggt_dense_action_align_grad_cosine` 如果长期显著为负，需要降低 `loss_weight`、换 `vlm_layer_index`，或延后启用 VGGT loss。

## 13. Current Limitations

当前版本还有几个需要实验确认的点：

- Qwen 对齐层默认 `-1`，但 Spatial-Forcing 通常对齐中后层；更合理的层可能是 decoder 中间层。
- 当前 teacher cache 固定 resize 到 `518 x 518`，与 Qwen processor 的真实视觉尺寸可能不同，因此依赖 per-view pooling 对齐 token grid。
- VGGT dense feature 只作为训练时辅助监督，推理时不会读取 VGGT cache，也不会增加推理成本。
- 当前方案不显式将 VGGT feature 注入 action DiT，只通过 VLM 中间视觉表征改变 action token hidden states。

## 14. Files Changed

主要代码入口：

```text
starVLA/model/framework/QwenOFT.py
starVLA/model/modules/vlm/qwen3_vl/modeling_qwen3_vl.py
starVLA/model/modules/vlm/qwen3_vl/modular_qwen3_vl.py
starVLA/training/train_starvla.py
starVLA/cache/navsim_feature_cache.py
starVLA/dataloader/navsim_dataset.py
starVLA/config/training/cfg_yaw_1225.yaml
8-train_vggt_dense_align.sh
tools/precompute_vggt_dense_cache.py
```
