# VGGT V2 完整算法与实现契约

## 1. 一句话说明

V2 是一个独立、端到端训练的轨迹规划算法：VGGT 只在离线阶段为当前三视图生成 teacher
特征和物理几何目标；训练后的 Qwen 自己构造几何记忆，8 个 action query 从这份记忆中定向
读取，再由原 FlowMatching ActionHead 直接生成完整轨迹。它不加载 baseline planner 权重，不
生成 draft，也不预测相对 draft 的 residual trajectory。

本版相对 V1 的改动不是单独的“最后层换成 layer 11”，而是下列设计的组合。`layer 11
global` 只定义 teacher 来源。

## 2. Teacher：layer 11 pure-global，而不是最终层拼接特征

VGGT Aggregator 为 DPT 缓存的层特征由 frame branch 和 global branch 拼接：

```text
aggregated[11]: [B,3,1374,2048]
                       = frame 1024 + global 1024
pure global:    [B,3,1374,1024] = aggregated[11][...,1024:]
```

V2 只使用 `global_blocks[11]` 对应的 1024 维分支。它已完成低成本 geometry probe；本轮不
新增 LiDAR/occupancy teacher、不做多层融合，也不切换其他 VGGT 变体。

VGGT、Depth/Point DPT heads 只在离线 cache 脚本中运行。训练和推理 forward 不 import
VGGT，也不会在线执行 teacher。

## 3. 15 个文本 query 与 195 个几何 memory 的区别

### 3.1 Global 分支：15 个 Qwen 文本 token

每个 view 保留 VGGT 的 5 个 special/global token：

```text
3 views × 5 = 15
Qwen global query hidden: [B,15,2048]
```

只有这 15 个 token 加入 tokenizer 和 Qwen 语言序列。它们负责跨视角整体关系、状态/指令
和几何语义融合。提示词中 action query 位于 global query 之前，因此 causal Qwen 不能让
action hidden 偷看 global query；VGGT memory 进入 ActionHead 的唯一入口仍是 waypoint reader。

### 3.2 Spatial 分支：180 个视觉 memory slot

180 不是文本 token 数量。空间表示直接从同一次 Qwen forward 的 image-placeholder hidden
state 提取，不二次运行 visual encoder：

```text
Qwen visual language hidden
  -> 按 image_grid_thw 与 spatial_merge_size 恢复每个 view 的二维 map
  -> adaptive pool 到 6×10
  -> [B,3,6,10,2048]
  -> flatten [B,180,2048]
```

Teacher 侧对 37×37 patch map 先裁掉 VGGT pad 区域，再 pool 到 6×10：

```text
layer11 global patches [B,3,37,37,1024]
  -> content crop
  -> [B,3,6,10,1024]
  -> flatten [B,180,1024]
```

这将 V1 每视角 4×4、实际约 2×4 有效格子的过度压缩改成每视角 60 个有效格子。

### 3.3 Shared Geometry Memory

15 个 global hidden 和 180 个 spatial hidden 先拼接，再经过同一个 GeometryAdapter：

```text
[B,15,2048] + [B,180,2048]
  -> SharedGeometryAdapter
  -> raw memory [B,195,1024]
  -> alignment normalization/projection
  -> planning memory [B,195,1024]
```

对齐 loss 的输出表示就是 planning memory；物理 probe 和 waypoint reader 也读取这份表示。
V1 中“projection 只用于 alignment，planner 却读取 raw query”的分叉已移除。

## 4. 训练监督

### 4.1 Raw feature alignment

Scene-residual 训练默认关闭，先检验提高空间分辨率后 raw teacher 是否能产生场景差异：

```yaml
alignment:
  mode: raw
  scene_residual_enabled: false
  log_scene_residual_metrics: true
```

Global 15 slots 和 spatial 180 slots 分开形成 loss，避免数量更多的 spatial slots 淹没 global
信号。跨场景 relation loss 使用全局分布式 batch 的 scene similarity，而不是只计算每卡 batch
size 2 的随机 top-1。

Cache 同时原子保存 train split 的 per-slot mean/scale/variance。它们目前只用于
slot-template、scene-residual、shuffled teacher 等无梯度诊断，不参与主要优化目标。

### 4.2 Physical geometry probe

离线 VGGT depth/point heads 为 180 个空间位置保存：

```text
x/z, y/z, log(depth / scene_median_depth): [180,3]
confidence:                              [180]
valid mask:                              [180]
```

共享 planning memory 的 spatial 部分经小 MLP 预测这三个量，使用 confidence-weighted Huber
loss。这个 head 不参与最终推理输出，其作用是回答“memory 是否真的学到可线性读取的几何”，
而不是只看容易被固定槽位模板抬高的高维 cosine。

### 4.3 Auxiliary trajectory head

Waypoint reader 的 8 个 readout 通过训练期小 MLP 直接回归完整 normalized GT action：

```text
[B,8,2048] -> [B,8,4] = x,y,sin(theta),cos(theta)
```

它使用 Huber loss，同样不替代最终 FlowMatching 输出。若 geometry probe 已学好但 auxiliary
trajectory 学不好，可以将问题定位到 reader；若 auxiliary head 学好但最终规划不改善，则
问题位于 ActionHead 对 readout 的利用。

## 5. 规划如何使用 VGGT 知识

V2 只保留一条显式入口：

```text
action queries    [B,8,2048] -- Query --> WaypointGeometryReader
geometry memory [B,195,1024] -- Key/Value -----------^
                                      |
                                      v
geometry readout   [B,8,2048]
```

ActionHead 接口为：

```text
vl_embs:       action queries  [B,8,2048]
extra_context: waypoint readout[B,8,2048]
```

内部只拼成 `[B,16,2048]`。V1 的两条旁路均已删除：

1. reader 不再用 scalar gate/residual 直接修改 action query；
2. 完整 195-slot raw memory 不再作为 `extra_context` 进入 DiT。

因此所有进入 ActionHead 的 VGGT 信息都必须经过 8-waypoint reader，attention 和干预实验才
具备明确含义。

## 6. Mask 与训练/推理一致性

同一份 valid/confidence contract 控制：

- feature alignment；
- physical geometry loss；
- waypoint cross-attention key padding。

缺失、低方差或无物理置信度的 teacher slot 不会被静默当成有效监督。训练 cache 严格
fail-fast；推理不读取 teacher cache，Qwen 当前三视图构造相同的 15+180 布局。

## 7. 总损失与学习率

```text
L = 1.00 L_flow
  + 0.05 L_global_alignment
  + 0.10 L_spatial_alignment
  + 0.05 L_scene_relation
  + 0.10 L_physical_geometry
  + 0.05 L_aux_plan
```

辅助 loss 在前 1000 optimizer steps 从 0 线性增加到目标权重。Qwen 和 ActionHead 保持
`1e-5`；随机初始化的 GeometryAdapter、aligner、reader、geometry probe 和 auxiliary head
使用 `3e-5`。正式配置保持有效 batch 32。

## 8. 训练期归因诊断

表示层记录：

- raw global/spatial cosine；
- scene-residual cosine（仅诊断）；
- correct minus slot-mean / shuffled margin；
- distributed retrieval top-1/top-5；
- student/teacher 跨场景方差比。

几何和 reader 记录：

- `x/z`、`y/z`、relative-depth MAE；
- special/spatial/front/left/right attention mass；
- 每个 waypoint entropy、相邻 waypoint JS divergence；
- geometry readout norm；
- auxiliary ADE/FDE；
- adapter/alignment/reader/probe/aux head gradient norm。

每 500 optimizer steps，在同一个 batch 和相同 RNG seed 下比较：

```text
real / zero / batch-shuffled / slot-mean geometry memory
```

记录各自 flow loss、ADE/FDE/heading error、相对 real 的轨迹 L2 变化。这个检查用于判断最终
规划是否对具体场景 geometry memory 具有因果敏感性，不用 attention 变尖锐替代“利用
有效”的证据。

## 9. 归因顺序

```text
raw/跨场景对齐是否超过模板？
  -> physical geometry probe 是否学好？
  -> auxiliary trajectory 是否学好？
  -> real/zero/shuffled 干预是否改变最终 flow loss？
  -> NAVSIM 指标是否优于相同训练预算的基线？
```

这样可以区分 teacher 信号不足、student 表示没学好、reader 没利用、ActionHead 没利用，以及
teacher 知识与 NAVSIM 目标不匹配这五类问题。

## 10. 代码入口与 tensor shape

| 入口 | 契约 |
|---|---|
| `tools/precompute_vggt_query_cache.py` | layer11 pure-global `[195,1024]` + physical `[180,3]` |
| `geometry_memory.py` | Qwen visual hidden → spatial `[B,180,2048]`; shared memory `[B,195,1024]` |
| `alignment.py` | raw alignment、distributed relation/retrieval、template diagnostics |
| `planning_heads.py` | physical probe、8-waypoint reader、auxiliary trajectory head |
| `QwenOFT_VGGT.py` | 独立 V2 framework；ActionHead 只接收 8 readouts |
| `vggt_query_main.yaml` | 完整正式训练配置 |

## 11. 当前有意不做的事项

- 不启用 Scene-residual loss，只保留配置和诊断；
- 不增加 LiDAR/occupancy 训练监督；
- 不做多层 VGGT 融合或 VGGT-Omega；
- 不做 baseline draft、trajectory residual 或 proposal cache；
- 不在训练/推理中在线运行 VGGT；
- 不把 GT future trajectory 输入 teacher/world representation。
