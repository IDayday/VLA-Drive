# Field2Plan Phase 0：Tensor Contract

该契约锁定 commit `30505ee3a86326892f8be6c2cc04ca30ab18c93f` 上 193514 baseline 的真实接口。除“未来接口边界”一节外，所有 shape 都来自当前代码或本地数据检查；`B` 是 scene batch，`L` 是 Qwen token length，`V=3`，`H=8`。

## 1. 坐标与时间索引

- 当前帧：全局 pose/image index 3。
- 历史状态：index 2→3 的相对运动。
- future trajectory：index 4:12，共 8 个 waypoint。
- 每个 future waypoint 都表示在 current index 3 坐标系中的绝对未来 pose，不是前一 waypoint 到后一 waypoint 的逐步 delta。
- heading 被 wrap 到 `[-pi, pi)`，模型用 `(sin,cos)` 表示。
- 当前代码借助 NAVSIM `absolute_to_relative_poses(..., origin_index=3)` 得到 planning frame。相机外参只标记为 raw sensor-to-lidar metadata；在 Phase 1 明确验证前，不声明 lidar frame 与 planning ego frame 相同。

## 2. Dataset sample contract

生产者：`starVLA/dataloader/navsim_dataset.py`。当前 collate 结果是长度为 `B` 的 sample dict list，而不是把所有值自动 stack 成单个 dict。

| key | 单样本类型/shape | dtype | 语义 |
|---|---|---|---|
| `image` | list 长度 3，每项 PIL RGB 1024×576 | uint8 image | 源顺序 front/left/right，当前 index 3 |
| `state` | `[1,4]` | NumPy float32 | `[dx_n,dy_n,sin(dtheta),cos(dtheta)]`，index 2→3 |
| `action` | `[8,4]` | NumPy float32 | `[x_n,y_n,sin(theta),cos(theta)]`，future 4:12 relative to index 3 |
| `lang` | scalar string | string | 当前 command 构造的 planning prompt |
| `token` | scalar string | string | NAVSIM scene token |
| `action_copy` | `None`，或 `[8,4]` only S2 | NumPy float32 | legacy stage-2 path |
| `adv` | `None`，或 bool only S2 | bool | legacy stage-2 path |
| `qwen_feature_cache` | cache-dependent nested tensors | cache dtype | 可选 Qwen visual/deepstack cache |

193514 legacy sample 不含 `camera`、`proposal` 或 Field2Plan teacher keys。

Raw pickle 中已核实但尚未暴露给 sample 的字段：

| raw metadata | shape | 备注 |
|---|---:|---|
| per-view intrinsics | `[13,3,3]` | 对应 raw 1920×1080 pixel coordinates |
| per-view sensor-to-lidar rotations | `[13,3,3]` | frame semantics 需 Phase 1 明确转换 |
| per-view sensor-to-lidar translations | `[13,3]` | 与 rotation 配套 |
| per-view distortions | `[13,5]` | 当前 policy image path 未消费 |
| global ego poses | `[14,3]` | `(x,y,heading)` |

## 3. Action codec contract

常量定义在 `navsim_dataset.py:147-151`，逆变换重复定义在 `infer.py`：

```text
X_MEAN = 10.172484
X_STD  = 8.805105
Y_MEAN = 0.360762
Y_STD  = 2.277741
```

对任意前置 batch/candidate 维 `...`：

```text
physical trajectory: [...,8,3] = [x_m, y_m, theta_rad]
normalized action:   [...,8,4] = [x_n, y_n, sin(theta), cos(theta)]

x_n = (x_m - X_MEAN) / X_STD
y_n = (y_m - Y_MEAN) / Y_STD
theta_wrapped = (theta_rad + pi) % (2*pi) - pi

x_m = x_n * X_STD + X_MEAN
y_m = y_n * Y_STD + Y_MEAN
theta_rad = wrap_to_pi(atan2(sin(theta), cos(theta)))
```

约束：

- horizon 必须为 8，最后一维只能是 physical 3 或 normalized 4。
- baseline encoder 产生的 heading pair 是 unit norm；flow prediction 本身不强制 unit norm，decoder 用 `atan2`，因此不能在 parity path 中擅自 renormalize。
- `wrap_to_pi(pi) == -pi`。
- `act_norm=1` 是 193514 contract；旧 `act_norm=0` 不属于本 baseline。
- zero physical delta 的未来 refiner 定义应使 final 与 draft 完全相同；Phase 0 只记录要求，不提供实现。

## 4. Qwen multimodal contract

入口：`QwenOFT._build_qwen_batch` 与 `QwenOFT._qwen_language_forward`。

| tensor | shape | dtype/device | 生产者 → 消费者 |
|---|---:|---|---|
| `input_ids` | `[B,L]` | int64 / model device | processor → Qwen text model |
| `attention_mask` | `[B,L]` | integer/bool / model device | processor → Qwen text model |
| `position_ids` | Qwen3 mRoPE layout | int64 / model device | custom preparation → Qwen text model |
| `image_grid_thw` | `[B*3,3]` | integer / model device | processor → custom visual model |
| merged image embeddings | `[N_visual,2048]` | bf16 / model device | visual merger → text input replacement |
| deepstack embeddings | list 长度 3，每项 `[N_visual,2048]` | bf16 / model device | visual deepstack → early text layers |
| `last_hidden_state` | `[B,L,2048]` | bf16 / model device | Qwen language model → query gather |
| action positions | `[B,8]` | int64 / same device | prompt token scan → gather |
| action queries | `[B,8,2048]` | model dtype/device | gather → action head |

Vision config contract：patch 16、spatial merge 2、vision hidden 1024、merged output 2048、vision depth 24、deepstack indexes `[5,11,17]`。

自定义 `vit_tokens_to_featmap` 的当前 shape：

```text
input hidden_states: [N_visual,2048]
input grid_thw:      [B*3,3]
per-view output:     [B,3,2048,Hm,Wm]
horizontal output:  [B,2048,Hm,3*Wm]
```

`Hm/Wm` 必须由每个真实 `grid_thw` 与 spatial merge size 计算，不能从 resize 尺寸硬编码。helper 默认把 source front/left/right 重排成 left/front/right。

## 5. Flow action head contract

入口：`starVLA/model/modules/action_model/GR00T_ActionHeader.py:276`。

### Training

```text
scene batch                                           B
repeated_diffusion_steps                              R=8
B'                                                    R*B
vl_embs                                               [B',8,2048]
ground-truth normalized actions                       [B',8,4]
noise/noisy trajectory/velocity target                [B',8,4]
t continuous                                          [B',1,1]
t discretized                                         [B']
action features                                       [B',8,1536]
projected VLM context                                 [B',8,1536]
DiT/decoder output                                    [B',8,4]
action loss                                           scalar
```

Flow interpolation is：

```text
z_t = (1-t) * epsilon + t * action
velocity_target = action - epsilon
loss = mean((velocity_prediction - velocity_target)^2)
```

### Inference

```text
initial action                                        [B,8,4] ~ N(0,I)
num_inference_timesteps                               10
dt                                                    0.1
returned normalized action                           [B,8,4]
```

没有 seed 参数传入 head；全局 torch RNG 决定初始轨迹。

## 6. Loss contract

Baseline framework 输出 flat dict，每个 loss 均为 scalar tensor：

| key | 193514 enabled | framework 内 scale |
|---|---:|---:|
| `action_loss` | yes | 1.0 |
| `rgb_loss` | yes | 1.0，且包含启用的 rgb query loss |
| `gs_loss` | yes via `w_depth=1` | return 时乘 0.1，且包含启用的 gs query loss |
| `reward_loss` | no | 1.0 |

Trainer 的 baseline 聚合是按开关直接相加，不读取 `trainer.loss_weights`：

```text
total_loss_193514 = action_loss + rgb_loss + gs_loss_returned
```

Phase 1 的结构化 `output_dict["losses"]` 聚合只能是向后兼容的新分支。legacy output 不含 `losses` 时必须继续精确使用上述逻辑。

## 7. Inference/evaluator contract

| 阶段 | shape | dtype | 格式 |
|---|---:|---|---|
| framework result | `[B,8,4]` | NumPy, observed float32 | `normalized_actions` |
| `deal_action_1225` result | `[B,8,3]` | NumPy | `[x_m,y_m,heading_rad]` |
| single prediction file | `[8,3]` | float32 | `{token}.npy`, `allow_pickle=False` compatible |
| evaluator split | 12146 files | — | `navtest`, human agent |

`infer.py` 用同目录临时文件、flush、fsync、`os.replace` 原子提交单 token prediction。未来诊断文件不能代替或改变这一 evaluator-facing 文件。

## 8. Camera contract 的已知缺口

Phase 0 不新增 camera tensor。Phase 1 在保持 legacy keys 的前提下新增 `sample["camera"]` 时，至少需要显式携带：view names、current frame index、K、rotation、translation、raw image size、resized image size、外参 source/destination frame 名称和 validity。必须先完成以下检查：

1. raw K 从 1920×1080 到 1024×576 的缩放；
2. image crop/resize 是否只有 resize；
3. sensor-to-lidar 与 planning ego frame 的变换链；
4. source front/left/right 与 feature tap left/front/right 的 permutation；
5. 投影点在图像前方且 pixel/grid 坐标约定一致。

在这些检查通过前，不应声称已有 `[B,V,3,3]` K 可以直接用于 geometry writer。

## 9. 未来 Field2Plan 接口边界（尚未实现）

以下只用于阻止 Phase 1 接口漂移，不表示当前代码已经提供：

- visual feature map：`[B,V,C,Hf,Wf]`；
- geometry field：`[B,Cg,Ny,Nx]`；
- draft normalized action：`[B,M,8,4]` 或 M=1 简写 `[B,8,4]`；
- draft physical trajectory：`[B,M,8,3]`；
- per-waypoint tube readout：`[B,M,8,Cr]`；
- final normalized action：与 draft normalized action shape 相同。

这些新 tensor 必须继承输入的 device/dtype，只有 projection/grid math 显式使用 float32；forward 内不得动态创建 module/parameter。World writer 不允许读取 GT future action。
