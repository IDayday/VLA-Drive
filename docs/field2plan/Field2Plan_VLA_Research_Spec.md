# Field2Plan-VLA：基于 DriveDreamer-Policy 的外部世界知识写入与轨迹条件化读取研究方案

> **文档类型**：研究方案与算法实现规格  
> **开发基线**：DriveDreamer-Policy 的纯轨迹规划分支  
> **建议论文定位**：ICLR oral 级机制研究，而非简单的多任务增量改进  
> **核心命题**：外部知识应先以保留空间/时间结构的方式写入特征场，再在形成轨迹假设后按需稀疏读取；不应在动作未知时过早压缩为固定场景 slots。  
> **暂定项目名**：Field2Plan-VLA  
> **核心实现形态**：**Dense/Structured Write → Draft-Conditioned Sparse Read → Access-Preserving Refine**

---

## 0. 执行摘要

本工作以已经复现稳定的 DriveDreamer-Policy 纯轨迹模型为起点，不再沿用“增加 depth/video 辅助头，然后期待 action query 自动吸收世界知识”的主要路线。

新的研究方案将外部模型知识的作用拆成三个可验证环节：

1. **Prior Writing：外部知识是否被写入学生模型**
   - VGGT/Depth-Anything 类先验用于几何特征场；
   - V-JEPA/Drive-JEPA 类先验用于时序动态特征场；
   - VLM 本身保留少量全局语义、规则与导航 token；
   - 不强迫异构 teacher 对齐到同一组固定 slots。

2. **Trajectory-Conditioned Reading：规划器是否能直接访问**
   - 先由原 DriveDreamer-Policy 规划器产生 draft trajectory；
   - 再沿 draft 的时空 swept tube 构造带坐标、时间、航向和速度条件的 queries；
   - queries 从几何场、动态场和语义 token 中读取与当前轨迹后果有关的信息；
   - slot/query 是读取接口，不是整个世界的唯一存储介质。

3. **Access-Preserving Refinement：外部知识是否改变最终规划**
   - 使用零初始化或近零初始化的轨迹 refinement 模块；
   - 初始状态严格退化为原始 baseline；
   - 通过访问开关、teacher 随机化、teacher shuffle、GT-MLP、等参数容量等控制，区分：
     - 外部知识；
     - 额外参数；
     - 未来 GT 深监督；
     - 普通正则化；
     - planner direct access。

### 最小可行研究路径

第一阶段不直接实现复杂的“每条候选轨迹对应一个反事实世界 rollout”，也不把 Wan/Cosmos 视频生成保留为核心模块。原因如下：

- 日志数据只观测 demonstrated action 对应的未来，无法为任意扰动轨迹提供真实反事实未来；
- flow-matching 的早期 noisy trajectory 不是有效物理轨迹，不适合直接做几何查询；
- 复杂的多候选 world rollout 会显著扩大归因空间。

因此，核心模型先采用：

```text
原 DriveDreamer-Policy action planner
                 │
                 ▼
          draft trajectory τ₀
                 │ stop-gradient / frozen proposal
                 ▼
   trajectory-tube sparse readout
                 │
                 ▼
       geometry/dynamics/semantic context
                 │
                 ▼
       lightweight trajectory refiner
                 │
                 ▼
          final trajectory τ*
```

后续再增加：

- 多 draft/candidate 读取与排序；
- flow denoising 后几步的 self-conditioned readout；
- 仅在具备 CARLA/Bench2Drive branch rollout 时增加 action-reactive world correction。

---

## 1. 研究问题与中心假设

### 1.1 研究问题

现有世界模型/外部模型融入 VLA 自动驾驶的方法通常存在以下问题：

- 外部监督改善了中间表示，但 planner 未必直接使用；
- 固定 slots 在动作未知时压缩整个世界，容易损失局部几何和动作相关信息；
- 几何、动态和语义 teacher 的知识拓扑不同，却常被强制映射到同一 token 空间；
- 训练阶段使用 GT future，推理阶段使用 predicted future，产生 oracle-conditioning gap；
- 结果提升可能来自额外参数、额外梯度或未来 GT 深监督，而非外部模型知识；
- 对一个总 PDMS/EPDMS 的回归可能退化为 metric emulator，而非世界理解。

### 1.2 中心假设

传统 scene-global early bottleneck：

\[
z=C(X),\qquad
p(Y_\tau\mid X,\tau)\approx p(Y_\tau\mid z,\tau),\ \forall\tau
\]

要求固定表示 \(z\) 对所有潜在动作都充分，信息保留要求过强。

本工作采用 action-conditioned late bottleneck：

\[
r_\tau=R(\mathcal M(X),\tau),\qquad
p(Y_\tau\mid X,\tau)\approx p(Y_\tau\mid r_\tau,\tau)
\]

其中：

- \(\mathcal M(X)\) 是保留结构的几何/动态特征场；
- \(\tau\) 是已经形成的 draft trajectory；
- \(r_\tau\) 只需保存当前轨迹后果相关的信息。

### 1.3 可证伪的主张

本工作不预设 late bottleneck 必然更优，而通过同代码、同模型规模、同数据、同种子进行以下直接比较：

1. fixed scene slots；
2. dense feature fusion；
3. dense feature + static expert slots；
4. proposed trajectory-conditioned sparse readout；
5. online teacher upper bound。

只有在 teacher-specific、access-specific、candidate/draft-specific 和闭环结果上同时成立，才主张外部知识被真正用于规划。

---

## 2. 对原方案的关键修正

### 2.1 不再让固定 slots 承担完整世界存储

固定 slots 仍可作为以下对象：

- 少量全局语义/规则 token；
- trajectory query；
- agent/lane/occupancy 的坐标锚定 query；
- 对 dense field 的稀疏读取接口。

固定 slots 不再作为：

- VGGT 高分辨率几何的唯一载体；
- V-JEPA 时序动态的唯一载体；
- 所有 teacher 的共享压缩空间；
- planner 唯一可访问的世界状态。

### 2.2 不再默认对每条候选轨迹生成完整未来世界

日志数据中只存在 demonstrated future。主模型先预测一次 action-free dynamics field：

\[
D_{t+1:t+H}=F(X_{\leq t})
\]

不同 draft/candidate 只沿不同轨迹路径读取：

\[
r_{\tau^m}=R(G_t,D_{t+1:t+H},S_t,\tau^m)
\]

只有在 simulator branch rollout 存在时，才增加低秩 action-reactive correction：

\[
\Delta D_{\tau}=F_{\mathrm{reactive}}(D,\tau)
\]

### 2.3 不在 world writer 中输入 GT future trajectory

必须保持：

\[
\mathcal M_t=E(X_{\leq t})
\]

world writer 构造阶段不得读取：

- GT future trajectory；
- NAVSIM final score；
- 未来 evaluator 输出；
- candidate-specific future label。

轨迹只在 readout/refinement 阶段出现。

### 2.4 解决 flow-matching 早期噪声轨迹不可查询的问题

原 action head 的 flow path 早期由随机噪声主导。直接沿 noisy action 查询几何场会产生无意义坐标。

核心实现采用 **Draft-Read-Refine**：

1. 原 action planner 完整产生有效 draft；
2. 对 draft `stop_gradient`；
3. 读取世界场；
4. 使用轻量 refiner 生成最终轨迹。

高级版本才在 flow 后几步使用 estimated clean trajectory 进行 self-conditioned readout。

### 2.5 不以最终 EPDMS 作为唯一辅助目标

优先预测可解释的物理后果：

- collision probability；
- minimum clearance；
- TTC；
- off-road/lane-boundary signed distance；
- stop-line/red-light violation；
- progress；
- acceleration、jerk、lateral acceleration；
- uncertainty。

最终 utility 可以由这些量组合，但不能只训练一个不透明的 EPDMS 回归器。

---

## 3. DriveDreamer-Policy 基线审阅与改造入口

本规格参考公开仓库 `youngzhou1999/DriveDreamer-Policy` 的 `main` 分支结构。Codex 开发时必须先检查本地仓库，不得假设本地代码与公开版本完全一致。

公开版本的重要入口如下：

| 路径 | 当前职责 | Field2Plan 改造方向 |
|---|---|---|
| `starVLA/model/framework/QwenOFT.py` | Qwen3-VL、world/action special queries、action/video/depth heads 的主框架 | 新建并行框架 `QwenOFT_Field2Plan.py`，避免直接破坏 baseline |
| `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | flow-matching action DiT；通过 `encoder_hidden_states` 读取 action query/video token | 保持 proposal planner；后续支持 world readout context 或独立 refiner |
| `starVLA/model/modules/vlm/QWen3.py` | Qwen3-VL wrapper；公开代码中存在视觉 feature-map 转换能力 | 增加稳定的 visual feature tap，优先显式返回，hook 仅作为 fallback |
| `starVLA/dataloader/navsim_dataset.py` | 读取 3 个前向相机、状态、8 步 4D 轨迹及可选 depth/video 数据 | 返回相机标定、ego-motion、teacher cache、baseline draft |
| `navsim_data_process/make_data.py` | 已保存 intrinsics、sensor-to-lidar rotation/translation、全局位姿 | 复用现有 metadata，不重复生成不一致坐标 |
| `starVLA/training/train_starvla.py` | 当前按开关硬编码累加 action/rgb/gs/reward loss | 增加向后兼容的通用 weighted-loss 聚合与详细日志 |
| `starVLA/config/training/cfg_yaw_1225.yaml` | action dim=4、horizon=8、flow steps 等基线配置 | 新建独立 Field2Plan 配置，不修改默认 baseline 配置 |
| `infer.py` | 输出每个 token 的规划轨迹 | 支持同时保存 draft、final、delta、readout 诊断信息 |
| `4-infer.sh`、`6-eval_v2.sh` | 推理和 NAVSIM-v2 评测 | 新增 Field2Plan 脚本，不覆盖原脚本 |

### 3.1 当前 action 表示

公开 dataloader 中的 action 为 8 步、4 维：

\[
a_h=[x_h,y_h,\sin\Delta\theta_h,\cos\Delta\theta_h]
\]

其中 \(x,y\) 是相对当前 ego pose 的未来位置，并可按代码中的均值/标准差归一化。

新代码必须集中实现 `TrajectoryCodec`，统一完成：

- normalize / denormalize；
- \(\sin,\cos\leftrightarrow\theta\)；
- heading wrap；
- draft/refined trajectory 组合；
- tube offset 生成；
- NAVSIM 输出格式转换。

不得在多个模块重复硬编码归一化常数。

### 3.2 当前 action head 可复用点

现有 flow-matching head已经通过：

```python
encoder_hidden_states=vl_embs
```

读取 action query，以及可选的 video token。因此有两条实施路线：

- **推荐 MVP**：保留原 action head作为 frozen proposal planner，新建轻量 refiner；
- **高级版**：扩展 action head，使后几步 denoising 可以读取 trajectory-tube context。

---

## 4. 总体架构

```text
历史/当前多视角图像 + ego state + navigation command
                            │
                            ▼
                    Qwen3-VL backbone
                            │
           ┌────────────────┼─────────────────┐
           ▼                ▼                 ▼
    visual feature maps   LLM semantics    action queries
           │                │                 │
           ▼                ▼                 ▼
  Geometry Field Writer   Semantic Tokens   Frozen/Base Planner
           │                                  │
           │                                  ▼
           │                           draft trajectory τ₀
           │                                  │ stop-gradient
           │                                  ▼
           │                       Trajectory-Tube Queries
           │                                  │
           ├──────────────┐                   │
           ▼              ▼                   ▼
   Geometry Field G   Dynamics Field D   Sparse Readout R(τ₀)
           │              │                   │
           └──────────────┴───────────────────┘
                            │
                            ▼
               Physical Outcome Heads（可选）
                            │
                            ▼
                  Trajectory Refiner
                            │
                            ▼
                    final trajectory τ*
```

### 4.1 三类知识空间

#### Geometry Field

\[
G_t\in\mathbb R^{B\times C_g\times N_y\times N_x}
\]

用于保存：

- free space；
- obstacle boundary；
- lane corridor；
- metric depth/point geometry；
- visibility/occlusion；
- relative geometry。

#### Dynamics Field

\[
D_{t+1:t+H}\in
\mathbb R^{B\times H\times C_d\times N_y\times N_x}
\]

用于保存：

- future occupancy tendency；
- agent motion；
- interaction-sensitive future features；
- temporal uncertainty。

#### Semantic Tokens

\[
S_t\in\mathbb R^{B\times K_s\times C_s}
\]

用于保存：

- route intent；
- traffic rules；
- signal semantics；
- yielding/interaction semantics；
- global scene context。

### 4.2 Dense Write，Sparse Read

外部 teacher 的知识写入结构化场，但 planner 不对整个 field 做 full attention。对 draft 的每个 waypoint 构造 query：

\[
q_h=
E_{xy}(x_h,y_h)+E_t(h)+E_\psi(\psi_h)+E_v(v_h)+E_c(c_\mathrm{nav})
\]

沿车辆 swept volume 读取：

\[
r_h=\operatorname{Read}(q_h,G,D,S)
\]

最终得到：

\[
r_\tau=[r_1,\ldots,r_H]
\]

---

## 5. 核心创新点

## 创新点一：Source-Native Factorized Prior Writing

### 目标

不把 VGGT、V-JEPA 和 VLM 语义强行压入同一种 slot topology，而是在它们最适合的空间中监督学生模型。

### Geometry teacher

第一版优先使用仓库已有的 Depth-Anything-3/metric depth 数据路径完成端到端开发，再接入 VGGT/StreamVGGT。

监督可包括：

\[
\mathcal L_\mathrm{geo}=
\lambda_d\mathcal L_\mathrm{depth}
+\lambda_o\mathcal L_\mathrm{occupancy}
+\lambda_f\mathcal L_\mathrm{free}
+\lambda_e\mathcal L_\mathrm{equivariance}
+\lambda_r\mathcal L_\mathrm{relative}
\]

其中：

- depth：metric/scale-aware depth；
- occupancy/free-space：ego BEV 可通行结构；
- equivariance：相机/ego-motion 变化下的一致性；
- relative：agent/lane/ego 的相对几何。

### Dynamics teacher

V-JEPA/Drive-JEPA 仅离线运行，提供：

- future latent target；
- temporal correspondence；
- masked future prediction target；
- agent-motion-sensitive feature。

监督：

\[
\mathcal L_\mathrm{dyn}=
\lambda_j\mathcal L_\mathrm{JEPA}
+\lambda_t\mathcal L_\mathrm{temporal}
+\lambda_m\mathcal L_\mathrm{motion}
+\lambda_u\mathcal L_\mathrm{uncertainty}
\]

### Shared-private decomposition

每个分支可使用：

\[
F_c=[F_\mathrm{shared},F_c^\mathrm{private}]
\]

避免以高 CKA/低 feature MSE 为目标将互补知识错误压成相同表示。

---

## 创新点二：Draft-Conditioned Trajectory-Tube Sparse Readout

### 为什么使用 draft

- draft 已是有效物理轨迹；
- 不使用 GT future 作为读出条件；
- 避免 flow 早期随机噪声坐标；
- 可固定 proposal，严格比较不同 reader/scorer/refiner；
- 可直接衡量 draft→final 的改进与失败。

### Tube 构造

对 waypoint \((x_h,y_h,\psi_h)\)，沿车身和安全缓冲采样：

- lateral offsets；
- longitudinal offsets；
- 多个 future time；
- 可选多个 height anchors；
- 车辆 footprint 和 braking margin。

配置而非硬编码车辆尺寸。

### Reader 输出

```text
per-waypoint context: [B, M, H, C]
per-candidate context: [B, M, C]
source gates:          [B, M, H, 3]
validity masks:        [B, M, H, P]
```

其中 \(M=1\) 为 MVP；高级版使用 \(M=4\sim8\)。

### 计算复杂度

full attention 近似：

\[
O(MHN)
\]

局部 trajectory-tube 读取：

\[
O(MHP),\quad P\ll N
\]

---

## 创新点三：Access-Preserving Refinement 与知识使用证据链

### Refiner

MVP 使用轻量 transformer/MLP refiner：

\[
\Delta\tau=
F_\mathrm{refine}(\tau_0,r_\tau,a_\mathrm{query})
\]

\[
\tau^*=\operatorname{Compose}(\tau_0,\Delta\tau)
\]

要求：

- delta gate 零初始化；
- 初始输出严格等于 draft；
- heading 使用角度组合后重新编码为 sin/cos；
- 可配置最大修正幅度；
- 提供 out-of-field fallback。

### 证据链

1. **Representation**：field probe 是否能解码几何/动态信息；
2. **Teacher specificity**：真实 teacher 是否优于 random/shuffled/GT-MLP；
3. **Access necessity**：同监督下禁用 readout 是否退化；
4. **Draft improvement**：final 是否稳定优于固定 draft；
5. **Causal response**：关键参与者/红绿灯干预是否产生正确方向变化；
6. **Closed-loop transfer**：NAVSIM-v2 navhard 与 Bench2Drive 是否共同提升。

---

## 6. 实现数据契约

## 6.1 Dataset sample

在不破坏原 key 的前提下，新增：

```python
sample = {
    # legacy
    "image": List[PIL.Image],
    "state": np.ndarray,          # [1, 4]
    "action": np.ndarray | None,  # [H, 4]
    "lang": str,
    "token": str,

    # field2plan
    "camera": {
        "view_names": List[str],
        "frame_index": int,
        "intrinsics": np.ndarray,          # [V, 3, 3] 或仓库真实格式
        "sensor2ego_rotation": np.ndarray, # [V, 3, 3]
        "sensor2ego_translation": np.ndarray, # [V, 3]
        "image_hw": np.ndarray,            # [V, 2]
    },
    "ego_motion": {
        "current_global_pose": np.ndarray,
        "future_relative_poses": np.ndarray,
    },
    "teacher": {
        "geometry": Optional[dict],
        "dynamics": Optional[dict],
        "manifest_hash": Optional[str],
    },
    "proposal": {
        "draft_action": Optional[np.ndarray], # [M, H, 4] or [H,4]
        "source": Optional[str],
    },
}
```

### 强制要求

- 缺失 cache 时默认 fail-fast；
- 只有 debug 配置可允许显式 `allow_missing_cache=true`；
- 不得用裸 `except:` 静默回退到上一个样本；
- 所有 coordinate convention 必须写入 manifest；
- loader 必须检查 token、split、teacher 版本和 preprocessing hash。

## 6.2 Teacher cache

建议布局：

```text
field2plan_cache/
├── manifests/
│   ├── geometry_da3_v1.json
│   ├── geometry_vggt_v1.json
│   ├── dynamics_vjepa_v1.json
│   └── baseline_draft_v1.json
├── geometry_da3/
│   └── {split}/{token}.npz
├── geometry_vggt/
│   └── {split}/{token}.npz
├── dynamics_vjepa/
│   └── {split}/{token}.npz
└── baseline_draft/
    └── {split}/{token}.npz
```

manifest 至少包含：

```yaml
schema_version: 1
teacher_name: ...
teacher_checkpoint: ...
teacher_commit: ...
source_repo_commit: ...
split: ...
camera_order: [cam_l0, cam_f0, cam_r0]
frame_indices: [...]
coordinate_frame: ego_at_t3
preprocessing_hash: ...
tensor_schema: ...
dtype: ...
created_at: ...
```

写文件时使用临时文件加原子 rename，避免多进程缓存损坏。

## 6.3 Tensor contract

建议主配置：

```text
visual feature map: [B, V, Cv, Hv, Wv]
geometry field:     [B, 256, 64, 64]
dynamics field:     [B, 8, 192, 64, 64]
semantic tokens:    [B, 4~8, 256]
draft trajectory:   [B, M, 8, 4]
tube samples:       [B, M, 8, P, 2]
read tokens:        [B, M, 8, 256]
final action:       [B, M, 8, 4]
```

debug 配置可使用 24×24/32×32 field 和 64/128 channels。

---

## 7. 视觉特征获取与 Field Writer

## 7.1 Visual feature tap

优先级：

1. 修改本地 Qwen3-VL wrapper，使 full forward 显式返回视觉 feature maps；
2. 若本地 custom Qwen 实现无法稳定修改，使用受控 forward hook；
3. 不允许默认重复运行完整 visual encoder；
4. 只有 debug fallback 可允许二次 visual forward。

feature tap 必须：

- 支持 gradient checkpointing；
- 每次 forward 后清空缓存；
- 不保留跨 batch tensor；
- 不改变 baseline 关闭功能时的输出；
- 在多 GPU/DeepSpeed 下不动态创建参数；
- 返回明确的 view/token 到 feature-map 映射。

## 7.2 Ego-centric field writer

建议使用可微投影 + `grid_sample`：

1. 建立 ego BEV grid；
2. 为每个 BEV cell 生成多个 height anchors；
3. 使用 intrinsics/extrinsics 投影到各相机；
4. 从多视角 visual feature maps 采样；
5. 对有效视角做 learned weighted aggregation；
6. 通过卷积/transformer 形成 geometry field。

建议范围：

```yaml
x_range_m: [-8.0, 56.0]
y_range_m: [-32.0, 32.0]
field_size: [64, 64]
height_anchors_m: [0.0, 1.0, 2.0]
```

这些只是初始值，必须通过 NAVSIM 坐标可视化确认：

- x 是否前向；
- y 是否左向；
- sensor2lidar 与 planning ego frame 的偏差；
- 相机 view 顺序；
- projected points 是否落在正确像素。

---

## 8. Draft-Read-Refine 规划流程

## 8.1 Proposal source

训练阶段按优先级：

1. **推荐**：离线缓存 frozen pure-trajectory baseline 的 draft；
2. debug fallback：在线 `torch.no_grad()` 运行 baseline action head；
3. 不允许使用 GT trajectory 伪装为 proposal。

推理阶段：

- 在线运行原 baseline proposal planner；
- 使用固定随机种子或确定性采样配置进行可复现消融；
- 可额外采样 \(M>1\) 个 draft。

## 8.2 Stop-gradient 边界

MVP：

```python
draft = draft.detach()
read_context = tube_reader(fields, draft)
final = refiner(draft, read_context, action_queries)
```

这样可以明确回答：

> 在 proposal 固定时，外部知识读取是否改善了最终轨迹？

高级联合微调再逐步解除 stop-gradient。

## 8.3 Refiner 组合

建议在物理空间组合：

```text
draft normalized action
        │ decode
        ▼
draft physical [x, y, theta]
        │ + bounded delta
        ▼
final physical [x, y, theta]
        │ encode
        ▼
final normalized [x, y, sin(theta), cos(theta)]
```

损失：

\[
\mathcal L_\mathrm{plan}=
\lambda_{xy}\operatorname{SmoothL1}(x^*,y^*;x^{gt},y^{gt})
+\lambda_\theta(1-\cos(\theta^*-\theta^{gt}))
+\lambda_\Delta\|\Delta\tau\|_1
\]

可增加：

- temporal smoothness；
- curvature；
- acceleration/jerk；
- out-of-bounds penalty。

---

## 9. 总训练目标

\[
\begin{aligned}
\mathcal L=&
\lambda_\mathrm{plan}\mathcal L_\mathrm{plan}
+\lambda_\mathrm{geo}\mathcal L_\mathrm{geo}
+\lambda_\mathrm{dyn}\mathcal L_\mathrm{dyn}\\
&+\lambda_\mathrm{out}\mathcal L_\mathrm{outcome}
+\lambda_\mathrm{rank}\mathcal L_\mathrm{rank}
+\lambda_\mathrm{ret}\mathcal L_\mathrm{retention}\\
&+\lambda_\mathrm{inv}\mathcal L_\mathrm{nuisance}
+\lambda_\mathrm{cal}\mathcal L_\mathrm{calibration}.
\end{aligned}
\]

### 9.1 Loss 聚合接口

新 framework 返回：

```python
{
    "loss": total_loss,
    "losses": {
        "plan": ...,
        "geometry": ...,
        "dynamics": ...,
        "outcome": ...,
        "ranking": ...,
        "delta_reg": ...,
    },
    "metrics": {
        "draft_l2": ...,
        "final_l2": ...,
        "delta_norm": ...,
        "field_valid_ratio": ...,
        "geometry_probe": ...,
        "source_gate_geo": ...,
        "source_gate_dyn": ...,
        "source_gate_sem": ...,
    }
}
```

trainer 必须兼容 legacy output；Field2Plan 模式使用配置中的 loss weights，不再简单无权相加。

---

## 10. 分阶段开发路线

## M0：锁定 baseline

目标：

- 记录当前纯轨迹 checkpoint、配置、commit、环境和 NAVSIM 成绩；
- 固定 seed、推理步数、归一化、评测 cache；
- 保存逐场景预测；
- 建立 regression test。

交付物：

```text
docs/field2plan/BASELINE_AUDIT.md
artifacts/baseline_manifest.json
tests/test_baseline_contract.py
```

验收：

- 原始 `debug.sh` 或等价单 batch forward/backward 通过；
- 固定输入、固定 seed 下 baseline 输出可复现；
- 不改算法时 NAVSIM 输出文件格式不变。

## M1：Field2Plan 基础设施与严格退化

实现：

- `TrajectoryCodec`；
- camera calibration 数据输出；
- visual feature tap；
- geometry field writer 的最小版本；
- trajectory-tube reader；
- zero-init refiner；
- 新 framework/config；
- weighted loss aggregator。

此阶段不使用真实外部 teacher，只验证架构。

验收：

- `field2plan.enabled=false` 与 baseline 完全一致；
- `refiner.gate=0` 时 final 与 draft 在数值容差内一致；
- reader 可反向传播；
- CPU unit tests 全部通过；
- 单 GPU debug forward/backward 通过。

## M2：Geometry Prior

先接入现有 DA3/depth pipeline，再接入 VGGT。

实现：

- geometry cache adapter；
- depth/occupancy/free-space supervision；
- ego-frame projection；
- geometry probe 和可视化；
- teacher-specific controls。

验收：

- geometry field 的 metric probe 明显优于 random/equal-capacity control；
- planning 增益必须比较：
  - supervision only；
  - access only；
  - supervision + access。

## M3：Dynamics Prior

实现：

- V-JEPA/Drive-JEPA offline cache；
- action-free future dynamics field；
- temporal alignment 和 ego-motion compensation；
- future latent/track/motion loss；
- dynamics-only 与 geometry+dynamics 对照。

验收：

- future probe 通过；
- temporal shuffle teacher 显著劣化；
- navhard/交互场景收益不只出现在普通 navtest。

## M4：Physical Outcome Grounding

实现：

- draft perturbation/candidate generation；
- NAVSIM metric component label 生成；
- collision/clearance/TTC/lane/red-light/progress/comfort heads；
- pairwise ranking；
- uncertainty calibration。

验收：

- final 相对同一 draft 的 oracle regret 降低；
- unsafe draft 误选率下降；
- ECE/Brier/NLL 改善；
- 不只报告总 EPDMS。

## M5：Advanced Self-Conditioned Readout

可选高级版本：

- 在 flow 后 \(K\) 个 denoising step 使用 estimated clean trajectory；
- 一次无 readout pass 估计 endpoint，再进行 read-guided pass；
- 或对最后若干步使用 tube context；
- 对计算开销和稳定性单独消融。

不得在 M1–M4 未稳定前提前实现。

## M6：闭环与因果实验

- NAVSIM-v2 navtest；
- NAVSIM-v2 navhard Stage 1/Stage 2；
- Bench2Drive；
- influential/non-influential agent masking；
- red/green light、lead braking、pedestrian crossing 等最小干预；
- nuisance augmentation consistency。

---

## 11. 建议代码结构

```text
starVLA/
├── model/
│   ├── framework/
│   │   └── QwenOFT_Field2Plan.py
│   └── modules/
│       └── field2plan/
│           ├── __init__.py
│           ├── types.py
│           ├── visual_feature_tap.py
│           ├── camera_geometry.py
│           ├── trajectory_codec.py
│           ├── geometry_field_writer.py
│           ├── dynamics_field_writer.py
│           ├── semantic_writer.py
│           ├── trajectory_tube_reader.py
│           ├── trajectory_refiner.py
│           ├── outcome_head.py
│           ├── losses.py
│           ├── controls.py
│           └── diagnostics.py
├── dataloader/
│   ├── navsim_dataset.py
│   └── field2plan_cache.py
├── config/training/
│   ├── cfg_field2plan_mvp.yaml
│   ├── cfg_field2plan_geometry.yaml
│   ├── cfg_field2plan_full.yaml
│   └── cfg_field2plan_controls.yaml
└── training/
    └── train_starvla.py

tools/field2plan/
├── cache_baseline_drafts.py
├── cache_geometry_da3.py
├── cache_geometry_vggt.py
├── cache_dynamics_vjepa.py
├── build_outcome_labels.py
├── validate_cache.py
├── visualize_coordinates.py
├── visualize_fields.py
└── analyze_access_ablation.py

scripts/field2plan/
├── 00_audit_baseline.sh
├── 01_debug_mvp.sh
├── 02_cache_drafts.sh
├── 03_train_mvp.sh
├── 04_cache_geometry.sh
├── 05_train_geometry.sh
├── 06_cache_dynamics.sh
├── 07_train_full.sh
├── 08_infer.sh
└── 09_eval_navsim_v2.sh

tests/field2plan/
├── test_trajectory_codec.py
├── test_camera_projection.py
├── test_feature_tap.py
├── test_tube_reader.py
├── test_zero_init_parity.py
├── test_cache_manifest.py
├── test_loss_aggregation.py
└── test_framework_smoke.py
```

---

## 12. 配置建议

```yaml
framework:
  name: QwenOFT_Field2Plan

field2plan:
  enabled: true

  proposal:
    source: cache            # cache | online
    cache_dir: null
    checkpoint: null
    freeze_base_planner: true
    stop_gradient: true
    num_candidates: 1
    online_fallback: false

  visual_tap:
    enabled: true
    mode: explicit           # explicit | hook | duplicate_debug_only
    selected_layers: [final]
    output_dim: 256
    detach_backbone: false

  geometry:
    enabled: true
    field_size: [64, 64]
    channels: 256
    x_range_m: [-8.0, 56.0]
    y_range_m: [-32.0, 32.0]
    height_anchors_m: [0.0, 1.0, 2.0]
    teacher_type: da3        # da3 | vggt | none
    cache_dir: null
    allow_missing_cache: false

  dynamics:
    enabled: false
    horizon: 8
    channels: 192
    teacher_type: vjepa
    cache_dir: null
    ego_motion_compensation: true
    allow_missing_cache: false

  semantics:
    enabled: true
    num_tokens: 4
    channels: 256

  reader:
    type: trajectory_tube
    output_dim: 256
    lateral_offsets_m: [-1.0, 0.0, 1.0]
    longitudinal_offsets_m: [0.0, 2.5]
    invalid_policy: masked_zero
    source_dropout: 0.0

  refiner:
    type: transformer
    hidden_dim: 512
    num_layers: 4
    zero_init: true
    max_delta_xy_m: 4.0
    max_delta_heading_rad: 0.5

  outcome:
    enabled: false
    targets:
      [collision, clearance, ttc, offroad, redlight, progress, comfort]

  controls:
    disable_access: false
    shuffle_teacher_across_batch: false
    random_teacher: false
    gt_mlp_teacher: false
    equal_capacity_no_teacher: false

trainer:
  learning_rate:
    base: 1.0e-5
    geometry_field_writer: 1.0e-4
    dynamics_field_writer: 1.0e-4
    trajectory_tube_reader: 1.0e-4
    trajectory_refiner: 1.0e-4

  loss_weights:
    plan: 1.0
    geometry: 0.2
    dynamics: 0.1
    outcome: 0.2
    ranking: 0.1
    delta_reg: 0.01
```

实际 module path 必须与本地模型属性一致。

---

## 13. 测试与工程约束

### 13.1 必须测试

1. **Baseline parity**
   - 功能关闭时，输出与原 baseline 一致；
   - 相同 checkpoint 和 seed 下 inference 文件一致。

2. **Zero-init parity**
   - refiner 初始 gate 为零时 final=draft。

3. **Trajectory codec**
   - encode→decode round trip；
   - heading wrap；
   - normalization parity；
   - invalid sin/cos 归一化。

4. **Camera projection**
   - synthetic camera；
   - known 3D point→pixel；
   - view validity；
   - batch/view 维度。

5. **Reader**
   - field boundary；
   - invalid mask；
   - gradient；
   - variable M/H/P。

6. **Cache**
   - manifest mismatch；
   - split/token mismatch；
   - corrupted file；
   - atomic writer。

7. **Distributed safety**
   - forward 内不创建参数；
   - 不使用无条件 `.cuda()`；
   - dtype/device 继承输入；
   - DeepSpeed/gradient checkpointing smoke test。

### 13.2 非功能约束

- 新功能默认关闭；
- 不改 vendored `navsim/` 和 `depth_process/Depth-Anything-3/`；
- 所有路径来自 config/env；
- 不在训练中在线运行 VGGT/V-JEPA；
- 不下载模型；
- 不使用裸 `except:`；
- 每个 tensor 边界有 shape assertion；
- 保存 config、git commit、teacher manifest hash；
- 诊断输出不能默认保存大 tensor；
- W&B 不可用时支持本地 JSONL。

---

## 14. 实验矩阵

## 14.1 核心结构比较

| ID | 知识存储 | Planner 访问 | Refiner | 目的 |
|---|---|---|---|---|
| A0 | 无 | baseline action path | 无 | pure trajectory |
| A1 | fixed world slots | static slot access | 同容量 | 原始 slots 范式 |
| A2 | dense field | 普通 action token | 同容量 | 纯视觉/BEV 融合 |
| A3 | dense field + static slots | static expert slots | 同容量 | 混合但非轨迹条件 |
| A4 | dense field | trajectory-tube read | refiner | proposed |
| A5 | online teacher field | trajectory-tube read | refiner | teacher upper bound |

## 14.2 Supervision × Access

| 外部监督 | Reader access | 解释 |
|---|---|---|
| 无 | 无 | baseline |
| 有 | 无 | supervision/regularization only |
| 无 | 有 | architecture/capacity only |
| 有 | 有 | full model |

## 14.3 Teacher-specific controls

- real teacher；
- frozen random teacher；
- scene-shuffled teacher；
- temporal-shuffled dynamics teacher；
- GT-MLP；
- equal-capacity adapter；
- online teacher upper bound。

## 14.4 Slot/readout 消融

- generic slots；
- position-anchored slots；
- static typed expert slots；
- trajectory-conditioned queries；
- dense bypass on/off；
- reader source gate on/off；
- 4/8/16/32 queries；
- normal/crowded scene 分层。

---

## 15. 评测与统计

### 15.1 Benchmark

- 开发：NAVSIM-v2 navtest；
- 主结果：NAVSIM-v2 navhard，分别报告 Stage 1、Stage 2；
- 历史兼容：NAVSIM-v1；
- 闭环：Bench2Drive；
- 鲁棒性：Bench2Drive-Robust。

### 15.2 必报指标

- EPDMS 及各 component；
- draft score 与 final score；
- paired per-scene delta；
- trajectory L2/heading；
- minimum clearance；
- TTC calibration；
- unsafe draft correction rate；
- safe draft degradation rate；
- proposal recall；
- candidate ranking regret；
- field probe；
- readout valid ratio；
- source gate；
- latency、FLOPs、显存。

### 15.3 统计规范

- 普通消融至少 3 个训练 seed；
- 核心结果 5 个 seed；
- 保存逐场景输出；
- paired bootstrap 10,000 次；
- 报告 mean、std、95% CI；
- 固定 checkpoint selection；
- 不只报告 best seed；
- 将训练方差与固定 checkpoint 推理方差分开。

---

## 16. 成功标准与停止条件

### 必须满足

1. **Baseline 不退化**
   - 功能关闭时完全一致；
   - zero-init 时 final=draft。

2. **Teacher-specific**
   - real teacher 显著优于 random/shuffled/GT-MLP/equal-capacity。

3. **Access-specific**
   - 相同监督下，允许 trajectory readout 显著优于 no-access。

4. **Draft-specific**
   - 同一组 draft 上，final 的安全性/规划指标稳定改善。

5. **Causal correctness**
   - influential agent、红绿灯、前车刹车等干预方向正确；
   - nuisance edit 不导致大幅无关轨迹变化。

6. **Closed-loop transfer**
   - NAVSIM 提升至少在 Bench2Drive 某些交互类别复现。

### 应停止或降级论文主张的情况

- real teacher 不优于 GT-MLP；
- access on/off 无显著差异；
- field probe 提升但 final planning 不提升；
- 只提升总 EPDMS，但 clearance/TTC/违规不改善；
- navtest 提升但 navhard/闭环退化；
- 增益小于多 seed 方差；
- 主要收益来自 refiner 参数量而非 teacher。

---

## 17. 明确禁止的实现捷径

- 使用 GT future trajectory 作为 field writer 输入；
- 将 demonstrated future 当作任意 perturbed candidate 的未来真值；
- 训练时在线运行外部大模型；
- 缺少 cache 时静默使用零 tensor；
- 只预测 EPDMS，不预测物理后果；
- 不做 random/shuffled/GT-MLP 控制；
- 修改 baseline 默认配置导致不可复现；
- 在新模块中散落硬编码坐标与归一化常数；
- 在 flow 的纯噪声阶段直接把 noisy trajectory 当作物理路径；
- 通过增加大量参数而不做 equal-capacity control；
- 只展示 attention/t-SNE，声称知识被使用。

---

## 18. 推荐论文叙事

### 核心标题方向

**Field2Plan-VLA: Action-Conditioned Late Bottlenecks for Distilling External World Knowledge into Autonomous Driving**

### 三项贡献

1. **Source-Native Factorized Prior Writing**  
   在空间、时间和语义各自适合的表示拓扑中写入外部先验，而不是强制统一为固定 slots。

2. **Draft-Conditioned Trajectory-Tube Sparse Readout**  
   在形成轨迹假设后才进行信息压缩，并只读取当前动作后果相关的局部世界信息。

3. **Access-Preserving Knowledge Utilization Protocol**  
   通过 frozen proposal、zero-init refinement、supervision×access 和 teacher-specific controls，区分“知识被学习”与“知识被规划器使用”。

### 一句话主张

> 外部世界知识的有效压缩不应在场景编码后立即发生，而应在形成动作假设后，以轨迹条件化的方式延迟发生。

---

## 19. 开发优先级

### 第一优先级

- baseline audit；
- TrajectoryCodec；
- calibration contract；
- visual feature tap；
- draft cache；
- trajectory-tube reader；
- zero-init refiner；
- no-teacher architecture control。

### 第二优先级

- DA3 geometry supervision；
- ego-BEV field；
- supervision×access；
- geometry teacher controls。

### 第三优先级

- V-JEPA dynamics cache；
- action-free future field；
- physical outcome heads；
- multi-candidate ranking。

### 暂缓

- Cosmos/Wan 作为核心；
- 每 candidate RGB future generation；
- 全量 action-reactive world rollout；
- denoising 每一步 world readout；
- 大规模 RL 后训练。

---

## 20. 参考入口

- DriveDreamer-Policy repository: `https://github.com/youngzhou1999/DriveDreamer-Policy`
- DriveDreamer-Policy paper: `https://arxiv.org/abs/2604.01765`
- 本文档中的具体路径和配置应以本地 checkout 为准。
