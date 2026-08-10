# GroundedWorld-VLA：基于 VLM + DiT 的外部世界知识迁移与规划利用研究方案

> **版本**：Revised Research Plan  
> **基线边界**：仅复用 DriveDreamer-Policy 的 **VLM + DiT / Flow-Matching 轨迹生成框架**。  
> **明确不继承**：原方法的 depth/video query、world query、世界模型、辅助监督设计、候选轨迹评分器等。  
> **研究目标**：从零设计并实现一套能够**获取外部世界知识、预测未来世界状态、并让这些知识真实参与轨迹生成**的 VLA 自动驾驶算法。

---

# 0. Executive Summary

本研究的核心问题不是“再给 VLA 加一个 world model head”，而是：

> **外部基础模型中的几何、动态和交互知识，应该如何进入 VLA，并且如何证明这些知识真正改变了最终轨迹规划？**

当前方案最终收敛为三阶段主线：

\[
\boxed{
\text{Acquire}
\rightarrow
\text{Predict}
\rightarrow
\text{Ground Action}
}
\]

即：

1. **Acquire：获取并内化外部先验**
   - VGGT 提供 metric geometry / 3D structure；
   - driving-adapted JEPA 提供 temporal dynamics prior；
   - 外部 teacher 仅在训练阶段存在。

2. **Predict：构造可预测的世界表征**
   - 当前场景保留高分辨率 ego-aligned geometry；
   - temporal path 学习 action-free future representation；
   - 第一阶段不声称学习完整 action-conditioned counterfactual world model。

3. **Ground Action：让 world knowledge 直接参与轨迹规划**
   - 原 VLM+DiT 先生成 physically meaningful draft trajectory；
   - 沿 draft trajectory 从 current/future world memory 中读取局部物理信息；
   - residual refiner 直接修正最终轨迹；
   - planner-side 新路径采用 zero-init，初始严格退化回纯轨迹 baseline。

研究的三个主要创新点为：

1. **Dual-Path Foundation Prior Internalization**  
   同时保留 VLM semantic path 和独立的 physical world path，避免所有精细几何信息被语言/VLM hidden-state bottleneck 压缩。

2. **Confound-Separated Predictive World Representation**  
   严格区分“foundation-model prior transfer”和“future privileged supervision”，避免把未来 GT 深监督误认为外部模型知识。

3. **Trajectory-Conditioned World-Grounded Refinement**  
   不在高噪声 diffusion state 上直接查询世界；先生成有效 draft，再沿轨迹 swept tube 读取世界信息并直接修正最终规划。

---

# 1. 研究边界

## 1.1 我们真正拥有的 baseline

本研究将 DriveDreamer-Policy 简化为：

\[
I_{t-L:t},\ c_{\mathrm{nav}},\ s_{\mathrm{ego}}
\xrightarrow{\text{VLM}}
H_{\mathrm{sem}}
\xrightarrow{\text{DiT / Flow Matching}}
\tau
\]

其中：

- \(I_{t-L:t}\)：历史多视角图像；
- \(c_{\mathrm{nav}}\)：导航命令或 route context；
- \(s_{\mathrm{ego}}\)：ego 状态；
- \(H_{\mathrm{sem}}\)：VLM 视觉-语义表示；
- \(\tau\)：规划轨迹。

除此之外，以下能力全部视为**尚不存在，需要自行设计**：

- geometry representation；
- world memory；
- future prediction；
- temporal dynamics representation；
- external foundation-model distillation；
- world-to-planner access；
- planning consequence grounding；
- prior retention；
- causal utilization validation。

---

# 2. 核心科学问题

## 2.1 VLM 语义理解不等于 planning-sufficient world representation

VLM 可以理解：

> 前方存在车辆。

但规划器实际需要的是：

- 距离；
- 相对速度；
- free-space；
- road boundary；
- collision margin；
- temporal evolution；
- TTC；
- lane relation；
- future interaction。

因此：

\[
\boxed{
\text{Semantic Understanding}
\neq
\text{Planning-Sufficient World Representation}
}
\]

本研究希望学习：

\[
M_t=f_{\theta}(I_{t-L:t})
\]

使 \(M_t\) 对规划至少保存：

1. 当前 metric geometry；
2. temporal motion structure；
3. future world evolution；
4. route-relevant interaction information；
5. 可被 trajectory-conditioned planner 读取的物理局部信息。

---

# 3. 经审计后的设计决策

下面区分哪些设计**保留**、哪些**修改**、哪些**删除**。

| 设计 | 结论 | 处理 |
|---|---|---|
| VGGT 作为 geometry teacher | 保留 | 用于 metric geometry / 3D structure |
| driving-adapted JEPA 作为 dynamics prior | 保留 | 优先于直接使用通用 V-JEPA |
| ego-centric 坐标统一 | 强保留 | 作为硬约束 |
| multi-scale geometry memory | 保留 | 细节与大范围上下文并存 |
| action-free future prediction | 保留 | 第一阶段避免不可识别反事实监督 |
| planner direct access | 强保留 | world knowledge 必须影响最终轨迹 |
| zero-init world-to-plan path | 强保留 | 初始严格等价 baseline |
| prior retention | 新增 | 避免 planner fine-tuning 擦除 teacher prior |
| 所有知识强制写入 VLM hidden | 修改 | 改为 semantic path + physical path |
| VGGT feature MSE 为核心 | 修改 | feature KD 降为辅助，主监督变成物理 grounding |
| 直接 future JEPA feature 作为 teacher-specific target | 修改 | 拆分 prior transfer 与 future GT supervision |
| dense future field 必然最好 | 不预设 | 作为实验变量 |
| 高噪声 DiT 内 trajectory query | 删除 | 缺乏足够直接证据 |
| 多候选 trajectory reranking | 删除 | 避免候选搜索成为主要增益来源 |
| EPDMS scalar prediction | 删除 | 避免 metric emulator |
| NAVSIM offline action-conditioned counterfactual world | 删除 | 日志数据不可识别 |
| Cosmos / Wan 作为主模型组成 | 删除 | 第一篇工作控制变量 |
| 固定 GEO/DYN/RULE world slots | 删除主路径 | 仅保留为对照 baseline |

---

# 4. 总体算法

整体结构如下：

```text
                Multi-view / temporal images
                           │
                    Shared Visual Encoder
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
     Semantic VLM Path             Physical World Path
            │                             │
     route / semantics              ┌─────┴─────┐
     interaction context            │           │
            │                    Geometry    Dynamics
            │                     Grounder     Encoder
            │                       │            │
            │                     VGGT      Driving-JEPA
            │                  TRAIN ONLY    TRAIN ONLY
            │                       │            │
            │                       ▼            ▼
            │                    G_t          D_t
            │                       └─────┬──────┘
            │                             │
            │                    Predictive Memory
            │                     M_{t+1:t+H}
            │                             │
            ▼                             │
        Baseline DiT                      │
            │                             │
            ▼                             │
      Draft trajectory τ⁰                 │
            │                             │
            └──────────────┬──────────────┘
                           ▼
                 Trajectory-Tube Reader
                           │
                    local world evidence
                           │
                           ▼
                 World-Grounded Refiner
                           │
                           ▼
                 Final trajectory τ¹
```

---

# 5. Innovation 1：Dual-Path Foundation Prior Internalization

## 5.1 为什么不是“所有外部知识都写进 VLM”

VLM 的高层 hidden states 更适合：

- navigation semantics；
- global scene semantics；
- traffic rule reasoning；
- interaction semantics。

而 metric geometry 对：

- spatial resolution；
- camera geometry；
- 3D consistency；
- local obstacle boundary；

要求更高。

因此不应要求所有信息都穿过：

\[
\text{visual feature}
\rightarrow
\text{VLM bottleneck}
\rightarrow
\text{planner}
\]

而改为：

\[
F^{vis}
\rightarrow
\begin{cases}
H^{sem} & \text{Semantic VLM Path}\\
M^{phy} & \text{Physical World Path}
\end{cases}
\]

两者共享底层 visual encoder，但保持不同 representation topology。

---

# 6. Shared Visual Encoder

输入：

\[
I_{t-L:t}
\]

通过现有 VLM visual encoder 获取多层特征：

\[
F_l^{vis},\quad l\in\mathcal L
\]

不要求每一层都接入 external teacher。

第一版推荐只选择：

- 一个中层；
- 一个高层；
- 或 backbone 已提供的多尺度输出。

具体层数通过消融确定。

输出分成：

```text
F_vis
 ├── Semantic/VLM projector → H_sem
 └── Physical World Adapter → M_phy
```

---

# 7. Geometry Grounder

## 7.1 Teacher

核心 teacher：

\[
\text{VGGT / StreamVGGT}
\]

teacher 只用于训练和离线 cache。

主要提供：

- depth；
- camera/pose geometry；
- point map；
- 3D tracks（可选）；
- multi-view correspondence。

---

## 7.2 坐标统一

所有 geometry target 必须转换到：

\[
\boxed{\text{current ego coordinate}}
\]

禁止将：

- teacher world frame；
- first-frame camera frame；
- arbitrary reference frame；

直接作为 planner geometry representation。

定义：

\[
P^{ego}_{t}
=
T_{cam\rightarrow ego,t}P^{cam}_t
\]

历史帧再进行 ego-motion compensation：

\[
P^{ego_t}_{t-k}
=
T_{ego_{t-k}\rightarrow ego_t}P^{ego_{t-k}}_{t-k}
\]

最终所有时刻都统一到当前规划时刻 \(t\)。

---

# 8. Geometry Memory

物理路径输出当前几何 memory：

\[
G_t
=
\{
G_t^{1},
G_t^{2},
G_t^{3}
\}
\]

其中每层可以表示不同 BEV resolution：

\[
G_t^l\in
\mathbb R^{H_l\times W_l\times C_l}
\]

推荐保留 multi-scale：

- 高分辨率：lane boundary / collision margin；
- 中分辨率：local free-space；
- 低分辨率：larger interaction context。

---

## 8.1 Geometry Loss

不将 feature MSE 作为主要目标。

推荐：

\[
L_{\mathrm{geo}}
=
\lambda_dL_{\mathrm{depth}}
+
\lambda_pL_{\mathrm{3D}}
+
\lambda_fL_{\mathrm{free}}
+
\lambda_kL_{\mathrm{prior}}
\]

### Depth grounding

\[
L_{\mathrm{depth}}
=
L(
\hat d,
d^{teacher}
)
\]

可使用：

- log-depth；
- scale-invariant loss；
- depth rank；
- confidence weighted depth。

### 3D grounding

\[
L_{\mathrm{3D}}
=
\sum_i
w_i
\|
\hat p_i^{ego}
-
p_i^{teacher,ego}
\|_1
\]

### Free-space

从 geometry teacher / map / occupancy proxy 得到：

\[
y^{free}(x,y)
\]

预测：

\[
\hat y^{free}(x,y)
\]

### Weak prior retention

\[
L_{\mathrm{prior}}
=
d(
P(G_t),
\operatorname{sg}(F^{VGGT})
)
\]

但其权重应低于 explicit physical grounding。

---

# 9. Dynamics Encoder

当前 geometry 不能区分：

- 静止前车；
- 急减速前车；
- 横穿行人；
- cut-in 车辆。

因此需要 temporal dynamics representation。

---

## 9.1 Teacher 选择

第一优先级：

\[
\boxed{
\text{driving-domain JEPA}
}
\]

推荐候选：

1. Drive-JEPA；
2. V-JEPA 2/2.1 初始化后在驾驶视频上继续自监督训练；
3. 通用 V-JEPA 作为对照。

原因：

\[
\text{generic video prior}
\neq
\text{driving-domain motion prior}
\]

是否存在 domain adaptation 收益本身也应成为消融项。

---

# 10. 必须区分两类 dynamics supervision

这是本研究一个非常重要的 methodological constraint。

## 10.1 External-prior supervision

只从**当前和历史可见帧**提取 teacher feature：

\[
Z_{\le t}^{DJ}
=
E_{\mathrm{DriveJEPA}}
(I_{t-L:t})
\]

student：

\[
D_t
=
E_{\mathrm{dyn}}
(F^{vis}_{t-L:t})
\]

优化：

\[
L_{\mathrm{dyn-prior}}
=
d(
P(D_t),
\operatorname{sg}(Z_{\le t}^{DJ})
)
\]

这个 loss 才代表：

> pretrained external dynamics prior transfer。

---

## 10.2 Future prediction supervision

future GT image 属于 privileged future supervision。

因此所有 teacher ablation 必须共享同一种 future target。

定义 student/EMA target encoder：

\[
Z_{t+h}^{target}
=
\operatorname{sg}
\left(
E^{EMA}_{world}(I_{t+h})
\right)
\]

predictor：

\[
\hat Z_{t+h}
=
T_{\theta}
(
G_t,D_t,h
)
\]

优化：

\[
L_{\mathrm{future}}
=
\sum_{h=1}^{H}
d(
\hat Z_{t+h},
Z_{t+h}^{target}
)
\]

因此 foundation-teacher 的真实贡献可以定义为：

\[
\Delta_{\mathrm{prior}}
=
S(
RealPrior+SameFutureSupervision
)
-
S(
NoPrior+SameFutureSupervision
)
\]

避免把 future GT deep supervision 当成 foundation model knowledge。

---

# 11. Predictive World Memory

第一阶段只学习：

\[
p(M_{t+1:t+H}\mid I_{\le t})
\]

而不是：

\[
p(M_{t+1:t+H}\mid I_{\le t},\tau)
\]

因此称：

> **Predictive World Representation / Predictive World Memory**

而不是：

> fully interactive counterfactual world model。

---

## 11.1 Memory 定义

\[
M_t=
\{G_t,D_t\}
\]

未来：

\[
\hat M_{t+h}
=
T_{\mathrm{world}}
(
M_t,h
)
\]

完整 memory：

\[
\mathcal M
=
\{
G_t,
D_t,
\hat M_{t+1},
\dots,
\hat M_{t+H}
\}
\]

---

# 12. Future Representation 不预设为 dense

这是一个明确需要实验回答的问题。

实现三种形式：

## A. Dense Future Field

\[
D_{t+h}^{dense}
\in
\mathbb R^{H\times W\times C}
\]

优点：

- 保留空间局部结构；
- 便于 trajectory sampling。

缺点：

- 算力和显存较大；
- 可能预测大量与规划无关信息。

---

## B. Sparse Dynamic Tokens

按动态区域/agent/cluster 表示：

\[
D_{t+h}^{sparse}
=
\{z_1,\dots,z_K\}
\]

但这些 token 必须有：

- spatial anchor；
- temporal anchor；
- confidence。

---

## C. Planning Intent Representation

只预测与未来 ego planning 高相关的 compact latent：

\[
D_{t+h}^{intent}
\]

问题转化为：

> 为规划所需的未来信息到底需要多稠密？

这应通过相同 planner、相同训练预算进行对照。

---

# 13. Innovation 2：Confound-Separated Predictive World Representation

完整目标不是：

\[
F_{student}^{future}
\approx
F_{external}^{future}
\]

而是将两件事拆开：

### External prior transfer

\[
I_{\le t}
\rightarrow
\text{external-prior-aligned current representation}
\]

### Future prediction

\[
M_{\le t}
\rightarrow
M_{t+1:t+H}
\]

所有模型获得完全相同的 future supervision。

这样才能回答：

> foundation model 预训练知识是否真正超过普通 future deep supervision？

---

# 14. Baseline DiT 保留

当前 VLM+DiT planner 继续负责生成 first-pass trajectory：

\[
\tau^{(0)}
=
P_{\mathrm{DiT}}
(
H^{sem},
s_{\mathrm{ego}},
c_{\mathrm{nav}}
)
\]

此处不立即强制 world memory 参与。

原因：

- baseline 已经能产生 physically meaningful trajectory；
- 避免在高噪声 flow state 上使用不可靠 spatial query；
- 保持 strong baseline；
- 便于做 planner-access causal comparison。

---

# 15. Innovation 3：Trajectory-Conditioned World-Grounded Refinement

世界信息进入规划的关键路径：

\[
\tau^{(0)}
\rightarrow
\operatorname{Read}(
\mathcal M,\tau^{(0)}
)
\rightarrow
\Delta\tau
\]

最终：

\[
\tau^{(1)}
=
\tau^{(0)}
+
\alpha\Delta\tau
\]

其中：

\[
\alpha_{init}=0
\]

因此初始化时：

\[
\tau^{(1)}
=
\tau^{(0)}
\]

确保 world-refinement 模块刚加入时不会破坏 baseline。

---

# 16. Trajectory-Tube Reader

slot/query 在本研究里只承担：

> **读取世界信息**

而不是：

> 存储整个世界。

给定 draft：

\[
\tau^{(0)}
=
\{p_h\}_{h=1}^{H}
\]

其中：

\[
p_h=
(x_h,y_h,\psi_h,v_h,t_h)
\]

每个轨迹点构造 trajectory query：

\[
q_h=
E_x(x_h,y_h)
+
E_t(t_h)
+
E_{\psi}(\psi_h)
+
E_v(v_h)
\]

---

## 16.1 不只采样中心 waypoint

轨迹真实占据的是 swept volume。

定义采样区域：

\[
\Omega_h=
\{
p_h+\Delta p_i
\}_{i=1}^{K}
\]

包含：

- ego footprint；
- 前向安全距离；
- braking corridor；
- lateral margin；
- lane-adjacent region。

然后：

\[
r_h
=
\operatorname{DeformRead}
(
q_h,
\mathcal M,
\Omega_h
)
\]

最终：

\[
R_{\tau}
=
\{r_1,\dots,r_H\}
\]

---

# 17. World-Grounded Refiner

输入：

\[
[
\tau^{(0)},
R_{\tau},
H^{sem}
]
\]

输出：

\[
\Delta\tau
\]

可以先实现轻量 Transformer/DiT-style refiner：

\[
\Delta\tau
=
F_{\mathrm{refine}}
(
\tau^{(0)},
R_{\tau},
H^{sem}
)
\]

第一版只做一次 refinement：

\[
\tau^{(1)}
=
\tau^{(0)}
+
\alpha\Delta\tau
\]

后续实验：

- 1 iteration；
- 2 iterations；
- 4 iterations。

若多次 refinement 没有显著收益，则保持单步版本。

---

# 18. 为什么不使用 candidate reranking

不使用：

```text
sample N trajectories
→ world critic
→ score
→ choose best
```

因为这会引入额外混淆：

\[
\text{planning improvement}
=
\text{better world knowledge}
+
\text{larger search}
+
\text{reranking}
\]

本研究希望证明：

\[
\boxed{
\text{world knowledge 直接改变同一个 planner 的最终 action}
}
\]

因此主实验只输出一条最终轨迹。

---

# 19. Planning Consequence Grounding

以前称“Decision-Preserving External KD”不够严谨。

修改为：

# Planning Consequence Grounding

目标：

> world representation 必须能区分不同轨迹的物理后果。

---

## 19.1 轨迹扰动

围绕：

\[
\tau^{GT}
\]

生成：

\[
\tilde\tau^k
=
\tau^{GT}
+
\delta^k
\]

扰动包括：

- lateral shift；
- curvature perturbation；
- speed scaling；
- delayed braking；
- extra braking；
- heading perturbation。

---

## 19.2 可识别物理结果

只监督能够从当前数据可靠获得的量：

\[
y(\tau)=
[
d_{\min},
TTC,
collision,
lane\ distance,
progress,
comfort
]
\]

第一阶段不把记录到的 future agent behavior 当成所有 ego perturbation 的真实反事实响应。

因此这些标签应区分：

### 可直接计算

- static geometry collision；
- free-space violation；
- lane boundary；
- progress；
- acceleration / jerk。

### 近似计算

- 基于 logged agents future 的 TTC；
- non-reactive collision proxy。

### 只有 simulator 才可靠

- reactive cut-in response；
- yielding；
- negotiation；
- interactive merge；
- ego action 改变其他 agent 行为。

后者只在 Bench2Drive/CARLA branch rollout 阶段使用。

---

# 20. Consequence Head 仅训练时存在

\[
\hat y^k
=
C_{\mathrm{con}}
(
Read(\mathcal M,\tilde\tau^k)
)
\]

训练：

\[
L_{\mathrm{con}}
=
L_{clearance}
+
L_{TTC}
+
L_{lane}
+
L_{progress}
+
L_{comfort}
\]

禁止直接训练：

\[
\widehat{\mathrm{EPDMS}}
\]

禁止推理时：

\[
\arg\max_{\tau}\widehat{\mathrm{EPDMS}}
\]

Consequence head 的唯一作用是：

> 让 world memory 保留 trajectory-discriminative physical information。

---

# 21. Prior Retention

外部 prior 即使在预训练阶段学到了，也可能在后续 planning fine-tuning 中被覆盖。

因此整个联合训练期间保留：

\[
L_{\mathrm{ret}}
=
d(
P(M_t),
\operatorname{sg}(Z_t^{teacher})
)
\]

但不用监督所有位置。

优先采样：

- near-route region；
- dynamic object region；
- geometry-sensitive region；
- high-uncertainty region。

这样可以降低 teacher-cache 读取和训练成本。

---

# 22. 总训练目标

最终控制在五类核心 loss：

\[
\boxed{
L
=
L_{\mathrm{plan}}
+
\lambda_gL_{\mathrm{geo}}
+
\lambda_fL_{\mathrm{future}}
+
\lambda_rL_{\mathrm{ret}}
+
\lambda_cL_{\mathrm{con}}
}
\]

其中：

### Planning

\[
L_{\mathrm{plan}}
=
L_{\mathrm{flow}}
+
\lambda_{\mathrm{refine}}
L_{\mathrm{traj-refine}}
\]

### Geometry

\[
L_{\mathrm{geo}}
=
L_{\mathrm{depth}}
+
L_{\mathrm{3D}}
+
L_{\mathrm{free}}
+
\lambda_kL_{\mathrm{geo-prior}}
\]

### Future

\[
L_{\mathrm{future}}
=
d(
\hat M_{t+1:t+H},
M^{target}_{t+1:t+H}
)
\]

### Retention

\[
L_{\mathrm{ret}}
=
d(
P(M_t),
Z_t^{teacher}
)
\]

### Consequence

\[
L_{\mathrm{con}}
=
L_{\mathrm{clearance}}
+
L_{\mathrm{TTC}}
+
L_{\mathrm{lane}}
+
L_{\mathrm{progress}}
+
L_{\mathrm{comfort}}
\]

不要在第一版继续加入更多 loss。

---

# 23. 三阶段训练策略

## Stage I：Foundation Prior Internalization

目标：

\[
I_{\le t}
\rightarrow
M_t
\]

先训练 Physical World Path。

训练：

- Geometry Grounder；
- Dynamics Encoder；
- optional shared visual adapters。

损失：

\[
L^{I}
=
L_{\mathrm{geo}}
+
\lambda_dL_{\mathrm{dyn-prior}}
\]

此阶段：

- 不训练 trajectory refiner；
- 不训练 consequence head；
- 不修改 DiT；
- 先验证 representation 是否真的获得物理能力。

---

## Stage II：Predictive Memory Learning

加载 Stage I。

训练：

\[
M_t
\rightarrow
M_{t+1:t+H}
\]

损失：

\[
L^{II}
=
L_{\mathrm{future}}
+
\lambda_rL_{\mathrm{ret}}
\]

重点验证：

- future latent error；
- future geometry；
- motion probe；
- TTC probe；
- temporal consistency。

---

## Stage III：Planning Co-Training

加载：

- pure VLM+DiT trajectory baseline；
- Stage II world representation。

新增：

- Trajectory-Tube Reader；
- World-Grounded Refiner；
- Consequence Head。

### Phase III-A

固定或低学习率 DiT。

只训练：

- reader；
- refiner；
- planner adapter。

初始化：

\[
\alpha=0
\]

### Phase III-B

逐步联合 fine-tune：

- shared visual encoder：低 LR；
- VLM：低 LR / partial LoRA；
- world path：正常 LR；
- reader/refiner：正常 LR；
- DiT：低 LR。

持续保留：

\[
L_{\mathrm{ret}}
\]

防止外部 prior 被 planning objective 擦除。

---

# 24. 关键实验原则

任何 external teacher 增益都必须使用：

\[
\boxed{
\Delta_{teacher}
=
S(
Teacher+SameSupervision
)
-
S(
NoTeacher+SameSupervision
)
}
\]

而不是：

\[
S(full)-S(baseline)
\]

因为：

\[
S(full)-S(baseline)
\]

混合了：

- extra supervision；
- extra parameters；
- future privileged targets；
- reader；
- refiner；
- teacher prior。

---

# 25. 核心消融矩阵

## 25.1 主模型逐步增加

| ID | External Prior | Future | World Access | Refiner | Consequence | 目的 |
|---|---|---|---|---|---|---|
| B0 | × | × | × | × | × | pure VLM+DiT |
| B1 | VGGT | × | × | × | × | geometry auxiliary effect |
| B2 | VGGT | × | ✓ | ✓ | × | geometry direct access |
| B3 | VGGT + JEPA | × | ✓ | ✓ | × | current dynamic prior |
| B4 | VGGT + JEPA | ✓ | ✓ | ✓ | × | predictive world |
| B5 | VGGT + JEPA | ✓ | ✓ | ✓ | ✓ | full model |

最关键差值：

\[
\Delta_{\mathrm{access}}
=
B2-B1
\]

如果几何 supervision 有帮助，但 direct access 没帮助，则不能声称 planner 使用 geometry memory。

---

# 26. Teacher-Specific Controls

至少实现：

| Control | 目的 |
|---|---|
| No teacher | 基础监督 |
| Random frozen teacher | 排除结构/参数影响 |
| Scene-shuffled teacher | 排除 feature statistics |
| GT-task MLP teacher | 排除 privileged deep supervision |
| Generic V-JEPA | 通用 temporal prior |
| Driving-JEPA | 驾驶域 prior |
| Online teacher upper bound | 测量蒸馏损失 |

核心比较：

\[
RealTeacher
>
GTTeacher
>
RandomTeacher
\]

才支持较强的：

> foundation pretrained prior 具有独立价值

这一论文主张。

若：

\[
RealTeacher
\approx
GTTeacher
\]

则必须诚实修改论文主张：

> structured privileged supervision 有效，但 foundation-model prior 没有提供明显额外收益。

---

# 27. Supervision × Access 2×2

必须单独做：

| World Supervision | Planner Access | 解释 |
|---|---|---|
| × | × | baseline |
| ✓ | × | auxiliary/deep supervision |
| × | ✓ | extra architecture / capacity |
| ✓ | ✓ | full |

只有：

\[
S_{\mathrm{sup+access}}
>
S_{\mathrm{sup-only}}
\]

才能证明：

> planner 直接读取 world representation 有独立价值。

---

# 28. Future Representation Ablation

相同 planner 下比较：

| Future Representation | 说明 |
|---|---|
| None | current only |
| Dense field | dense spatial future |
| Sparse spatial tokens | compact geometry/dynamics |
| Intent latent | planning-oriented future |
| GT future oracle | upper bound |

重点指标：

- navhard；
- TTC；
- collision；
- braking；
- cut-in；
- crossing；
- inference cost。

---

# 29. Memory Storage vs Reader Ablation

需要回答最初关于 slots 的问题。

比较：

1. **Static world slots**
2. **Dense world field**
3. **Dense field + static queries**
4. **Dense field + trajectory-conditioned queries**
5. **Online teacher + trajectory queries**

这组实验回答：

> world knowledge 应该存在哪里？

以及：

> slot 是更适合作为 memory，还是 reader？

本研究的主假设是：

\[
\boxed{
\text{Structured/Dense Memory}
+
\text{Sparse Trajectory-Conditioned Read}
}
\]

优于：

\[
\text{Sparse Generic Memory}
+
\text{Sparse Read}
\]

但最终必须由实验验证。

---

# 30. Causal Knowledge Utilization Validation

仅看 EPDMS 不够。

需要建立：

\[
\text{external prior}
\rightarrow
\text{world memory}
\rightarrow
\text{world read}
\rightarrow
\text{trajectory}
\]

证据链。

---

## 30.1 World Access Removal

推理时：

\[
R_{\tau}=0
\]

比较轨迹变化和 benchmark score。

---

## 30.2 World Memory Shuffle

将 scene A 的 world memory 输入 scene B planner：

\[
M_B\leftarrow M_A
\]

如果 planner 真正依赖 memory，应导致显著且方向合理的性能下降。

---

## 30.3 Dynamic-Agent Occlusion

分别遮挡：

- influential lead vehicle；
- crossing pedestrian；
- cut-in vehicle；
- irrelevant distant vehicle；
- random background patch。

比较：

\[
\Delta\tau
\]

和：

\[
\Delta R_{\tau}
\]

要求：

\[
\Delta_{\mathrm{influential}}
>
\Delta_{\mathrm{irrelevant}}
\]

---

## 30.4 Counterfactual Visual Nuisance

只改变：

- weather；
- illumination；
- texture；
- color style。

保持 geometry/dynamics 不变。

理想：

\[
\tau(x)
\approx
\tau(I_{nuisance}(x))
\]

---

## 30.5 Geometry Intervention

人工改变：

- obstacle position；
- free-space boundary；
- road geometry。

要求 trajectory change 方向符合物理约束。

---

# 31. Benchmark

## 31.1 NAVSIM-v2 navtest

用途：

- 快速开发；
- 大规模消融；
- representation/pretraining 迭代。

---

## 31.2 NAVSIM-v2 navhard

作为论文主要 benchmark。

要求：

- Stage 1；
- Stage 2；
- final score；

分别报告。

重点观察：

- difficult interaction；
- TTC；
- collision；
- lane；
- progress。

---

## 31.3 Bench2Drive

用于：

- real closed-loop interaction；
- reactive agent behavior；
- long-tail interaction；
- final planning transfer。

重点场景：

- lead braking；
- cut-in；
- crossing pedestrian；
- lane change；
- unprotected turn；
- merging；
- red-light；
- blocked lane。

---

## 31.4 CARLA / Bench2Drive Branch Rollout

只有到这一阶段，才学习：

\[
M_{t+1:t+H}
=
f(
M_t,\tau
)
\]

真正的 action-conditioned interaction model。

通过：

\[
(s_t,\tau_i,s_{t+1:t+H}^{(i)})
\]

收集相同起始世界下不同 ego action 的 response。

然后可以扩展：

\[
M^{future}(\tau)
=
M^{exo}
+
\Delta M^{interaction}(\tau)
\]

这属于第二阶段研究，不属于第一版 MVP 的必要组成。

---

# 32. 主要指标

不要只报告一个 aggregate score。

## Planning

- EPDMS；
- navhard Stage 1 / Stage 2；
- collision；
- DAC；
- TTC；
- traffic light；
- lane keeping；
- progress；
- comfort。

## Representation

- depth error；
- 3D point error；
- free-space IoU；
- future latent error；
- temporal consistency；
- motion probe；
- TTC probe。

## Consequence

- clearance error；
- TTC error；
- collision classification；
- trajectory pair ranking accuracy；
- unsafe-vs-safe margin。

## Efficiency

- parameters；
- FLOPs；
- peak memory；
- teacher cache size；
- training speed；
- inference latency。

---

# 33. 统计规范

所有关键结果：

- 普通 ablation：至少 3 seeds；
- 最终主结果：建议 5 seeds；
- 使用相同 dataset/split；
- 使用相同 checkpoint-selection rule；
- 报告 mean ± std；
- 对 paired scene predictions 做 bootstrap CI；
- 不报告 best seed 代替均值；
- 对 0.x 增益尤其需要显著性检验。

---

# 34. Go / No-Go Criteria

## Gate A：External Prior 是否真的有效

要求：

\[
S_{\mathrm{RealTeacher}}
>
S_{\mathrm{GTTeacher}}
\]

并且最好置信区间不跨零。

否则：

> 不再声称 foundation model prior 提供额外知识。

---

## Gate B：Planner 是否使用了 world representation

要求：

\[
S_{\mathrm{Access}}
>
S_{\mathrm{NoAccess}}
\]

否则：

> world representation 只是辅助训练正则。

---

## Gate C：Future Prediction 是否有规划价值

要求：

\[
S_{\mathrm{PredFuture}}
>
S_{\mathrm{CurrentOnly}}
\]

尤其在：

- navhard；
- TTC；
- collision；
- dynamic subsets；

上有一致趋势。

否则：

> future model 不进入第一版最终模型。

---

## Gate D：Consequence Grounding 是否有价值

要求：

\[
S_{\mathrm{Consequence}}
>
S_{\mathrm{NoConsequence}}
\]

且改善不是来自额外参数。

否则删除 consequence branch。

---

## Gate E：Distillation Gap

定义：

\[
Gap_{\mathrm{distill}}
=
S_{\mathrm{onlineTeacher}}
-
S_{\mathrm{student}}
\]

若 gap 很大：

> 当前研究瓶颈在 prior internalization，而不是 planner。

后续优化应聚焦：

- adapter；
- representation target；
- distillation objective；
- spatial topology；

而不是继续叠加 planning modules。

---

# 35. 研究实施顺序

不要直接实现完整模型。

---

## Milestone 0：Pure VLM+DiT Baseline Lock

目标：

- 复现现有纯轨迹结果；
- 固定数据、训练、评测；
- 保存逐场景 prediction；
- 建立 3-seed baseline variance。

输出：

```text
baseline/
  config
  checkpoint
  predictions
  navtest metrics
  navhard metrics
```

后续所有结果都必须以此为比较对象。

---

## Milestone 1：Geometry Only

实现：

- VGGT offline cache；
- ego-coordinate transform；
- geometry adapter；
- multi-scale geometry memory；
- geometry auxiliary heads。

实验：

```text
B0 Pure trajectory
B1 + structured geometry supervision
B2 + VGGT prior
B3 + VGGT prior + world access/refine
```

这一步首先回答：

> external geometry prior 是否超过普通 geometry supervision？

以及：

> planner direct geometry access 是否有价值？

---

## Milestone 2：World Reader + Refiner

不做 dynamics。

只使用：

\[
G_t
\]

构造：

\[
\tau^0
\rightarrow
Read(G_t,\tau^0)
\rightarrow
\tau^1
\]

测试：

- center sampling；
- tube sampling；
- deformable sampling；
- 1/2/4 refinement。

如果 geometry access 都无法显著提升 planning：

> 暂停 future world 开发，先解决 representation/planner interface。

---

## Milestone 3：Driving Dynamics Prior

实现：

- Drive-JEPA / V-JEPA cache；
- temporal adapter；
- no-teacher / generic / driving teacher control；
- current-history prior transfer。

先不预测 future。

验证：

\[
Geometry
\quad vs \quad
Geometry+DynamicsCurrent
\]

---

## Milestone 4：Predictive Memory

实现：

\[
M_t\rightarrow M_{t+1:t+H}
\]

比较：

- dense；
- sparse；
- intent。

同时实现：

- same future supervision across all teacher controls；
- GT future oracle upper bound。

---

## Milestone 5：Planning Consequence Grounding

实现：

- perturbed trajectory generator；
- static/non-reactive physical metric；
- training-only consequence head。

重点比较：

\[
Prior
\]

和：

\[
Prior+Consequence
\]

---

## Milestone 6：Final NAVSIM-v2 Study

完成：

- teacher control；
- access control；
- future control；
- slot-vs-field；
- online-vs-distilled；
- efficiency。

---

## Milestone 7：Bench2Drive Closed Loop

验证：

- dynamic interaction transfer；
- safety；
- long-tail；
- closed-loop robustness。

只有这一阶段之后才决定是否开发：

> action-conditioned interactive world model。

---

# 36. 基于 DriveDreamer-Policy 的实现原则

再次强调：

\[
\boxed{
\text{DriveDreamer-Policy 只作为 VLM + DiT 框架}
}
\]

不要：

- 复用其 world-query assumptions；
- 延续其 depth/video branch；
- 依赖其 auxiliary task design；
- 以其 world model 为我们的起点。

---

## 36.1 应保留

- visual/VLM backbone；
- multimodal input pipeline；
- navigation/ego inputs；
- action representation；
- DiT / flow-matching trajectory decoder；
- existing trajectory loss；
- inference integration；
- NAVSIM evaluation pipeline。

---

## 36.2 新建模块建议

逻辑目录：

```text
world/
├── geometry/
│   ├── geometry_adapter.py
│   ├── ego_projector.py
│   ├── geometry_memory.py
│   └── geometry_losses.py
│
├── dynamics/
│   ├── dynamics_adapter.py
│   ├── temporal_encoder.py
│   ├── future_predictor.py
│   └── future_losses.py
│
├── teachers/
│   ├── vggt_adapter.py
│   ├── jepa_adapter.py
│   ├── cache_dataset.py
│   └── cache_manifest.py
│
├── planner/
│   ├── trajectory_tube_reader.py
│   ├── world_refiner.py
│   └── consequence_head.py
│
└── utils/
    ├── coordinates.py
    ├── trajectory_geometry.py
    └── perturbations.py
```

具体文件路径应以本地 repository 为准，不应为了匹配上述示意而破坏现有代码结构。

---

# 37. Teacher Cache

训练时不实时运行大 teacher。

建议 cache manifest：

```yaml
scene_token: ...
frame_token: ...

vggt:
  version: ...
  coordinate_frame: current_ego
  feature_level: ...
  depth_path: ...
  point_map_path: ...
  confidence_path: ...

jepa:
  version: ...
  model: ...
  domain: driving
  history_feature_path: ...

metadata:
  cameras: [...]
  timestamps: [...]
  ego_pose: ...
  intrinsics: ...
  extrinsics: ...
```

要求：

- teacher version 可追踪；
- 坐标系显式记录；
- dtype 记录；
- feature shape 记录；
- cache generation code 可复现。

---

# 38. Zero-Init Contract

所有新 planning path 都需要满足：

\[
f_{\mathrm{new}}(x;\theta_0)=0
\]

因此：

\[
\tau_{\mathrm{new}}
=
\tau_{\mathrm{baseline}}
+
f_{\mathrm{new}}
\]

初始：

\[
\tau_{\mathrm{new}}
=
\tau_{\mathrm{baseline}}
\]

必须写 unit test 检查：

```text
baseline_output == new_model_output
```

在：

- eval mode；
- same seed；
- zero-init；
- world modules enabled；

情况下数值误差应接近 floating-point tolerance。

---

# 39. 坐标 Contract

统一定义：

- ego origin；
- x/y orientation；
- meters/unit；
- yaw sign；
- camera extrinsics direction；
- historical frame compensation；
- trajectory waypoint coordinate；
- BEV resolution；
- future timestep definition。

禁止每个模块自己实现一套 coordinate convention。

所有：

```text
camera -> ego
ego_old -> ego_current
world -> ego
trajectory -> BEV index
BEV index -> metric point
```

集中在同一个 coordinate utility 中。

必须有 round-trip test。

---

# 40. 失败模式与对应处理

## 40.1 Geometry probe 提升但 planning 不提升

解释：

> world representation 学到了 geometry，但 planner 没利用。

优先检查：

- planner access；
- reader locality；
- coordinate alignment；
- refiner capacity；
- gradient flow。

不要继续增加 teacher。

---

## 40.2 Teacher KD 很好但 GT-task teacher 一样好

解释：

> foundation prior 没有提供独立价值。

处理：

- 修改论文主张；
- 尝试 stronger domain-adapted teacher；
- 或转向 structured world supervision。

---

## 40.3 Future loss 下降但 planning 不提升

解释：

\[
\text{world prediction quality}
\not\Rightarrow
\text{planning value}
\]

检查：

- representation 是否预测过多无关信息；
- future representation 是否太 dense；
- planner 是否访问 future；
- future horizon 是否不匹配 planning horizon。

优先测试 compact intent/sparse future。

---

## 40.4 Online teacher 显著好于 distilled student

说明：

> prior internalization 是主要瓶颈。

重点改：

- layer selection；
- residual fusion；
- spatial alignment；
- local KD；
- retention；
- adapter capacity。

不要继续增加 planner modules。

---

## 40.5 World refiner 改善 navtest 但 navhard 不改善

可能：

- 学到 metric-specific static correction；
- dynamic future 不够；
- memory 对 closed-loop state shift 不稳。

重点看：

- TTC；
- dynamic subsets；
- Stage 2；
- Bench2Drive。

---

# 41. 最终论文故事

论文核心问题：

> **How should external world knowledge be internalized and utilized by a VLA planner?**

答案：

> 外部 world knowledge 不应简单作为 auxiliary prediction target，也不应过早压缩成几个 scene-global slots。几何和动态 prior 应分别进入一个与 VLM semantic path 互补的 physical world representation；未来预测必须与 foundation prior transfer 解耦；最终 planner 通过轨迹条件化局部读取直接访问世界信息，并使用 residual refinement 改变最终动作。

形成三层证据链：

\[
\boxed{
\text{Acquire External Prior}
}
\]

证明：

\[
RealTeacher
>
GT/RandomControl
\]

↓

\[
\boxed{
\text{Build Predictive World Representation}
}
\]

证明：

\[
PredFuture
>
CurrentOnly
\]

↓

\[
\boxed{
\text{Use It for Planning}
}
\]

证明：

\[
PlannerAccess
>
NoAccess
\]

最终只有三个条件都成立，才能声称：

> **外部 foundation model 的知识被成功迁移，并被下游轨迹规划实际使用。**

---

# 42. 最终三个创新点

## Innovation 1 — Dual-Path Foundation Prior Internalization

共享 VLM visual encoder 后分成：

\[
\text{Semantic VLM Path}
+
\text{Physical World Path}
\]

VGGT/Driving-JEPA 分别向物理路径注入 geometry/dynamics prior，而非要求所有精细物理信息经过语言瓶颈或压缩到固定 world slots。

---

## Innovation 2 — Confound-Separated Predictive World Representation

严格拆开：

\[
\text{foundation prior}
\]

和：

\[
\text{future privileged supervision}
\]

所有 teacher control 使用相同 future target，只让 external model 负责 current/history prior transfer，从实验设计上解决“teacher knowledge vs future GT deep supervision”的混淆。

---

## Innovation 3 — Trajectory-Conditioned World-Grounded Refinement

保留强 VLM+DiT planner 作为 draft generator。

随后：

\[
\tau^0
\rightarrow
\text{trajectory-tube world read}
\rightarrow
\Delta\tau
\rightarrow
\tau^1
\]

world representation 直接进入最终 action computation，同时避免高噪声 diffusion trajectory query 的不稳定性。

---

# 43. 第一版最终模型

第一篇工作最终模型建议控制为：

```text
VLM visual encoder
        │
        ├── semantic VLM path
        │
        └── physical world path
                 │
           VGGT geometry KD
           Driving-JEPA dynamics KD
                 │
         current world memory
                 │
          future predictor
                 │
        predictive world memory
                 │
VLM semantics ── DiT
                 │
              draft τ0
                 │
          trajectory reader
                 │
             world evidence
                 │
              refiner
                 │
              final τ1
```

不包含：

- Cosmos；
- pixel video generation；
- world-query generation；
- fixed expert slots；
- candidate reranking；
- EPDMS critic；
- action-conditioned counterfactual world rollout；
- high-noise in-DiT world querying。

---

# 44. 最重要的 Go/No-Go 总原则

如果最终：

\[
RealTeacher
\approx
GTStructuredSupervision
\]

那么应停止声称：

> foundation model knowledge transfer 是核心收益来源。

如果：

\[
PredFuture
\approx
CurrentOnly
\]

那么删除 future module。

如果：

\[
PlannerAccess
\approx
NoAccess
\]

那么删除 world-refinement 主张。

只有：

\[
\boxed{
RealTeacher
>
GTControl
}
\]

且：

\[
\boxed{
PredFuture
>
CurrentOnly
}
\]

且：

\[
\boxed{
PlannerAccess
>
NoAccess
}
\]

三者同时成立，才能形成最强论文结论：

> **VLA 通过外部 foundation priors 学到了超越普通深监督的物理世界知识，这些知识能够预测与规划相关的未来状态，并通过显式 world-to-action interface 被轨迹规划器真实使用。**

---

# 45. 一句话总结

最终方案可以概括为：

\[
\boxed{
\textbf{Learn a physically grounded world representation, predict only what planning needs, and force the planner to read it before committing to the final trajectory.}
}
\]

中文：

> **学习具有物理约束的世界表征，只预测规划真正需要的未来，并在输出最终轨迹前强制规划器读取这些世界知识。**

这应当作为后续算法开发、实验设计和论文写作的统一主规范。
