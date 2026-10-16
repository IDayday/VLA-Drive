# WM-VLA 时空表征与决策后果内化学习
## 完整实验计划、多 Agent 分工与训练前验收合同

**版本**：v1.0，2026-10-17  
**用途**：交给 Codex 执行研究与开发；本文件是研究计划，不是已完成实验报告。  
**项目**：`IDayday/VLA-Drive`  
**建议仓库位置**：`docs/research/WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md`  
**新任务建议分支**：`research/st-world-consequence-flow-<实际日期>`

---

## 0. 一页决策摘要

### 0.1 研究主线

建立两种互补学习能力，最终直接改变 VLA 的原始轨迹生成：

1. **时空理解**：利用当前与历史三视图，从空间细节充分的教师特征中学习当前状态和未来变化。
2. **决策后果内化**：利用 NAVSIM 仿真的真实道路、责任碰撞、TTC 与进度反馈，直接训练原 Flow-Matching / DiT 策略；先验证真值反馈，再研究后果预测。

部署目标：

```text
当前与允许的历史三视图 + 导航 + 合法ego状态
                         ↓
             Qwen / 有预算约束的历史融合
                         ↓
                 原DiT / 10步Flow生成
                         ↓
                    唯一最终轨迹
```

不增加固定的生成后修正器、在线真值查询、候选轨迹评分择优或额外规划器。训练时允许多候选采样、仿真评价和辅助头；这些不等于部署需要相同流程。

### 0.2 不再混淆的两条梯度路径

```text
时空辅助损失 → W / 历史融合 / Qwen可训练部分 → 改善规划条件
仿真后果驱动的策略目标 → 原DiT速度场参数 → 改变实际生成分布
```

只训练一个准确的后果预测头，不等于已经完成策略学习。原始 NumPy/几何 PDMS 分数没有 PyTorch 梯度，也不能直接加到 `loss` 后声称约束了生成。[P03][P04]

### 0.3 本轮执行优先级

| 级别 | 必须完成或探索的内容 | 暂不做什么 |
|---|---|---|
| P0 | 核实论文与官方代码；恢复正确工程；数据/输入/目标合同；独立正确性与效率验收 | 不启动未审查的正式训练 |
| P1-W | 当前细网格监督、W容量、真实历史、绝对未来与时间差分的分阶段对照 | 不同时叠加多个视觉教师 |
| P1-C | 官方后果提取；FPO真值策略训练；分项筛选；真实生成轨迹评价 | 不先训练复杂后果模型，不恢复MIN修正器 |
| P2 | 视频/几何教师的定向比较；可微代理与DFQL式梯度注入 | 不把代理称为官方PDMS梯度 |
| P3 | 只预测已验证有用的后果；时空×后果联合验证 | 不开展所有配置的笛卡尔积 |

**本轮不是“把所有待定方向各跑100k”。先完成可复用实现、真实短程配对与必要的阶段确认，再将有限入围组推进共同终点。失败分支保留证据，其他独立分支继续。**

---

## 1. 两条不可绕过的规定

### R1：方法和实现必须有可追溯依据

每个拟采用的非平凡设计，先建立一张 `Design Evidence Card`，再编码或改变正式配方。卡片至少包含：

```yaml
design_id: W-HISTORY-001
question: 要解决的具体问题
claim_type: literature | existing_evidence | new_hypothesis
paper:
  title: 完整题名
  url: 论文/作者主页/会议官方页面
  section_or_equation: 实际阅读的节、式、表或算法
  read_depth: abstract_only | full_text | paper_and_code
code:
  repository: 官方仓库或已验证本地实现
  commit: 精确SHA
  path_and_symbol: 文件和函数/类
  status: inspected | reference_executed | port_verified | unavailable
mapping: 文献/源码到本项目的对应关系
new_adaptation: 哪些是我们新增的适配，不能冒充原论文方法
risks: 域差异、输入差异、损失/精度/可用信息等限制
control: 用什么对照验证
acceptance_evidence: 什么测试/实验能支持或否定这一实现
```

要求：

- 方法事实优先论文全文、作者项目页、会议页面；技术行为以固定版本官方源码和官方文档为准。
- 博客、综述和搜索摘要用于寻找线索，不能代替关键公式、标签规则或代码接口的核验。
- 数值如图像尺寸、W96、K96、学习率和采样比例可作为**待验证工程起点**，必须标注为我们提出的假设，不声称文献已证明其最优。
- 新方法允许原创；“有依据”不是要求已有论文完整做过同一方法，而是清楚区分成熟构件、适配和新假设。
- 对代码，必须追踪实际调用链、数据布局、损失归约和梯度；只读 README 不足以证明实现正确。
- 无官方代码时先写最小参考实现与可解/数值对照；如关键方法定义仍无法核实，该分支为 `BLOCKED_EVIDENCE`，不得凭记忆训练。
- 不以“别人做过”否定研究，也不以“组合了多个模块”宣称创新；评价最终可复现增量。
- 新设计、重要变更、论文/代码不一致必须经研究证据Agent和主Agent记录。禁止只在报告末尾堆一串不对应方法的引用。

### R2：正式训练前同时通过正确性和资源效率审查

每个正式实验必须绑定以下三项批准：

1. **研究依据批准**：关键设计卡完整，参考代码与新适配明确。
2. **实现正确性批准**：由未主要编写该模块的审阅Agent出具，检查真实数据、真实梯度、标签与部署边界。
3. **资源效率批准**：实测合适的多进程/多GPU训练、采样、教师提取和评价拓扑；不是简单设置 `num_workers>0` 或“用满所有卡”。

仅允许在批准前运行已登记的单元测试、参考数值对照、有限 `SMOKE/PROFILE`；学习smoke例如32—64个真实场景、最多64个优化更新；独立PROFILE按第17节执行，每候选预登记至多256个完整更新以覆盖预热和两个稳定窗口。二者的权重都不进入正式初始化，也不算正式效果证据。正式筛选训练即使只有数千步，仍须通过批准。

审批针对**实际整合提交与配置**，不是各Agent在不同分支上分别说“通过”即可。运行入口必须检查 approval 中的源码、配置、数据、目标、教师、精度、资源和有效批量指纹，不能只检查文件存在。

```json
{
  "experiment_id": "T3_seed42",
  "source_sha": "<actual>",
  "config_sha256": "<actual>",
  "data_manifest_sha256": "<actual>",
  "target_schema": "<actual>",
  "teacher_ids": ["<actual>"],
  "environment_fingerprint": "<actual>",
  "precision_policy": "<actual>",
  "global_batch": 32,
  "world_size": 8,
  "resource_lease": "<authorized>",
  "evidence_review": {"status": "PASS", "reviewer_agent": "...", "artifacts": []},
  "correctness_review": {"status": "PASS", "reviewer_agent": "...", "artifacts": []},
  "efficiency_review": {"status": "PASS", "reviewer_agent": "...", "artifacts": []},
  "budget_and_schedule": "<registered>",
  "formal_launch_allowed": true
}
```

这是工程记录，不是加密安全认证。不能让实现者自行生成三个PASS来代替独立审阅。重大源码/标签/冻结范围/精度/批量/资源变化使受影响批准失效；文档拼写修正不触发全仓重审，主Agent记录影响范围并继承未受影响证据。

---

## 2. 既有实验与代码起点

### 2.1 已确认的经验边界

- C0/C1/C4都只保留每视图48个目标位置和48个W；C2/C5的高W配置没有正式长训规划结论。不能据C0/C1/C4否定空间容量问题。[P01]
- S0相对C1还改变了时间曝光、归一化及权重；不能归因于单一“多帧”因素。[P01]
- S2/S3的视频教师没有稳定优于S0；本轮视频实验必须是在新的输入与目标合同下的定向验证，而非重复同一实验。[P01]
- GT-MAE、直接W拼接、生成后修正和MIN均保留为历史证据，不是本轮部署模块。MIN正式预测关系阶段为NOT_RUN，不把小集smoke写成效果。[P02]
- 原始GT日志、原FM与单候选10步推理是稳定参照；不要恢复r1/V12、新制动权重或历史未获接受的训练目标。[P01]
- 用户现已允许训练与推理使用真实历史图像。旧“仅当前图像”限制被这一要求替代；未来图像/GT地图/对象状态仍不得进入部署。

### 2.2 先定位，再迁移

| 资产 | 固定入口或已知版本 | 本轮用途 |
|---|---|---|
| 历史方法/结果 | `0c03b32f17bfe0d9dd82ff95f141739258e6c33c` 的 `reports/experiment_results/20261014/` | 准确复用已知事实，避免重跑旧矩阵 |
| S系列执行源码 | `1493deda247107efe076b9294863a6753391298a` | Qwen、原DiT、W和纯当前接口 |
| 最新已提供报告 | `7ffd77d9d0d6acbe6f7d39852e57eccd02160ae9` 的 `reports/minimal_repair/20261015/` | 资源边界、历史纠偏结论；不启动旧任务 |
| 原始动作头 | `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | 线性FM路径、噪声/时间、10步Euler、动作编码 |
| 原始scorer | `navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py` | 真实后果和canonical分数 |
| 原始执行器 | 同目录 `simulation/pdm_simulator.py` | LQR跟踪＋自行车模型后的实际状态 |
| 已有训练/评估/恢复/profile | 通过报告及调用链定位 | 复用，不另造通用训练平台 |

报告SHA不等于每个run训练SHA。已有环境历史为PyTorch 2.5.1+cu124、Transformers 4.57.0、DeepSpeed 0.16.9等，实际运行先核对，不依据最新网页接口盲目升级。

### 2.3 资源与历史任务

读取当前共享授权/禁用清单。`vla-zt2`及其他已撤销临时主机不得连接或派发。历史上使用过的主机不自动构成本轮授权。G2等其他任务只读取完成产物，不热修改、不迁移、不停止。旧RAM副本已释放时使用身份相同的持久原件，不假定历史RAM路径仍存在。[P02]

新代码、标签、运行目录独立；不自动merge、force push、reset或stash其他工作区。不得在main/default分支上无审批写入。

---

## 3. 两条学习路径与信息合同

令 `O_{≤0}` 为允许的当前/历史三视图、导航、合法ego状态与图像标定：

\[
H,W=F_\theta(O_{\le0}),\qquad q=\Phi_\phi(\epsilon;H).
\]

`Φ`仍是原DiT的10步生成；flow时间记为`τ`，驾驶物理时间记为`t`，两者必须分开命名。

### 3.1 部署输入

- 相机：当前及历史的三个原前视，相机身份固定。
- 历史候选：`t={-1.0,-0.5,0}`秒，核实每个数据包实际时间戳。
- 导航、当前/合法历史ego状态；仅到决策时刻为止。
- 缺历史显式mask，不用未来补齐，不跨log读取；重复当前只作为明确的无运动控制，不冒充真实历史。

### 3.2 训练标签

- 当前/未来DINO目标；视频或几何专家仅在对应分支中产生标签。
- 实际生成轨迹在锁定PDMS执行器中的反馈。
- 诊断所需地图/目标/深度均置于标签侧。

### 3.3 梯度与缓存

- 教师冻结，目标`detach`；学生的W、历史融合和允许更新的Qwen语言层不能因缓存而被意外冻结。
- 时空主实验更新共享表示及原动作头；冻结W小探针仅用于容量/代码诊断，不作VLA能力提升结论。
- 后果真值第一阶段可冻结Qwen，仅训练原DiT及其既有投影，复用对应检查点H_A缓存；所有对照相同。
- 联合训练若解冻Qwen，旧H_A/W缓存失效；不得跨模型身份读取冻结表征。
- 冻结视觉backbone的确定性视觉输出可以缓存；历史记忆若依赖上下文或模型更新，不能无条件跨step缓存。
- 推理不使用DINO/视频教师、后果预测头、价值网络或仿真器来选择/修正轨迹。若后续研究改变这一点，须用户确认并单列部署合同。

---

## 4. 参考文献与代码：必须先研究的最小集合

这里列的是入口，不等于代码全部已经验收。执行Agent必须补精确SHA、符号和本地参考测试。

| ID | 工作/文档 | 借鉴内容 | 不能直接搬用的内容 | 本文件准备时核实深度 |
|---|---|---|---|---|
| R01 | DINO-Foresight | 时空分离注意力、多层特征、通道PCA、低分辨率到高分辨率训练 | DINOv2 predictor权重不能自动用于DINOv3；原完整训练预算不照搬 | 论文全文与官方README |
| R02 | DINOv3 | dense patch、分辨率适配、官方feature接口 | post-norm不等于L2单位向量，register不属于空间patch | 论文入口；运行接口需复核 |
| R03 | ResWorld | 静态/动态分工与几何对齐的动机 | 不能据其结果宣称相机DINO做差必然有效；不复用部署后修正流程 | 论文 |
| R04 | V-JEPA 2.1 | 真正的视频密集时空特征 | 双向clip token不等于因果时刻状态；不同teacher MSE不可直接比较 | 论文入口/摘要 |
| R05 | Video Depth Anything | 视频深度的一致性与时序梯度 | 尺度、动态错误、置信度须驾驶数据验证 | 论文入口/摘要 |
| R06 | Flow Matching Policy Gradients / FPO | 用CFM代理ratio接受不可微真实反馈，PyTorch参考可复用 | 不是精确似然；不能把8轨迹点伪装成8次MDP环境交互 | 论文全文、作者说明、官方仓库README |
| R07 | Direct Flow Q-Learning / DFQL | 终端后果/价值梯度作用于速度场，避免完整BPTT | 原JAX Q-learning与Bellman不能原样套到固定4秒contextual-bandit | ICML官方记录、官方代码入口 |
| R08 | Reparameterization Flow Policy Optimization / RFO | 可微生成＋可微执行，及稳定性问题 | 不可微NAVSIM没有现成这种梯度；需要验证可微代理 | 论文全文 |
| R09 | PyTorch DDP与Performance Tuning | 一进程一卡、数据切分、累积/通信、异步输入、同步测时 | 最新文档参数未必兼容项目固定版本 | 官方文档 |
| R10 | Codex Subagents/Worktrees | 真正分工、隔离上下文、独立审阅、集中整合 | 不伪造多Agent记录，不照搬与当前CLI不兼容的配置 | 官方文档 |

对可选VGGT、FeatUp、Flamingo式历史resampler：先由文献Agent查到作者一手资料和官方实现，再建立卡片；不在本文件中将其当作已选择的必需组件。

---

## 5. W线：空间分辨率、目标网格、W与输出query分别管理

### 5.1 四个独立预算

1. 教师输入像素数：影响离线特征提取与空间可见细节。
2. 教师原生网格：保留多少空间位置。
3. Qwen内W数量：影响学生容量与整个语言主干序列长度。
4. 辅助输出query数量：影响轻量读出头和loss成本，不必等于W数量。

固定DINOv3 ViT-L/16的合法通用权重，实际维度从模型读出。参考画布`384×288`（宽×高）对应`24×18=432`patch/视图。保持原始允许ROI，几何变换确定；源图不是4:3时明确说明resize/crop，不声称天然同宽高比。不放大旧128×96缓存来制造高分辨率标签。

### 5.2 先做四组有限空间筛选

| ID | teacher输入 | 目标 | W/视图 | 辅助监督位置/视图/时间 | 核心比较 |
|---|---|---|---:|---:|---|
| S-P0 | 256×192 | 2×2池化成8×6 | 48 | 全48 | 当前已知结构的重新登记控制 |
| S-P1 | 384×288 | 3×3池化成8×6 | 48 | 全48 | 同网格下教师分辨率影响 |
| S-P2 | 384×288 | 原生24×18，不空间平均 | 48 | 均匀采样96 | 保留细网格的监督方案 |
| S-P3 | 与S-P2相同 | 同上 | 96 | 同样96个位置 | 在同目标/同采样下增加W容量 |

第一轮均只使用当前输入和当前DINO辅助；旧未来/MAE监督在这组新空间pilot中统一关闭，但原代码、权重和既有结果保留，不修改历史实验。新的`current-only`实验不是历史C/S全四任务复现。S-P2−S-P1同时改变目标粒度和监督位置数量，应称“细网格监督方案”比较，不能称只改变W或纯分辨率单变量。

输出query带view、归一化二维位置、物理时间；从整组W读取。严禁将一个W硬说成一个物体或原图patch；也不将W随意reshape成有几何意义的BEV。

从同一S0迁移到W96时，显式记录query扩展、位置编码与state-dict映射；不能strict=False静默丢权重，也不能声称token数变更前后的初始输出逐位相同。公共参数及专用初始化RNG保持可复核。

不要一开始增加到192W/视图。S-P3若显示容量瓶颈且资源可行，才预登记追加192，不能一组OOM就偷偷减少图像/任务。

### 5.3 压缩和存储

- 先压教师channel，再考虑平均空间。PCA只在训练数据拟合；默认不whiten，保持目标尺度可追溯。
- 通道维度候选256/384先做离线重建和相同小读出探针，选择一套并冻结；不是两套都乘完整训练矩阵。
- 若PCA启用，S-P各匹配组使用同一个在高分辨率训练特征上拟合的变换；报告原空间与压缩空间指标。
- 均匀采样patch损失按有效元素归一化；时间差分必须在同view、同位置或已定义对应位置上计算。
- 稀疏监督减少学生decoder计算，不自动降低缓存大小。全网格缓存、固定位置池、有限shard轮换三种存储模式要先测成本。
- 不能每次监督新的位置但缓存只有旧固定位置，再用插值/均值冒充未存目标。
- 每个cache包含模型/层/norm/ROI/grid/dtype/源时间/划分/压缩身份；教师像素和特征可以按相同图像身份去重。

空间缓存估算：`N_scene × unique_time × 3 × Hpatch × Wpatch × D × bytes`。先统计图像真实去重率。101592场景、5个时间、432patch、1024维FP16约为1.23TiB的未去重张量量级；不得不经估算直接全量生成。

### 5.4 空间能力评价

PCA彩图只作可视化，不作为信息保留证明。使用同容量、固定划分的小读出头检查至少两种能力：小车辆/目标区域区分、道路边界或有效深度。所需GT仅训练标签，报告投影可见性和有效人口。当前重建MSE、细节读出、PDMS分别报告，不用更低raw MSE直接选最终模型。

---

## 6. W线：历史输入与时序任务

### 6.1 历史接入的两个工程版本

**正确性参考**：按当前Qwen官方接口输入真实历史多图，正确设置相机和物理时间。它可能贵，但用于核对压缩路径有没有丢掉基础时序能力，不默认长训。

**预算版本**：当前图像保持原详细路径；历史图像共享冻结视觉编码器，经两层轻量时空融合和少量resampler tokens进入Qwen；所有历史信息先于W/action query。起点为每历史时刻、每view 16个记忆位置，共96个历史位置；这是待验证预算，不是已证最优值。具体resampler优先复用核实后的成熟实现。

要求：

- 历史按camera、时间、二维位置保留身份，提供合法ego相对运动；不把时间维和视角维混合。
- 不使用将来帧、轨迹GT、未来mask决定学生输入。
- 当前/历史特征预处理保持允许ROI，不能历史支路偷看额外视图。
- mask全部历史时应有明确无历史模式，不能数值NaN；缺历史不删除整场景。
- 有/无历史组尽量匹配历史模块参数与slot定义；真实新增观测和额外编码成本单列，不能宣称严格等算力。
- 原Qwen视觉编码器第一轮共同冻结；语言融合层、W、历史模块、动作头按统一配方训练。冻结视觉允许合法缓存，不冻结整个语言路径。
- 滑动窗口复用只缓存与上下文无关且冻结的视觉输出；实际部署测端到端时延及预热/首帧差异。

### 6.2 时间对照矩阵

在空间筛选固定一套目标/W后运行：

| ID | 输入 | 当前DINO | 绝对未来 | 时间差分 | 目的 |
|---|---|---|---|---|---|
| T0 | 当前 | 开 | 关 | 关 | 基础视觉控制，可与对应S-P组复用 |
| T1 | 当前＋历史 | 开 | 关 | 关 | 真实历史观测的作用 |
| T2 | 当前 | 开 | 开 | 关 | 单当前条件的未来学习 |
| T3 | 当前＋历史 | 开 | 开 | 关 | 历史基础上的未来辅助增量 |
| T4 | 当前＋历史 | 开 | 开 | 开 | 时间变化约束增量 |

关键比较：T1−T0、T3−T1、T3−T2、T4−T3。一次性堆入历史、细网格和视频teacher，无法回答任何一项原因，禁止作为唯一主比较。

未来候选时间固定`{0.5,1,2,4}`秒。每场景每更新随机请求其中一对相邻未来时刻，T2/T3/T4使用同一时间对、patch位置和监督曝光；当前监督另算，不被未来任务稀释。原S0是另一套时间曝光，不充当这些组的严格匹配控制。

### 6.3 损失定义与边界

设`F_t*`为冻结教师空间特征，`Fhat_t=P(W,t,v,xy)`。

\[
L_{cur}=\operatorname{mean}_{valid}\|\hat F_0-F_0^*\|^2,
\]
\[
L_{future}=\tfrac12\sum_{t\in\{a,b\}}\operatorname{mean}_{valid}\|\hat F_t-F_t^*\|^2,
\]
\[
L_{delta}=\operatorname{mean}_{valid}\| (\hat F_b-\hat F_a)-(F_b^*-F_a^*)\|^2.
\]

差分表示变化，不称物理速度。间隔不同按时间对分别统计；若改成除以Δt的变化率，这是新配方，重新锁定尺度。保留绝对目标，不能只学差分丢掉静止障碍和绝对内容。

禁止逐token把近零残差强制单位化，避免幅度信息丢失和小噪声放大。主模型不读取真实当前DINO target作为skip；需要skip时只能用学生自身当前预测，并明确梯度。

第一轮用确定性相同相机/归一化图像位置的差分，称视觉变化，不能称分离了真实物体运动。几何对齐为后续条件性分支：使用GT位姿/可靠深度只在label侧做静态对齐，包含遮挡、越界mask；完整光流若对齐掉对象运动，需保留位移本身，否则不得将低差分误差称动态学习成功。

当前权重可保留1。新future/delta权重由固定训练batch上的损失尺度与共享梯度一次校准；不同方法共用，不按每组开发PDMS调参。所有新loss每原始场景计算一次，不被FM repeat8放大。

### 6.4 历史与未来评价

- 当前特征：训练模板、跨场景方差、相同位置读出能力。
- 未来特征：copy-current、训练均值、历史线性外推参考；各时间/view；高变化和低变化区域均报告。
- 历史：相机不变的时间顺序反转、历史去除、重复当前参考；敏感性不直接等同有效运动理解。
- 差分：绝对预测质量、变化方向/幅度、小目标位置、静态保持不能退化。
- 最终：完整开发PDMS、分项与零分转移；冻结头探针不替代动作收益。

---

## 7. 条件性专家分支：不要重复堆教师

### 7.1 V-JEPA 2.1 对照

复用已有合法权重与adapter，但重新核实版本、输出层、时间tubelet、view顺序与目标归一化。只与T3/T4的同历史输入版本比较。

逐帧DINO与视频teacher使用同一批真实源帧；视频需要8帧时，DINO控制也使用同8帧，不能将“监督更多时刻”的收益归功于视频编码器。缺帧按clip定义处理；不复制帧凑长度。不同teacher的raw MSE、channel和时间token数量不同，比较下游能力、相对各自基线与规划，不混为同一误差排行榜。[R04]

### 7.2 几何专家

只有细节/深度/多视图一致性诊断显示缺口时，二选一试验Video Depth Anything或VGGT。优先读可解释深度/点图及置信度，不先对齐另一套全部hidden。检查尺度、遮挡、动态对象、标定和训练统计来源。[R05]

只增加一个与DINO互补的专家，设置同容量无该监督控制。它不依赖较低视频MSE获得放行；只有教师目标质量、学生可学性和成本可接受才正式训练。

### 7.3 多层特征/FeatUp

优先保留两层DINO作为有限读出诊断，避免四层全拼的巨型缓存。特征上采样必须使用可核实算法，普通插值不恢复已丢失细节；作为待证分支，不替代原生细网格实验。

---

## 8. C线：简洁后果目标与仿真合同

### 8.1 先用实际执行后的真值

执行路径：

```text
冻结的当轮old policy生成整条q
      ↓
原轨迹转换、LQR tracker、自行车模型
      ↓
模拟ego状态序列 + 固定日志环境
      ↓
canonical scorer + 不改变原计算的事件记录
```

本研究是固定上下文下一条4秒轨迹的决策优化。没有持续环境互动时，先按contextual bandit / terminal trajectory outcome建模；不把8个轨迹点或10个生成步伪装成8/10个真实MDP transition，不默认引入Bellman和折扣回报。[P03][P04]

### 8.2 初始后果合同

| 量 | 原生记录 | 初始聚合 | 解释边界 |
|---|---|---|---|
| 道路违规 D | `_ego_areas[..., NON_DRIVABLE_AREA]`，按实际运行代码核实 | 8段内任意违规、全程任意违规、首次时刻 | 完整车身规则；不是只看中心点 |
| 责任碰撞 C | 原no-at-fault循环中的事件、对象类型、原分值 | 保留0/0.5/1严重性语义与最早时刻 | 不能把所有重叠替代责任碰撞 |
| TTC违规 T | 原TTC循环和时间索引 | 任意违规及首次时刻；区间记录 | TTC不是distance/speed或精确剩余碰撞秒数 |
| 原始进度 p | `_progress_raw` | 米制进度，固定场景参考下的辅助归一化 | 不把batch依赖EP当独立固定label |

最终最小预测集合最多`8×3事件/严重性 + 1进度`，先筛后减，不默认全部必需。Comfort/DDC作为确定性评价和保护项，不再训练新的视觉head。若连续安全余量用于可微梯度，标成代理的新字段，不混入canonical真值。

事件记录要保留精确仿真时间、右边界分箱约定、event发生/无事件、有效性。无事件不能把`inf`送入MSE；延后到4秒末端不自动获得更小违规代价。首个0.5惩罚和后续0惩罚可能不同，重建最终分值须以原min/any逻辑校验，不能只记录第一条事件。

### 8.3 必做等价验证

- 仪表化前后canonical各分项和最终分数在相同输入上不变。
- 事件归约能精确恢复对应分项，不遗漏道路恢复后曾经违规的时段。
- 训练组内4条候选与官方独立评价的EP归一化不能混淆。每条候选按固定canonical参考集合评价，或使用原始进度；固定其他候选改变组大小不应改变已声明为独立标签的值。[P03]
- 每个CPU worker独享scorer实例和可变状态；原scorer缓存`_states`等，不能在多线程共享实例。
- TTC需要更远观察范围时核实metric cache时域，不能越界填零。
- 仿真异常、缓存缺失、非法轨迹是明确状态；合法PDMS=0必须保留，不能在总体分母删除。
- 仿真反馈不是自然世界反应式反事实；替代ego轨迹仍面对相同日志环境。

---

## 9. C线主路A：不可微真值反馈直接更新Flow策略

### 9.1 为什么优先FPO

FPO提供CFM loss构造的策略更新代理，允许用不可微的真实评分训练flow policy；官方有PyTorch实现入口。[R06] 本轮优先复用其更新逻辑，而不是手写一个`PDMS_tensor`加到loss。

参考形式：

\[
r^{FPO}=\exp(L_{old}^{w}(q)-L_\phi^{w}(q)),
\]
\[
L_{policy}=-\mathbb E\min(r^{FPO}A,\operatorname{clip}(r^{FPO},1-\varepsilon,1+\varepsilon)A).
\]

这只是与论文对应的结构示意。具体时间权重、noise coupling、归约、温度/比例、clip和baseline必须按官方实现及本项目动作合同核验，不把原DDP unweighted FM loss随意塞进指数。这个ratio是代理，不是精确轨迹似然比或严格KL保证。

### 9.2 迁移要求

- 每轮old policy冻结后采样，old/current使用同一已评分真实q和对应CFM noise/time计算，验证初始ratio接近1、优势符号和梯度方向。
- 原DDP使用Beta时间采样，FPO定义的时间权重和采样可能不同；逐项核对，必要时独立使用策略loss时间采样并做正确重加权，原BC/FM不偷偷改变。
- 0到1方向、动作编码x/y/sin/cos、sum/mean、32维轨迹展开均核对。不能复制机器人动作clip[-1,1]到未具有同合同的驾驶输出。
- 每scene先采样4条实际10步轨迹作为训练反馈，单次H_A共享。4是成本起点，若几乎没有优势变化先测G=4/8的信息收益再有限调整。
- 不必把全量offline cache当on-policy forever。记录policy_version、生成seed、最大重用次数和更新预算，old policy变化后刷新反馈。
- group零方差时对应policy项可为零，原FM仍训练；不能筛掉该scene或添加伪差异。
- 先共同冻结Qwen、训练原DiT/既有投影，完全无ResidualRefiner；共同控制采用相同可训练参数和optimizer重置。
- 基线继续训练必须匹配数据和BC曝光；记录新策略特有的仿真成本与候选曝光，不能假装相同wall-clock。
- 训练目标使用官方指标信息后，不再称纯IL、不泄露评分的类别。

### 9.3 最小真值筛选矩阵

| ID | 原FM锚点 | 策略真值反馈 | 目的 |
|---|---|---|---|
| C0-BC | 开 | 无 | 共同继续训练控制 |
| C1-PDMS | 开 | canonical总PDMS | 总分反馈参照 |
| C2-D | 开 | 仅道路违规代价 | 道路反馈增量 |
| C3-C | 开 | 仅责任碰撞代价 | 责任碰撞增量 |
| C4-T | 开 | 仅TTC违规代价 | 危险接近增量 |
| C5-SEL | 开 | 前述证据选定的少量安全分项＋预登记进度防退化处理 | 检验组合，不同时加入全部杂项 |

建议单项为`r_D=-any(D)`、`r_C=NC_score-1`、`r_T=-any(T)`；这是明确起点，不是文献保证的最佳reward。每组都监控全部安全与进度，不能只按自身奖励选优。

组合配方由研究Agent在训练数据反馈分布、单项筛选和公开方法依据下提出**唯一一个**预登记候选。需明确安全与进度如何权衡：若使用门控进度、约束乘子或范围归一化，给出数学定义和局限。不能声称有限权重的加权和保证安全不退化，也不能规定每条危险快速轨迹都不得降低进度。参数在开发排名前锁定，不每组反复调权找分。

可选参考速度场锚点`||v_phi-v_ref||²`若使用，所有对应policy组与额外控制共同使用，区别old policy（每轮）与固定ref（本轮起点）。它不是精确KL。不要在同一轮同时增大clip、减少采样步、改actor初始化。

### 9.4 过程时间与信用分配的边界

仅在reward中使用`any(D/C/T)`是**终局分项反馈**，不是过程信用分配。即使记录了事件时间，只要更新没有利用时间，就不能声称使用了过程监督。

不得把某物理时刻事故的优势硬分配给最近一个trajectory token或相同编号的flow step。联合轨迹、跟踪器、前序速度与yaw会影响后续事件。

真正的过程方法须明确：事件时间怎样影响动作梯度/轨迹段，并通过局部扰动或可微执行对照验证。主路B负责该部分，不能用多写几个日志字段冒充算法贡献。

---

## 10. C线主路B：可微真值环境代理与速度场梯度

此分支先做工程/数学验证，不与C2/C3/C4同时全部长训。

### 10.1 三种信号严格分开

1. **canonical真值值**：原仿真器的分数/事件，无直接autograd。
2. **GT环境下的可微代理**：由真实地图、目标运动和经过验证的tensorized执行产生平滑代价`J_S(q)`；不是官方PDMS精确梯度。
3. **学习预测器**：从相机表示预测代价/事件或梯度，存在拟合与分布外误差。

先1，再验证2，最后才3。

### 10.2 最小可微验证

从真实10步生成轨迹出发，先只用道路签名距离/平滑余量进行验证，再扩展已有证据支持的对象/TTC相关代理。候选实现优先复用已有稳定几何与differentiable dynamics；没有可验证实现时不假装numpy `.requires_grad_()`能反传。

若将LQR＋自行车执行迁移到torch：对位置、yaw、速度、控制与原执行器逐步对照；运行精度和饱和规则相同。只在命令轨迹上计算距离、不包含tracker时，必须明确是command-space proxy，并验证其建议在真实执行中仍有效。

验证内容：

- 可微区域数值有限差分一致；离散切换点不要求数学上不存在的光滑梯度。
- 以固定小步沿负梯度移动，与同范数随机/正梯度比较真实canonical后果。
- 动力学可行性、进度、舒适和其他安全项不能被忽略。
- 梯度在当前GT混合bridge与实际采样轨迹上的分布差异。
- 不把GT未来视频配给替代动作；这里只用固定日志对象的几何后果。

### 10.3 两个已发表更新器的适配选择

**全链路参考**：对较小批量的完整生成轨迹通过`J_S(Φ_phi(ε,H))`反传，作为真实可微代理梯度参考。[R08]

**高效候选**：按DFQL原论文和实现，将终端梯度作用于各步速度场，明确stop-gradient、时间权重和尺度。[R07] 原JAX实现需做数值移植单测；不搬用Bellman critic、不clip所有驾驶坐标到[-1,1]。

DFQL是代理更新，不能要求它与完整BPTT梯度逐元素相同；应在可解小例上核实公式，在相同GT代理下比较改善方向、稳定性与成本。旧policy sampling和训练速度场的dropout/精度保持一致。

**禁止GT桥捷径**：`x_tau=(1-tau)ε+tau*y_GT`已包含目标；`x_tau+(1-tau)v`不是无条件生成结果，靠近tau=1时对v的导数又变小。必须主要检查实际生成轨迹，不以GT桥上低后果误差证明策略改进。[P04]

---

## 11. 后果预测：仅在真值反馈被证明有用后开展

选择阶段C已验证有用的最少后果。初期只预测必要的事件/严重性与时间段，不建立全部框、轨迹、dense风险和多种未知量的巨型head。

P-ORACLE：每次更新使用真实仿真反馈。
P-PREDICTED：同一更新算法，部分反馈由预测器提供，保留固定比例真值复核/刷新。
P-NO-POLICY：只有后果预测辅助loss，不对DiT施加policy项。

主要比较：`P-ORACLE`与`P-PREDICTED`反映替代误差；`P-PREDICTED`与`P-NO-POLICY`区分“会预测”和“改变动作”。预测器应按scene/log留出及按策略版本留出评估，不只在旧固定轨迹上训练/测试。

若使用其动作梯度，需额外验证方向有效性；低BCE/MSE并不保证梯度可用于优化。actor更新期间冻结predictor参数，但不能错误no_grad阻断到动作的梯度。保持真实反馈抽查，避免reward hacking与停止策略。

部署删除预测器，不用预测后果择优或后修正。若删掉后行为改变，说明并非本项目定义的内化版本。

---

## 12. 训练范围、进度与防止实验爆炸

### 12.1 两级训练身份

**机制/增量探索**：允许使用同一个已完成S0作为共同起点。控制也同样继续训练、同样重置optimizer、匹配可训练范围。不能把新增方法对比未继续训练的S0作为唯一证据。

**正式方法确认**：共同通用VLM和随机驾驶模块，或双方共同拥有且计入成本的基础训练阶段。历史输入、监督教师、状态输入和数据人口一致。旧S0在训练logs上已高度拟合，探索必须标注这一局限；真正泛化确认使用基础策略未见的合法训练池或重新划分后共享训练一个基础策略，不自动恢复旧MIN工作流。

### 12.2 有限默认日程

| 波次 | 任务 | 本轮默认规模与决定 |
|---|---|---|
| 0 | 文献/源码/数据审计、toy测试、cache规模 | 只做有限SMOKE/PROFILE，任何正式pilot前均需批准 |
| 1W | S-P0…P3空间筛选 | 共同S0起点，计划20k更新，5k阶段暂停；仅入围对继续至共同20k；不得把5k当从头充分训练 |
| 1C | 事件提取与C0…C4真值筛选 | 每组共同计划4096策略更新，512/1024诊断；C5仅根据预登记选择新增一组；计入每更新实际反馈候选数 |
| 2W | T0…T4历史/未来矩阵 | 先统一5k探索，按科学问题保留必要控制后共同20k；T0可在严格同配方时复用空间赢家 |
| 2C | 可微代理方向/有限FPO稳定性扩展 | 一条主路线稳定后，再给最多两种梯度更新同预算对比 |
| 3 | 单个专家替换、最小后果预测 | 条件性执行；不把所有教师乘全部时间矩阵 |
| 4 | 时空×后果联合 | 四格控制；最多关键两对第二训练seed |

20k/4096为**候选探索预算**，不是保证收敛。启动前根据profile、有效曝光和授权上限锁定。达到预算但尚不足以判断应报告证据不足，不单边无限延长。不因早期低分自动判定从头训练无效；最终长训使用共同预登记充分预算（参考原100k），不默认全候选都100k。

每个wave有候选清单、上限、升级条件；无事实根据的新分支不能偷开训练。预算不足则减少同时探索的可选分支，不删掉核心对照来制造结论。

---

## 13. 联合研究矩阵：历史输入和辅助监督不能混淆

选定一种输入协议后，最终四格为：

| | 无后果策略项 | 有已锁定后果策略项 |
|---|---|---|
| 相同输入、无新增时空辅助 | J00 | J01 |
| 相同输入、有新增时空辅助 | J10 | J11 |

四组历史帧/视图、W总数/历史module容量尽量一致，无辅助组保留相同部署参数/路径；关闭的是训练信号，而不是暗中缩网络。另用T矩阵报告新增历史观测的独立收益。

关键差值：J10−J00、J01−J00、J11−J10，以及差值之差。不把正的单项均值自动称协同。

共用初始权重和更新预算时，必须说明哪些阶段共享。若各自训练到了不同checkpoint再联训，这一历史差异不能被忽略。

结果至少回答：更好的时空表征是否让真实后果训练更容易、更稳定？后果训练是否把本来“可读出”的未来知识变成更多自然成功轨迹？

---

## 14. 数据、随机性、评价与结论边界

- NAVSIM主数据复用合法原始GT划分；实际N和log数核对。原开发1696/16logs、Navtest12146/136logs是历史合同，不硬编码替代manifest检查。
- 历史帧沿原log取决策时刻之前；未来教师按真实时间取决策时刻之后；相邻训练/验证样本不能因窗口读取跨log划分。
- 归一化/PCA/尺度/采样阈值只训练拟合。额外用于机制诊断的GT对象区域不进入当前输入或W数量选择。
- 环境/标签变化必须版本化。原日志视频仅监督日志对应未来，不给任意ego候选配同一视频称动作反事实真值。
- 状态、noise、patch采样、horizon、history缺失、rollout policy版本使用分离且可恢复的RNG。公共模块初始化不能被新head抽样扰动。
- 正式评价单候选10步、同scene绑定采样seed；多采样seed报告期望与波动，不选择最高分轨迹。训练候选数≠部署候选数。
- 报告PDMS/NC/DAC/TTC/EP/Comfort/DDC、ADE/FDE、原零分修复与新增、原高分退化、有效标签覆盖。
- 距离重建、事件判断、梯度方向和实际规划收益分开；不能只报总体IoU或MSE。
- 同一日志内相关场景用log-cluster统计；训练seed与推理seed分开；head复跑不是独立基础VLA复跑。
- 开发用于选择；Navtest方案冻结后有限统一评价。既往已反复观察，不能称新盲测。新监控脚本不重复唤起历史观察器。开发集故障和事件稀少不通过删除低分scene来掩盖；基于训练反馈进行困难分层时同时保留完整人口评价和采样权重。
- 正式精度必须与部署计时一致：不能FP32 PDMS配BF16时延。输入获取、历史首帧、滑窗复用分别计时。
- nuScenes保持接口可配置三秒六点/六相机；本轮不恢复暂停的nuScenes或Bench2Drive长训。NAVSIM反馈训练后若要扩展nuScenes，必须重新定义通用后果，不伪造nuScenes官方PDMS。

---

## 15. 多Agent分工：真实调用，不是角色扮演

### 15.1 组织结构

| Agent ID | 职责 | 文件所有权/输出 | 不允许自行做什么 |
|---|---|---|---|
| A0 Lead | 总体方法、接口、实验优先级、整合与唯一正式派发 | 计划、实验注册、审批汇总、集成分支 | 不代替独立代码/效率审阅签PASS |
| A1 Evidence | 文献全文与官方代码核验，检查新假设/对照 | `research/evidence/`、Design Cards | 不直接改共享模型或运行正式训练 |
| A2 SpaceTime | 高分辨率/稀疏读出、时间差分、专家adapter | `modules/st_world/`、相关tests/config | 不修改PDMS语义/评分器 |
| A3 HistoryData | 历史数据、相机/时间/位姿、缓存、PCA | `dataloader/st_history/`、目标cache工具 | 不默认缓存可训练Qwen hidden |
| A4 Consequence | canonical事件提取、单位、有效性、CPU仿真适配 | `modules/pdm_feedback/`、instrumentation/tests | 不偷偷改变官方分数与EP归一化 |
| A5 FlowPolicy | FPO/必要代理梯度实现、old/ref策略合同 | `modules/flow_posttraining/`、actor tests | 不复用修正器或将GT混合端点当真实rollout |
| A6 Correctness | 独立读代码、梯度、信息隔离、DDP/恢复审计 | 审阅报告、复现测试建议 | 不作为主要实现者后自审通过 |
| A7 Systems | 多进程、多GPU、I/O、teacher/rollout/eval流水线profile | `tools/.../profile*`、资源合同、效率审批 | 不以改科学目标或少算任务加速 |
| A8 Evaluation | 固定评价、统计、失败转移、阶段结果 | 指标schema、对照表、科学结论 | 不根据Navtest选择训练配方 |

这是9个角色，不要求9个同时运行。默认同时活跃4个子Agent；原生工具/许可支持时可由Lead调整到6。小任务留在Lead，避免重复搜索和消耗上下文。

### 15.2 原生执行方式

- 先核实当前Codex版本、实际子Agent工具、workspace和权限。按官方文档使用原生子Agent/thread/worktree能力。[R10]
- 明确要求实际创建独立Agent并记录任务/线程ID；不能在一个主线程中写“Agent1认为…”伪装已调用。
- 不硬编码本文件未验证的CLI参数或模型名。若原生能力不可用，记录`BLOCKED_MULTI_AGENT`并请求启用；可继续只读准备，但不能声称独立双审已完成。
- 每个实现Agent使用隔离worktree和明确写入范围。共享`DDPForesight.py`、训练入口、依赖文件由Lead整合，其他Agent返回patch；不能并行覆盖同一文件。
- 子Agent不得再无上限递归spawn。Lead统一分配CPU/GPU租约、端口、缓存目录和run_id。
- 不让多个Agent各自启动相同评测或teacher提取；共享缓存只有一个写者，其余只读。

### 15.3 推荐执行波次

**波次I并行**：A1文献与代码证据，A2现有模型映射，A3历史与缓存资产，A4原scorer事件映射。A0汇总出最小接口。

**波次II并行**：A2空间/时间实现，A3历史adapter，A4事件提取，A5FPO迁移。A7只对已可运行模块profile；A6审阅每个具体patch而不等待最后一刻。

**波次III**：A0整合冻结；A6在相同SHA做正确性审批，A7在该SHA做资源效率审批，A1签对应证据卡；A8确认实验可解释性。未通过则返回具体owner修复。

**波次IV**：A0唯一派发已批准训练。A7管队列/资源；A8分析已完成checkpoint；A1/A2/A5研究下一项有依据的分支，不能热改正在训练版本。

### 15.4 每次交接包

```yaml
task_id: 唯一ID
question: 一句话目标
base_sha: 读取/编辑起点
owned_paths: 允许编辑的路径
forbidden_changes: 输入/指标/历史任务等禁止变化
evidence_cards: 所依赖卡片
acceptance_tests: 必须通过的测试
resource_lease: CPU/GPU/内存/存储/时间上限
outputs: commit/patch、命令、结果位置
status: DONE | NEEDS_REVIEW | BLOCKED | FAILED
known_limits: 未验证内容
```

返回摘要与证据，不把大量未筛选日志塞入主线程。独立审阅者发现错误需写文件/行/触发条件与修复建议；不能只有“整体看起来合理”。

---

## 16. 正确性验收：按修改目标检查，不仅检查能否运行

### 16.1 通用检查

- 旧S0只读预测可复现；新模块关闭时旧行为保持，架构变化导致不能等价时明确建立新控制。
- 当前/历史输入白名单、未来target分离；替换GT未来/DINO/PDMS标签不改变固定参数的预测，只改变训练loss。
- 允许历史后，测试不再错误禁止历史；应禁止任何`t>0`输入。
- 改变历史内容只影响允许路径；camera/time/patch flatten顺序可逆，6/8轨迹点可参数化。
- 训练主目标梯度确实到达预定参数。世界loss到语言/W，policy loss到DiT；教师与old/ref policy无意外更新。
- 0项有效辅助标签rank仍参与一致collective；单卡与多卡全局loss/梯度尺度相同，不简单平均不等人口局部mean。
- FM repeat只增加ego训练噪声，不重复辅助loss；actor候选数不误当主batch乘数。
- checkpoint保存模型、optimizer/master、scheduler、CPU/CUDA与专用RNG、数据位置、rollout版本和累计曝光。
- 同环境连续/中断测试；A/A浮点非确定性与恢复边界位级一致分开，不为通过而放宽不相关错误。

### 16.2 W线专项

- DINO取真实空间patch，排除CLS/register；矩形网格按H/W而非sqrt推断。
- 当前teacher到未来teacher无隐式center crop；相同源ROI、RGB/mean/std正确。
- W数量与输出query数相互独立；采样位置真实存在于缓存；图像分辨率变化不偷偷改变Qwen输入像素。
- 历史相对ego变换、相机ID、时间编码和缺帧mask正确；时间反转对照不跨视角。
- PCA训练划分、均值和projection固定；稀疏采样与全网格loss在确定性小例的期望归约一致。
- 差分用相同view/坐标/时间目标；低残差不错误单位化；不可见位置的mask不把未来答案泄漏到输入。
- 删除训练期feature heads后，固定输入的ego输出不变；历史融合为执行组件不能删除。

### 16.3 C线专项

- 同一实际轨迹原scorer分值完全一致，instrumentation不改变碰撞历史列表或聚合顺序。
- D/NC/TTC原生trace→聚合值一致；多次事件与severity正确；无事件inf不进入loss。
- 固定候选的独立反馈对batch重排/无关候选加入保持声明的性质；否则明确canonical参考集合。
- FPO toy任务正/负advantage更新方向、old=current ratio、loss维度/时间权重/随机coupling正确。
- 对detach生成动作计算CFM政策loss可有DiT梯度；把真实得分加常数不应产生伪梯度。
- 可微代理分别验证数值微分和真实执行方向；BF16/clip等不能制造错误控制梯度。
- 部署删除真值/预测后果模块，仍直接从初始噪声执行同一10步策略，不调用修正/选择。

### 16.4 必要反例

每项mask/时序/评分测试至少有失败反例：缺未来、道路出界后回归、先静态对象0.5后动态0分、不同view排序、跨log历史、全零优势、全无效辅助rank、非法heading、长序列OOM、scorer子进程异常。不能只有一个正常forward截图。

---

## 17. 多进程多GPU效率验收：资源充分利用不等于机械占满

### 17.1 先确认真实资源

读取当前授权、现有进程、CPU quota、NUMA、GPU型号/显存/互联、存储位置/容量。只在获授权空闲槽位运行；不停止pressure或其他实验进程来清场。共享GPU的进程时间不称独占GPU-hours。

### 17.2 三种工作负载分开优化

**教师/数据**：多GPU按图像/clip ID分片编码；CPU worker解码/预处理；一次GPU批量提取后分shard原子写入。单写者manifest，resume仅跳过已完成且hash正确的分片。线程×进程×BLAS的总并发上限合理，防CPU过订阅。

**VLA训练**：优先每节点8卡单任务，与同节点2×4卡并行比较。保持effective global batch；同时实测2×4卡的aggregate throughput，不能把单任务速度乘2。数据按log/scene正确分片，DDP一进程一卡；不混DataParallel。

**反馈/评价**：GPU批量生成或推理→CPU多进程仿真→结果归约。队列容量与背压固定，GPU不每条候选阻塞等待CPU。CPU-only评分不持有GPU context。on-policy老版本候选不无限排队；policy version更换边界检查未消费数据。每worker独立scorer，候选/场景打包顺序不改变结果。

### 17.3 实测要求

每种被批准拓扑至少：20个warmup更新＋100个稳定更新，关键候选重复2个窗口；端到端计时以最慢rank和窗口同步完成为准。分别记录：

- 场景/秒、候选/秒、反馈/秒；实际optimizer updates与样本曝光；
- Qwen视觉/语言、DiT采样/FM、辅助head、反向、通信、CPU仿真、I/O等待；
- 参数/optimizer分配后峰值显存，CPU峰值内存；
- 时延P50/P95、GPU-process与allocation成本、CPU child-inclusive成本；
- 预处理、teacher cache、编译和保存的摊销。

训练rank少是否更慢、更多worker是否堵IO都以测量判断。对小头单卡胜于DDP完全可以通过审查；必须解释选择依据，不为“用了多GPU”做更慢实验。

### 17.4 优化优先级

1. 可冻结且身份稳定的视觉输出缓存、按scene一次VLM供多候选共享。
2. pinned memory、异步预取、批量传输；减少逐tensor `.item()/float()` 和无用同步。
3. 正确使用梯度累积与DDP `no_sync`（含forward范围）；按实际DeepSpeed版本采用等价机制。[R09]
4. 合并时间/view维度的等价计算，复用不依赖候选的条件投影；不跨optimizer更新缓存可训练结果。
5. 合理activation checkpoint和混合精度；区分会改变BN统计、随机流或梯度精度的方案，单独验收。
6. 小范围compile/更换kernel只有真实净收益时启用；保留eager与原精度回退，计入编译成本。

禁止以减少10步采样、未来时刻、patch监督数、候选预算、更新次数、数据人口或冻结研究对象冒充纯工程加速。确需更改应新建科学实验并重新批准。

### 17.5 评价流水线

使用固定manifest分片，独立输出目录，最终合并校验每scene恰好出现一次。失败记录不静默跳过，重复预测按run identity判定。异步评价对不可变checkpoint运行，不读取训练中未写完权重。正式训练可与不争用其资源的CPU评价并行，不能导致比较组性能差异来自资源争抢。

---

## 18. 工程落点与交付文件

优先复用已有模块和trainer，只新增目标相关组件。以下为建议路径，实施前由Lead核实仓库惯例，不生成调用不存在文件的命令。

```text
starVLA/model/modules/st_world/
  history_adapter.py
  sparse_feature_readout.py
  temporal_losses.py
  teacher_adapters.py
starVLA/model/modules/flow_posttraining/
  fpo_objective.py
  policy_snapshot.py
  consequence_contract.py
  differentiable_surrogate.py        # 条件性
starVLA/dataloader/st_history/
  sequence_index.py
  teacher_cache.py
  batch_contract.py
tools/st_world_policy/
  research_evidence.py
  validate_inputs.py
  cache_targets.py
  instrument_pdm.py
  profile_pipeline.py
  check_formal_approval.py
  run_registered_experiment.py
  evaluate.py
  summarize.py
configs/st_world_policy/
  spatial/ temporal/ consequence/ joint/
reports/st_world_policy/<date>/
  evidence/ reviews/ profiles/ registrations/ results/
```

不要创建与研究无关的大型MLOps平台。审批检查器只需可靠核对指纹与状态；队列、恢复、日志优先复用已验证工具。

每阶段提交：Design Evidence Cards、参考/迁移测试、真实profile、独立审阅意见、run配置与源码、数据/教师身份、执行命令、完整结果/失败、成本和下一阶段决定。

正式结果包含 `NOT_RUN`、`BLOCKED`、`FAILED` 与负结果；不要用smoke填正式表。文件中不上传原始图像、点云、模型权重、缓存或密钥。任务分支可push并核对SHA，不自动合并main。

---

## 19. 本轮完成标准与下一阶段决策

### A. 时空侧必须得到的答案

- 提高teacher分辨率、保留细网格、增加W，这三种改变分别带来什么？
- 固定原输入时，细网格监督是否改善小目标/边界而非只改变PCA可视化？
- 历史本身与未来辅助各贡献多少？
- 时间差分是否改善时间关系与实际规划，是否损害静态内容？
- 视频/几何专家是否有相对DINO的真实增量，代价多少？

### B. 后果侧必须得到的答案

- 哪几种canonical后果在真值条件下就能改善直接DiT生成？
- 只有终局分项是否足够，何种明确机制真正利用事件时间？
- 改善是否以停车、进度或其他安全退化交换？
- GT可微代理的梯度是否改善真实执行，而非只改善近似函数？
- 预测器是否仅在真值有效后建立，是否保留了关键更新方向？

### C. 联合侧必须得到的答案

- 额外时空理解是否帮助策略更有效利用后果反馈？
- 后果训练是否使时空辅助从“可预测”转化为“第一次生成更好”？
- 是否在同历史输入、同训练来源、同部署预算下成立？

**允许的结论**：有依据的正面结果、明确范围内的负面结果、受样本/预算限制尚不能区分。  
**禁止的结论**：仅辅助MSE下降即成功；仅1个高分seed即稳定；只有实现通过即效果成立；动机好所以无条件继续；文献有先例所以方向不值得做。

---

## 20. 一手参考与项目证据索引

以下链接供执行Agent打开全文和定位官方代码；访问版本与本地精确SHA须另行锁定。

### 项目证据

- **[P01] C到G方法历程**：https://github.com/IDayday/VLA-Drive/blob/0c03b32f17bfe0d9dd82ff95f141739258e6c33c/reports/experiment_results/20261014/METHOD_HISTORY_ZH.md
- **[P02] MIN正式报告与资源边界**：https://github.com/IDayday/VLA-Drive/blob/7ffd77d9d0d6acbe6f7d39852e57eccd02160ae9/reports/minimal_repair/20261015/FINAL_REPORT_ZH.md
- **[P03] 固定PDMS scorer**：https://github.com/IDayday/VLA-Drive/blob/7ffd77d9d0d6acbe6f7d39852e57eccd02160ae9/navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py
- **[P04] 原动作头**：https://github.com/IDayday/VLA-Drive/blob/1493deda247107efe076b9294863a6753391298a/starVLA/model/modules/action_model/GR00T_ActionHeader.py
- **[P05] 原执行器**：https://github.com/IDayday/VLA-Drive/blob/7ffd77d9d0d6acbe6f7d39852e57eccd02160ae9/navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py

### 文献与代码

- **[R01] DINO-Foresight: Looking into the Future with DINO**：https://arxiv.org/abs/2412.11673 ；官方代码：https://github.com/Sta8is/DINO-Foresight
- **[R02] DINOv3**：https://arxiv.org/abs/2508.10104 ；官方代码：https://github.com/facebookresearch/dinov3
- **[R03] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving**：https://arxiv.org/abs/2602.10884 ；官方代码入口：https://github.com/mengtan00/ResWorld
- **[R04] V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning**：https://arxiv.org/abs/2603.14482 ；官方代码入口：https://github.com/facebookresearch/vjepa2
- **[R05] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos**：https://arxiv.org/abs/2501.12375 ；官方代码入口：https://github.com/DepthAnything/Video-Depth-Anything
- **[R06] Flow Matching Policy Gradients**：https://arxiv.org/abs/2507.21053 ；作者说明：https://flowreinforce.github.io/ ；官方代码：https://github.com/akanazawa/fpo
- **[R07] Direct Flow Q-Learning**：https://proceedings.mlr.press/v306/cao26f.html ；官方代码：https://github.com/2012060/Direct-Flow-Q-Learning
- **[R08] Reparameterization Flow Policy Optimization**：https://arxiv.org/abs/2602.03501 ；代码入口需从作者一手页面重新核实，不猜仓库名。
- **[R09] PyTorch DDP**：https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html ；性能指南：https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html ；执行前必须切换/核对项目实际版本。
- **[R10] Codex多Agent官方入口**：https://developers.openai.com/codex/multi-agent/ （当前重定向至 https://learn.chatgpt.com/docs/agent-configuration/subagents）；工具与配置以实际安装版本为准。

---

## 21. 最后执行约定

1. 本文件中的候选数值是待审阅的起点，不是已经证实的最优配方；新方法可按证据调整，但要先登记，不事后改变主比较。
2. 每次正式训练前，必须同时取得独立正确性审阅与实际资源效率审阅；文献/源码依据同样不可缺少。
3. 不要求每个新想法先达到某个任意PDMS门槛才研究；要求其定义清楚、实现正确、比较公平、预算有限。
4. 不把“审核”扩成无限工程项目。具体阻塞具体修复，无关历史模块不重审；通过后按登记计划实际训练和分析。
5. 最终目标是**世界知识内化到直接动作生成**。后修正器只作为旧证据，不再被默认恢复。
