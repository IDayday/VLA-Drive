# V3 训练前纠错结果（2026-09-27）

本轮纠错完成，等待代码审阅；真实数据 optimizer updates **0**。保留单一 `JointSceneFlow`，执行 ego 直接来自同一联合样本 slot0。没有启动 ALL/MASK 长实验、完整视觉训练或 Navtest；没有恢复 V2 step1887。算法整体仍为 `engineering_status=PARTIAL`（视觉落地未完成），`research_status=NOT_RUN`；本轮只验证正确性修复。

- 审阅起点：`b4f08d05c0202fb483bc3247cab97abe59694d16`。
- 分支：`fix/joint-local-scene-v3-pretrain-review-20260927`。
- 主体修复及 GPU/恢复验证代码：`4de523bdefb99e7490d358754e5c464ecef70686`。
- 最终代码：`4893159c25dc7dfad6f718f8173dd46cb9533295`；随后提交只保存审阅证据和状态。
- 隔离工作区：`/mnt/project/VLA-Drive-v3-pretrain-review-20260927`。原 V3 工作区仍在 b4f08d0 且干净，旧报告、缓存和权重未覆盖。

## 原问题、最小反例与修复

先在 b4f08d0 运行原逻辑反例，见 [BEFORE.json](BEFORE.json)，没有 optimizer update。

| 问题 / 修改位置 | 修改前证据 | 修改后验证 |
|---|---|---|
| 未建模 yaw 参与采样；`joint_scene/flow.py`, `contracts.py`, `data.py` | 只改周车 yaw 噪声，xy 最大变化8.0774m；只改其速度，xy 变化3.5208m | 编码前、初始化、速度、每次积分及解码统一规范化；CPU/CUDA 输出逐元素不变，ego yaw 梯度非零 |
| context padding NaN；`flow.py` 的 `SafeGraphBlock` | forward 与参数梯度均非有限 | actor/context/scene 各自先按 mask 清理再运算/投影/embedding；CPU/CUDA 有效输出和参数梯度与规范 padding 相同且有限 |
| 有效输入契约；`contracts.py` | active class19 被接受 | 有效 NaN、非法类别/尺寸、ego/edge、自环缺失及 mask dtype 明确拒绝；无效类别先替换合法占位 |
| 距离主导且无类型/观测资格；`graphs.py` | 零支持对象仍占两个 active slot，近处 cone 占唯一角色 slot | 当前导航走廊/二跳候选；静态物及零支持风险转 context；导航、稍远相关车、停车车、去重、溢出及 ego-only/单邻居反例通过 |
| batch1 永远抽 ego；`masks.py` | 连续20次全是 ego | 跨调用调度与独立 RNG；batch1、奇数/尾 batch、回退、部分轨迹及序列恢复通过 |
| conditional 目标偏置；`runtime.py` | ALL 周车6点对 conditional 周车3点 | 模型评估前枚举所有合格 ego/邻居查询，同目标、同点、同噪声；失败保留，分母固定 |
| `role_loss_weight` 未读；`train_mechanism.py` | 源码反例确认 | J_ALL 权重 `[1/2,1/2]`；J_MASK `[1/(1+λ),λ/(1+λ)]`，实际反传并记录；恢复 smoke 使用 λ=2 |
| 缓存/恢复/预算；`data.py`, `train_mechanism.py`, `budget.py` | 未声称旧版所有恢复反例都已复现失败 | schema4、图规则/字节/时间跨度校验、run ID、预算锁、完整状态/RNG；CPU/CUDA 真训练器4 vs2+2一致 |

`SafeGraphBlock` 是本分支的局部安全封装，未修改旧 V2 使用的 GraphBlock。没有无条件清理有效输入以掩盖错误，没有第二个生成执行 ego 的 DiT。

## 四类 mask 与当前图

`modeled_state_mask` 是可部署的任务定义：ego 建模 xy 与 yaw sin/cos，周车只建模 xy。未建模通道固定为零哨兵，不是有效 yaw，不进 yaw 指标或预测可视化。`feature_valid` 只表示标签坐标是否存在，不改变部署积分通道；部分周车 xy 标签缺失不会冻结相应部署时刻。`known_mask` 是补全条件中的已知坐标，必须属于 active 且 modeled 通道；`active_actor_mask` 表示当前联合任务中的主体。

类别直接采用原始日志标签生成器 `structured_world/targets.py::CLASSES`：0 vehicle、1 pedestrian、2 bicycle 可作为轨迹主体；3 cone、4 barrier、5 construction-zone sign、6 generic object 只进当前 context。静止车不因速度被排除，周车速度未知，不填零推断安全。

默认 `decision_local` 使用当前 ego 状态/导航的保守走廊，按走廊关联、距离和必要二跳邻居选择；`nearest` 保留为距离基线。节点上限相同，K 不强制填满。几何支持、类别、距离、走廊净空、来源索引、排除/选中/去重理由及 context 去向保存在图元数据。track ID 仅用于标签对应和审计，不入网络。

正几何支持的交通参与者才成为轨迹候选；零支持或容量外对象可进入最多64个 context，仍受半径/容量限制。context 是当前对象记忆，**可与轨迹节点重叠**，不能把 context 与 selected 数相加当作去重人口。优先保留未选中但走廊相关的风险，溢出明确计数。几何支持只是投影/FOV 资格，没有声称解决真实遮挡。

构图不读取未来值/未来有效性/GT ego 未来，不使用 GT 地图、周车日志速度或历史图像。辅助任务在固定图之后使用标签；邻居资格只需至少一个 xy 两坐标均有效的点，不要求完整未来，不恢复旧2m/类别正确门槛。

## 实际验证与消耗

- 最终代码 **53 CPU tests passed，0 failed**，3.40s；一个环境 `pynvml` 弃用提示。见 [CPU_TESTS_FINAL.log](CPU_TESTS_FINAL.log)。单元测试无 optimizer update。
- 本机 A800 GPU0 用于合成恢复，GPU1 用于前反向和真实无更新检查。未使用 vla-zt2，未停止其他任务；结束时检查用卡无残留本任务进程。
- CPU、CUDA 各自连续4步与2+2步：模型、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、noise/time RNG、角色状态、epoch/offset、曝光、监督坐标和学习日志一致。见 [RESUME_CPU.json](RESUME_CPU.json)、[RESUME_CUDA.json](RESUME_CUDA.json)。不声称跨设备/非确定算子逐位一致。
- 不确认 STOP 的 resume 在更新前拒绝，checkpoint 字节不变；显式 `--acknowledge-stop` 归档标记后才恢复。
- 合计 **16/20 合成更新，0/0 真实更新**；GPU **0.008588/2 GPUh**，计费区间含模型加载、测试、评估、可视化和保存。GPU 初始化前 Python 导入及 CPU-only 测试不占 GPUh。5个账本 run 均 complete，非预期运行失败0；STOP 拒绝是 CUDA 分配前的预期负例。
- GPU 检查覆盖 ego loss 到交互 attention 的梯度、ego yaw 梯度、已知坐标逐步钳制、隐藏标签毒化、slot0 执行一致性。随机 condition tensor 梯度只是接口检查，**不是 VLM 验证**。见 [GPU_CHECKS.json](GPU_CHECKS.json)。
- 4个真实训练场景、21个固定查询，失败0；全尺寸随机模型仅用于数值检查，不能解释为预测/规划成绩。生成4张局部图并检查可视化，没有绘制预测周车 yaw。私有 CSV、图、checkpoint 仅留本机，见 [PRIVATE_ARTIFACT_INDEX.json](PRIVATE_ARTIFACT_INDEX.json)。
- 每个4步 synthetic run 共6次场景曝光：名义 ego/neighbor 各3，实际 ego4/neighbor2，回退1；role 有效 ego 坐标48、neighbor xy12、neighbor yaw0。完整分类坐标计数见 [AFTER.json](AFTER.json)。mask/noise/time RNG 独立。

GPU 和恢复证据绑定4de523b。之后4893159仅增加逐场景图规则/时间跨度校验、空输出参数拒绝、独立 all-hidden target CSV 和覆盖率字段；这些增量通过最终53项 CPU 测试，另在 small/train/holdout 各8条真实记录上验证加载，0更新。**没有把此前 GPU/恢复记录冒称最终提交重新跑过**，最终提交未再次执行 optimizer 更新。

## 新数据及评估

先验证8个真实场景的 source→track→future 对应，再按固定规则重建 schema4。新目录 `/mnt/project/v3-pretrain-review-artifacts/20260927/annotated_{small,train,holdout}_v4`，旧 `annotated_train_v1` 只读；没有按 holdout/PDMS 反调阈值。

| 计数 | train | holdout |
|---|---:|---:|
| 场景 / 构建失败 | 7284 / 0 | 64 / 0 |
| 当前源对象（已过滤） | 150039 | 1609 |
| 轨迹候选 | 54020 | 564 |
| selected 周车 | 44327 | 424 |
| selected 中有未来标签 | 43849 | 422 |
| context / 溢出 | 143163 / 2980 | 1489 / 45 |
| ego-only | 2315 | 21 |
| 有邻居补全任务的场景 | 4961 | 43 |

train978 logs、holdout59 logs，token/log 交集均0。类别/距离/支持分组及源身份见 [DATA_TRAIN_V4.json](DATA_TRAIN_V4.json)、[DATA_HOLDOUT_V4.json](DATA_HOLDOUT_V4.json)、[DATA_IDENTITIES.json](DATA_IDENTITIES.json)。都是训练域机制数据，不是 Navtest。

源缓存已做 ROI/FOV 筛选，未补入原始日志完整当前人口。字段改成 `source_current_objects`，`raw_log_population_available=false`；此前被删除的图外/低支持风险无法从缓存恢复。`current_unobserved_selected=0` **不能**证明可观测性筛选解决了问题。

评估先落盘 `queries.json`，固定枚举每场景所有具有完整 xy 点的 active ego/邻居。输出 `all_hidden_scenes.csv`、`all_hidden_targets.csv`、`conditional_queries.csv`；conditional 的其他主体 GT 未来属于机制诊断。指标包括同目标 ADE/FDE、有效 ego yaw、完整/局部目标与点覆盖、静态/动态、stationary；合法当前速度仅 ego 可用，故不算邻居 CV。失败逐行保留并使汇总不可用，不筛掉失败后报均值。ego/邻居来自同一联合样本，不拼接单车最优样本。all-hidden 仍是结构化 GT 当前输入，不能称相机部署成绩。

## 未完成的视觉链路

`build_inference_graph` 接受预测框只是接口。当前 bbox assignment 与同 track motion 的完整预测 query 监督、Reader/head/Qwen 真实联合梯度、annotated→predicted graph 过渡、完整当前相机 `predict_action` 封装仍待实现。没有新增冻结桥接替代这些路径。

机制训练器冻结未使用的视觉 `condition` 投影。配置记录本机 Qwen `text_config.hidden_size=2048` 来源；换骨干必须核对实际维度。完整视觉训练、长机制实验及 Navtest 全部 NOT_RUN，本轮无规划收益结论。

## 复现入口

本工作区已实测、零 optimizer update 的最短命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m pytest -q tests/joint_scene
```

环境 Python3.10 / PyTorch2.5.1，GPU NVIDIA A800-SXM4-80GB；没有改依赖环境。旧反例必须导入原代码：

```bash
PYTHONPATH=/mnt/project/VLA-Drive-joint-local-scene-v3-20260927 \
/root/miniconda3/envs/ddp/bin/python \
/mnt/project/VLA-Drive-v3-pretrain-review-20260927/tools/joint_local_scene_v3/reproduce_review.py \
--output /tmp/v3_original_review_new_result.json
```

实际恢复命令完整保存在 RESUME_CPU/CUDA.json，绑定4de523b和同数据/配置/设备。入口 `python -m tools.joint_local_scene_v3.check_resume --config configs/joint_local_scene_v3/mechanism.json --device cpu|cuda --ledger LEDGER --output NEW_OUTPUT`，每设备8个合成更新。**剩余4步不足重跑整套恢复比较，不再执行，也不新建账本绕过限制。**

GPU 入口 `python -m tools.joint_local_scene_v3.check_gpu_review --config configs/joint_local_scene_v3/mechanism.json --real-data DATA --ledger LEDGER --output NEW_OUTPUT`，0更新。训练器必须提供独立 `--run-id`、`--ledger`；本轮 real cap=0，在训练前拒绝真实更新。预算按安全边界停止，允许正在执行的 forward/保存的时间粒度，不宣称实时硬抢占 GPU。

数据重建入口 `python -m tools.joint_local_scene_v3.data --index INDEX --targets TARGETS --meta-root META --observations OBS --config CONFIG --output NEW_DATA --workers N`，实测完整参数在 DATA_IDENTITIES.json 各 split 的 `source` 字段。必须新目录，不覆盖现有当前/未来文件。纠错提交后结束本轮，下一轮先审阅通过。
