# 唯一 rollout-scale 正式对照：预登记科学设计

状态：**`PREREGISTERED_SCIENTIFIC_DESIGN_NOT_APPROVED`**。若已登记的真实 SMOKE 4 updates、PROFILE 240 updates 及独立正确性/系统审阅通过，科学上支持新增**一个** `scene_center_rollout_std_v1` 的 4096-update 组，与已完成的历史 C1 4096 直接比较。本文件只锁定问题、共同身份和读数；不派发资源，不签实现/系统 PASS，不授予正式训练或评测批准。当前实现参考为 `751fcfa2b41549ae16f7b3204aef8dd18b912016`；最终批准须指向经过实际验证的整合 SHA、配置和资源身份。

## 1. 唯一问题与两臂

问题：在原 C1 canonical PDMS、原 BC 锚点和同预算下，对完整 optimizer rollout 的 scene-centered advantage 作标准差缩放，是否改变最终单轨迹规划表现，且其安全/进度交换是什么？

| 臂 | 定义 | 新增训练 |
|---|---|---|
| L：历史 C1 | `C1_fulltrain_seed42_official_4096_v1`，`within_scene_center_only`，source `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345` | 0；复用真实完成的 4096 端点 |
| N：唯一新组 | `scene_center_rollout_std_v1`；建议 run label `C1_rollout_scale_seed42_official_4096_v1`，实际注册由 Lead 冻结 | 4096 optimizer updates，seed 42 |

唯一干预沿用封存提案 `b1a8cf49`：先保持原 FP32 的 `a[s,g]=r[s,g]−mean_g(r[s,g])`，再除同一 old-policy boundary 全 rank、所有有效完整 G4 候选的 FP64 sample std 加 `1e−10`，correction=1，分子不全局再 center，输出 detach 后回 FP32。零组计入 population 并保持零；不作 per-scene std。BC/policy 权重均为 1。固定 epsilon 与 correction 来自已核实官方 FPO source `418c2554` 的 `gridworld/models/fpo.py::FPO.learn` 和其 torch 2.2.2 std 定义；该导航适配不等同于官方约 2048 transition 的 GAE 配方。

独立 `rollout_scale_runtime_correctness_v1/LEGACY_CONTROL_BINDING.json` 已判定旧 C1 可在相同 Base/data/seed/recipe/evaluation 条件下复用；v2 将 timing-only 变更绑定到 `751fcfa2`。这是复用依据，不把两个 source hash 说成相同，也不把 CPU/Gloo 通过当作实际 CUDA/NCCL/ZeRO2 已通过。无需仅因新增显式分支而重训 legacy。既有 C0–C4 只保留历史上下文，不增加比较臂或从其 DEV 排名挑方法。

## 2. 共同训练身份与终点

| 项 | 两臂固定定义 |
|---|---|
| 起点 | 原 S0 `formal_S0_seed42_full100k_v2/milestone_100000` 的 learned FP32 masters 严格加载；binding `3b910a6d4245580d77f868e4e0af5faf2fc7fccc25c561c935f219ab1bf7f099`；共同新 optimizer，不从 C1、SMOKE 或 PROFILE 权重起步 |
| 数据 | 原 101,592 TRAIN scenes / 1,176 logs，完整 manifest SHA `b810cafbdaa42afd8554c08be443779e7d66edce56292c0c30940e728c0ed656`；同顺序、sampler、seed 42、当前三相机与原 GT；不筛掉零优势/低分/失败 scene |
| 模型 | 冻结原 Qwen，原 DiT/既有 action 投影可训练范围，保持原输入和部署路径；无 feature/history/teacher 新工作、无 refiner/scorer head |
| batch | global scene batch 32、world 4、micro 4、通常每 rank 8 scenes/两次微批；保留一个 24-scene 尾 batch，按真实 N 归约 |
| 反馈 | 每 scene 四条实际 Euler-10 完整轨迹，原 canonical C1-PDMS、同 score protocol `db383e17…b798143`；old 每更新刷新、maximum_reuse=1；MC50、mc_chunk5、原 coordinate mean/clip 定义 |
| BC | 原 GT、Beta(1.5,1.0)、repeat8、原 train-mode dropout/时间变换；BC=1、policy=1，不增减 BC 曝光 |
| 优化 | 原 AdamW，lr `1e−5`、betas `(0.9,0.95)`、eps `1e−8`、weight_decay `0.001`、grad_clip 1；原 4096 轴 cosine/min-lr `5e−7`、warmup100；不把 profile 长度作为 LR horizon |
| 精度 | `S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off`；原 deterministic/loader/checkpoint 设置。normalizer 的 FP64 统计是唯一新增数值适配 |
| 终点 | 精确 completed=4096，原 save_every512；512/1024 只作既有训练诊断，不据其分数选择 checkpoint、延长或提前判定效果 |

完整执行的匹配曝光应为 131,064 scene 呈现、524,256 feedback candidates、1,048,512 BC-repeat 呈现、每个 old/current 各 26,212,800 MC 点；4095 个 N=128 与一个 N=96。101,592 唯一 scene 中 29,472 重复一次，不能把曝光数当唯一场景数。正式完成后核对原 sampler/LR 顺序与这些计数，错误和中断不靠删除样本“补齐”。

各自参数更新后，候选、reward 和后续训练轨迹自然不同；N 必须新生成自己的 on-policy 反馈，不能消费 L 的历史账本充当当前策略数据。共同 seed 是随机性控制，不是承诺分化后轨迹或梯度逐 bit 相同。初始化、训练环境/源码兼容、native 恢复与最终身份由 Lead 和独立审阅绑定，资源路由另定，不改变上述科学定义。

## 3. 唯一主要差值与固定评价

主比较只有 **N@4096 − L@4096**。固定原 DEV 1696 scenes / 16 logs，manifest SHA `4ce40893281f3cf246989cbbac6bbd80c40c06ec937a4405a5a7c353c6dd0573`，完整人口逐 scene ID/log 配对。每 scene 一条原 Euler-10 轨迹，原 `foresight-action-v1` scene-noise 协议、推理 seed42；原 current-only 三相机、8×4 编码/8×3 物理轨迹、4 秒时域。推理采用原 **FP32 master eval / TF32 off**，不能拿 BF16 训练路径直接替代。无 best-of、多候选择优、外部 Scorer 或部署后修正。

L 的既有正式 DEV 单候选结果可复用：score identity `82992f77…767742`，`SCENES.csv` SHA `7c611333a2b4d3b53ac363008843bf78190a3b80c501e34afcbaa579fdcd468d`，native receipt SHA `96fff4a6…cf22a`。新 reader 可识别显式 normalizer 身份，但部署模型、种子、精度与 scorer/reference 语义必须与 L 一致并经独立核验；不得重标旧结果的 source。新的评价注册须绑定 N 的真实完整 native receipt/hash 和原固定 DEV/metric index，沿用原单候选官方 reference 协议。已有合法 L 结果不重复采样，不选择更有利的旧推理 seed。

主要估计量为 `ΔPDMS_pp = 100/1696 × Σ_i(PDMS_N,i−PDMS_L,i)`，分项分数也以百分点报告，ADE/FDE 以米报告。沿用现有 `evaluate_campaign.py::paired_statistics/estimate`：scene-weighted mean、同 log 整簇配对重采样，2000 次、NumPy PCG64、seed20261017、linear percentile 2.5/97.5 区间；每次抽 16 个 log 并保留其所有配对 scene。同步列逐 log 差和 equal-log sensitivity。只读完整配对结果后执行一次预登记分析，不按 outcome 改统计方法。

所有 1696 requested scene 均保留：合法 0 分是有效结果；invalid/unresolved 单列，缺失不填 0、不删除后宣称全人口完成。无完整配对时主结果标 INCOMPLETE；空事件/空分层给 denominator=0、estimate/interval=null，保留 bootstrap undefined 次数。只有一个训练 seed 和 16 个 log，区间只描述此条件下的日志抽样不确定性；不是跨训练 seed 复现、family-wise 结论或新盲测。旧 S0 已见 TRAIN，DEV 已在此前研究使用，本轮只能称同起点有限增量探索。

## 4. 安全与进度必须双向展示

继承原统计定义，不新增 outcome 驱动阈值：zero epsilon `1e−8`、high score `0.9`、large high-score drop `0.05`、EP drop `0.05`。完整报告 PDMS/NC/DAC/TTC/EP/Comfort/DDC、ADE/FDE、执行原始进度和平均速度，不把总分正差替代安全结果。

- 零分四格：L零→N非零、L非零→N零、两者零、两者非零；同时给总人口率、各起始组条件率、净修复及明确分母。
- NC/DAC/TTC 分别列改善/恶化/相同及实际 canonical 值转移：`N−L>1e−8` 为改善、`L−N>1e−8` 为恶化，其余为相同，保留 NC 的 0/0.5/1 语义；报告涉及 scene 和 log，不能只给“修复数”。DDC/Comfort/EP 同时保留分项差值。
- 以 L 定义的 high≥0.9、zero、中间组三个原分层，报告 high 的任何下降、大于0.05下降、降到零；与完整安全双向转移并列。保留原 `safety_gain_with_ep_drop`；相反交换固定为 `EP_N−EP_L>0.05` 且任一 NC/DAC/TTC 恶化，不另找有利阈值。
- 沿用停车代理：共同官方 reference progress≥5m，执行 progress≤1m 且平均执行 speed≤0.5m/s。列两臂数量、moving-reference 分母、新增/解除及净变化。未触发只说明该代理未触发，不能证明不存在所有停车捷径。

这些是科学读数，不是新增自动 PASS/FAIL quality 门槛。若总体 PDMS 正差但 NC/DAC/TTC 恶化，应直接报告交换，不称“安全提升”；若区间跨零、稀有事件支持不足或只一个 log 驱动，报告未建立对应收益/证据不足，不能以“不显著”证明等效、安全不劣或延长训练。即使区间全正，也不能跳过上述风险读数或自动扩大部署/实验。

## 5. 已知风险与训练解释范围

选择本候选的证据来自完整 TRAIN，不来自 DEV 排名：4096 个 update 均非零，原 80,933 非零组对齐，四固定窗口都有信号；数值异常 flags 全零，最大舍入能量上界比 `5.067594e−8`。但 2406 个 safe-EP-only update 的 scale 中位数1072.29，而1119个 D/C/T 可区分 update 为13.69，两组中位数比78.33，确认跨 update 重新加权风险。它不是配对梯度放大比，不能设为预期收益，也不能把归一化称无损修复。

原非零组98.34%为EP-only，却不代表梯度98.34%来自EP：带 NC×DAC 门控的优势值协方差份额为EP35.80%/TTC48.43%/Comfort15.78%，它们也不是参数梯度份额。固定3584的 policy/BC=0.0613169%仅是该 EP-only batch 的梯度证据，不选择1600倍或其近似系数。全常数组零信号仍保持零；normalizer不创造稀有事件。

训练中保留既定四窗口 `[0,1024)`/`[1024,2048)`/`[2048,3072)`/`[3072,4096)` 的原优势、std/scale 固定分位0/.25/.5/.75/.9/.99/1、零/非零组、D/C/T事件与可区分支持、safe-EP-only和安全update分层、数值异常、loss与gradient clipping及成本。用已存在的 receipts/计数，额外重型诊断须另获有限授权，不把本报告当新增逐步分解梯度任务。两臂各自 on-policy 分层是处理后的描述性统计，不能把其成员当固定匹配子集估计因果效果；安全 update 的全部 batch 成员也不能当作安全事件数。

同 batch 的正缩放保留相对 gap，跨 update 改变相对 BC 权重、Adam 状态和 clipping，因此不保证安全绝对代价被保留。old/current 在唯一更新前同起点、maximum_reuse=1，ratio≈1且clip0.2通常尚未激活；更新前clip统计不能担保optimizer step后的偏移。保留原clip、reuse和epochs，不追加第二方法来弥补此限制。

## 6. 条件、停止与交付

先由独立审阅确认真实 SMOKE4/PROFILE240 的完整链、normalizer身份、native恢复、NCCL/ZeRO2、非有限/失败处理和实测资源，并由 Lead 冻结正式注册、配置、源码、数据与新资源租约。SMOKE/PROFILE 权重弃用，成功不等于正式批准；本作者不替代两份审阅。当前真实 SMOKE/PROFILE 结果未被本报告读取，也未据其结果制定门槛。

达到4096即结束；identity/数据/数值/资源故障沿原合同停止并保留native状态，不能自动变更epsilon、cap、系数、batch/G/MC、精度、seed、reward、epochs或预算再试。不新增第二训练seed、scale搜索、C5、G8、过程reward、DFQL、Navtest或任何暂停的特征侧任务。基础控制若无法保持匹配，应先报告比较不可识别并交Lead处理，不悄悄替换控制或扩矩阵。

正式注册和有限真实运行双审完成后，仍由 Lead 的 `RUN_APPROVAL` 决定启动；评价/分析另绑定共同终点。最终交付为完整两臂身份/曝光/成本、固定 DEV 配对主差、全部分项与双向转移、逐 log/空分母说明，以及“收益/交换/未建立收益”结论。资源 wall、GPU allocation/process、CPU、IO与失败成本据实列出；同更新与曝光不意味着同wall成本。

来源和精确小文件 hash 见 `SOURCE_PINS.json`；结构化冻结值见 `COMPARISON.json`。本次只读取已完成小摘要、注册和源码，不读取模型/图像/metric cache、完整TRAIN账本或DEV逐scene数据，不生成轨迹、不评分、不训练、不执行bootstrap。原 `ROLLOUT_SCALE_PROPOSAL` 与 evidence decision 未修改。
