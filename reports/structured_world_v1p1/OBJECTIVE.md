在 IDayday/VLA-Drive 中继续开发与实验。

以提交：
393c53bbd685c77694d342496a0b5cd7aa1ae735
为审查和开发起点。

本轮目标不是重跑完整A0/A1/A2/B/C/D/E矩阵，而是：
修正测量 → 恢复世界分支可学习性 → 保真接入原策略 → 少量规划对照。

如已提供 CODEX_GOAL_STRUCTURED_WORLD_V1P1_393c53b.md，先完整读取并执行其细则。
不要只输出计划；实际修改代码、运行测试和有预算限制的实验。

一、工作区和旧证据

读取AGENTS.md、CODEX_GOAL_STATE.md和reports/structured_world_v1。

创建隔离worktree与：
feature/structured-world-v1p1-signal-rehab-<实际日期>

不覆盖原工作区、数据、检查点、缓存或报告。
不自动stash/reset/clean/force push，不停止其他任务。

旧campaign的11639/12000步封存。本轮建立新run ID、独立预算和报告目录。
所有新实验记录代码SHA、配置、数据与缓存指纹；中途修改实现必须新建run。

保留当前三前视、原Qwen3-VL/DiT、单候选、10FM步、原1696开发集和PDM协议。
本轮不同时改成单前视、环视、历史图像、多候选选轨或RL。
navtest不参与调参。

二、P0：先重算评估，不训练

检查：
tools/structured_world/evaluate.py
starVLA/model/modules/structured_world/matching.py
losses.py
以及最终汇总代码的实际调用链。

现有“类别无关2m召回”复用了包含类别、尺寸和yaw的训练匹配。
保留旧指标并标legacy，新增独立的当前几何匹配和同类约束匹配。

评估匹配必须：
- 与训练匹配分离。
- 几何召回不依赖类别、尺寸、yaw或未来轨迹。
- 阈值内一对一最大匹配数优先，再按距离破同分。
- 分开记录objectness过滤、ROI/FOV过滤和几何匹配。
- 保留完整GT和固定K提案诊断，不用GT数量控制输入或候选数。

必须复现以下测试：
GT位于(10,0)、(15,0)，类别0、1。
预测中心完全正确但类别互换，其他bbox相同。
旧训练匹配可能交换目标、给出0几何召回。
正确的类别无关2m召回应为100%，类别正确2m召回应为0%。

另测空GT、重复预测、阈值边界、位置正确但尺寸/yaw错误。

优先读取已保存的npz预测和v6目标，重算所有旧感知/运动指标。
不重新跑全量VLM，不改变既有轨迹与PDMS。

重点输出：
raw中心分布、ROI/FOV外比例、no-object概率、有效预测数、
无objectness过滤的固定K召回、带过滤召回、类别正确召回、
当前目标覆盖与运动有效覆盖。

检查C_support和adapter为什么TP=0且FP=0：
区分no-object塌缩、位置越界和匹配问题，不直接写成“全部因为评估错”。

同时分解A1相对A0的DAC/NC/TTC/EP和原高分→零分场景。
验证ego训练标签的坐标、归一化、yaw和时间采样与原模型一致；
推理回归通过不自动代表训练监督正确。

三、P1：无参数更新，隔离token插入扰动

保留旧legacy_pre_action布局。

新增append_tail布局：
原完整token序列、原action位置、原图像位置和原prefix位置编码不变；
world tokens追加在原完整序列之后；
使用原因果mask。

提取：
H_action_native：原位置的action hidden。
H_world：末尾world hidden。

新增桥：
H_action = H_action_native + g * Adapter(H_action_native, H_world)
g初始化为0。

原VLM和DiT冻结。只有native action条件不变时，
才能把gate=0称为保真初始化。

在固定64场景、共同噪声和原精度下比较：
原A0；
旧布局未训练；
append_tail未训练；
append_tail+gate0。

记录prefix/action hidden与最终轨迹差异。
不能通过放宽容差或只绕过全部新代码来假装通过。
原prompt中禁止出现GT自车未来、GT周车或未来图像。

四、P2：先让世界分支学会，不更新驾驶策略

使用v6标签，冻结原DiT、视觉和Qwen。
训练Reader、queries、投影、bbox和motion头。
冻结Qwen参数，但不能no_grad整段Qwen前向。

先做两个匹配的隔离实验：
W_PRE：Reader输出直接接结构化头。
W_POST：Reader输出经过冻结Qwen再接同容量结构化头。

使用相同输入、目标、查询数、初始化和有效曝光。
先16场景定位，再64场景拟合。
不要重复旧标签200步实验并当作新证据。

根据证据检查并修复：
1. 米制坐标直接输出与loss缩放的参数化组合。
2. 未匹配slot的no-object监督完全依赖预测位置形成的盲区。
3. future=pred_center+offset导致motion同时拉动当前框。

优先使用可验证的坐标编码/解码与参考点残差；
监督资格基于当前观测/标定，不把不可见区域强行当空；
未来位移标签用gt_future-gt_center构造，GT只进loss不进前向。

记录米制误差、分类分布、slot多样性、匹配变化、
各任务共享梯度、裁剪前后梯度及真实参数更新。
不要只看总loss，也不要直接把bbox系数扩大几百倍。

64场景工程目标：
类别无关2m召回≥80%，相应precision≥50%，
同时报告完整类别正确召回与失败对象。
这是工程诊断目标，不是论文标准，不准删目标或放宽半径过关。

通过后检查训练域holdout。
运动预测与同一预测当前中心保持不动的基线比较，
使用同一匹配集合，动态/静态对象分开。
无匹配ADE/FDE记缺失，不填0。

只有W_PRE学会而W_POST充分训练后仍失败，
才允许一组小规模Qwen LoRA瓶颈实验。
共享Qwen一旦更新，不得继续自动宣称原action条件保真；
进入规划前需保留独立冻结的原策略参考路径并记录额外成本。

五、P3：验证真正的外部特征

旧三层CNN provider只作历史对照，不冒充预训练BEV。

最多引入一种可核查的外部provider：
优先已有camera-only预训练BEV；
或已有预训练视觉/几何骨干加当前相机校准BEV投影。

后一种必须称为“预训练视觉骨干＋新BEV构建”，
不能称为已经预训练好的BEV模型。

不增加额外视角、历史帧或LiDAR输入。
记录来源、权重hash、训练数据、适配预算与部署成本。

先用v6训练标签做独立感知及holdout核验，再接入Qwen。
没有合适的外部provider时继续image-only主线，
external_prior_status标NOT_TESTED，不能换一个随机CNN充数。

六、P4：仅做三个主要规划对照

世界分支具备可学习性后，进行训练域预训练：
current-only和current+motion各至少4次8192场景完整遍历；
使用匹配的初始化、预算与数据曝光。

记录unique_scenes_seen、sample_presentations、
effective_epochs、global_batch、optimizer_steps。
FM重复次数不计作独立场景。

建议有效global batch=8，条件允许可16。
正确实现梯度累积/DDP全局归一化，避免保留多批完整Qwen计算图。

规划只做：
P_CAPACITY：同容量Reader/token/adapter，仅ego FM。
P_CURRENT：当前bbox监督/预训练。
P_FUTURE：当前bbox＋future motion监督/预训练。

采用相同provider与append_tail。
原Qwen/DiT先冻结，主要训练桥接adapter及允许的Reader/投影。
P_CURRENT和P_FUTURE的预训练及规划曝光保持匹配。

所有组从同一原策略开始，gate=0，实测step0保真。
报告固定中间进度和最终进度，不只挑短暂最高分。

若各组仍共同退化，最多追加一次共同的策略保持配方：
在训练域replay上，用相同FM噪声/时间约束原策略速度场；
所有比较组使用同一规则，保留此前退化记录。
不称其为KL或保证完整分布不变。

不自动恢复8.2亿参数DiT全量训练，不增加RL或新scorer。

统一评估原1696场景，原噪声种子为主；
至多追加两个固定采样种子检查稳定性。
训练种子与采样种子分开，只复跑最有信息量的关键配对。

必须同时比较原A0和匹配控制组。
比另一个退化模型好不等于提升原模型；
gate保持零、返回A0也不等于世界模型有效。

七、预算和结束条件

本轮默认上限：
累计48 GPU-hours及24000 optimizer steps，任一达到停止新训练。
已有更紧用户预算优先。
这是资源上限，不要求耗完，也不允许无限搜索。

provider适配、失败run、修复和更新测试全部计费。
小集隔离每个变体最多1000更新，优先有效batch8。
充分记录有效数据曝光，不沿用旧脚本每次steps<=1000
作为所有新实验的统一限制。

最多两个明确故障假设的修复回合。
未收敛或预算不足时写INCONCLUSIVE，不能宣判方向无效。

复用旧回归，只增测：
独立评估、坐标/位移、监督资格、append_tail/mRoPE、
gate0、标签污染、累计梯度及新配置保存恢复。
不重复全套历史验收消耗主要实验预算。

八、交付

输出：
METRIC_REAUDIT
INJECTION_SHIFT
WORLD_LEARNABILITY
PROVIDER_AUDIT
PLANNING_PILOT
RUN_LEDGER
完整命令、配置、场景CSV、可视化、QUICKSTART及恢复指令。

分别给出：
metric_status
world_status
planning_status
external_prior_status

不要用一个READY覆盖全部结论。

代码阶段提交，最终推送仅本任务新分支并核对远端SHA。
不合并，不上传大权重、缓存、私有数据或密钥。

现在从P0开始执行：
先修正和重算测量，再定位世界分支学习问题，
最后验证低扰动接入规划，而不是先扩大消融矩阵。