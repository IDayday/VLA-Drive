# D46 FPO 两条研究线 · 2026-11-04

当前优先进行训练代码提速，**正式训练0组**。成熟线已完整保存于1921/inner1，改进线47/inner1；Adam、scheduler、各rank RNG与待完成rollout保留。工程测试使用这些状态的独立副本，测试权重不作为训练结果。主开发与控制在 vlawm-tsinghua-zt3，其它RL终止状态保留。

最新工程测量、改动与待解决问题见 [提速进度](ENGINEERING.md)。D系最高完整单轨迹PDMS仍为 **90.35363365**，来自成熟FPO1711；提速测试尚未产生新的正式部署分数。

- **FPO_MATURE**：原FPO自身native1711/inner1接续，16卡Tsh5+WMzt，冻结科学源82d5f784；继承原Adam、scheduler、私有RNG、pending MC4材料。IL46原始初始化不变，历史完整最佳90.32032058。
- **FPO_VERIFIED**：同一完整IL46、fresh Adam独立初始化。B64/G16/M2、r16共享28层attention LoRA、SafetyEP+批级std、MC4 FPO不变。真实TRAIN B1对照，严格安全舒适且PDMS≥.9、改善≥.02的候选才作为教师；G32扩展只挖教师，不进入FPO ratio。CFM MC2和真实B1最后一步监督；保持初始同run IL46与自身高质量B1，bank全TRAIN容量，部署不含bank或评分器。

正式权重不继承资格权重。新代码复用稳定原生Trainer/Checkpoint，仅增加状态初始化和附加目标接口。17项针对性CPU检查通过；实际八卡首个更新、pending内层保存与冻结参数检查通过；中途恢复与梯度校准已通过：8个实际救援场景的加权CFM/FPO梯度范数比中位数14.14%、ODE-last/FPO为5.46%；保持梯度在初始自目标处为零，后续收益待评测。未据资格宣称提分。

采用原warmup0.08E/hold0.60E/cosine4E/floor0.05，成熟配方LR2e-6。正式配置与校准收据封存在运行账本。先按1/2/3E完整单B1评测复核；原生6E预算不暗改，外层3E复核保存边界。改进线额外早评400/800更新，避免盲目长训。

派发遵守2026-11-04最新五容器白名单；不派发到名单外的旧临时节点，不驱逐未知负载。checkpoint/scene结果位于共享存储；实验暂停与容器授权分别管理。

报告：`/mnt/project/d46-fpo-two-routes-20261104/report.html`。主报告展示提速开发、已保存路线与最新完整分数。原完整B1协议K10/shift5/seed2101、12146场景/44物理logs/invalid0、官方评分不变。NAVTEST只作预定评测/模型选择，标签不进入训练。


## 当前报告与两条路线的区别

主报告：`/mnt/project/qwen-world-performance-rl-20261103/report.html`。
独立报告：`/mnt/project/d46-fpo-two-routes-20261104/report.html`。
页面开头展示当前两条训练、全rank共同更新/有效曝光、保存点、新完整成绩和评测队列；两张SVG框图并排，后接逐项差异与后续决策。历史D/AUG/IL70实验在后面默认折叠。常规30分钟、完整结果事件立即更新，保留原唯一report watcher。

| 比较项 | 成熟FPO延长 | FPO Verified改进 |
| --- | --- | --- |
| 本轮起点 | 自身1711/inner1，继承Adam/RNG | 相同原始IL46，新Adam0，不继承资格权重 |
| 资源 | 两节点16卡 | 一节点8卡 |
| 主采样/更新 | B64/G16/M2，MC4 | 相同；G32扩展只作教师 |
| 核心损失 | FPO energy clip .01 + 原IL46 W3 reference KL | 原损失 + .05 CFM + .02真实B1末步 + B1高质量保持 |
| 教师 | 无新增教师 | 当前TRAIN G16/额外G16中安全舒适且PDMS≥.9、比真实B1高≥.02 |
| 历史保持 | 冻结IL46参考KL | 同run初始IL46与自身过去strict-high B1；系数.05，退化≥.02增至.25 |
| 调度 | LR2e-6，warmup.08E/hold.60E/cosine4E/floor5% | 相同注册调度，独立新曝光时钟 |
| 部署 | 原生单B1 K10/shift5 | 相同；教师、bank、评分器不部署 |
| 待验证 | 长训是否超过历史90.32032058 | 好候选能否转为B1改善，且高分退化减少 |

辅助系数是原始损失权重，不是梯度占比或beta50预算。当前16卡与8卡不是同资源消融；不同起点进度的早期分数不用于判定最终上限。所有新分数只接受完整12146场景/44logs/invalid0；历史项目IL70/SafetyEP91.45359756另列，不是D父基线。

[成熟线框图](plots/FPO_MATURE.svg) · [改进线框图](plots/FPO_VERIFIED.svg)


## 本轮首个完整评测

成熟FPO Adam1711（自身原生续训起点）的完整NAVTEST：**PDMS 90.35363365**，12146场景、44物理logs、invalid0。
相同IL46基线89.49552098提高 **+0.85811267**；超过此前D系FPO1614的90.32032058，增量 **+0.03331308**。
对IL46逐场景：修复零分243个，新增零分72个，其中68个来自原高分场景。说明修复与遗忘同时存在，新增保持路线仍需完整评测证明收益。

仅提交[完整分数](results/FPO_MATURE_u001711/SUMMARY.json)与[配对分析](results/FPO_MATURE_u001711/PAIRED_ANALYSIS.json)，不提交开发验证快照。该点为续训起点的完整评测，不能归功于本次1711之后的新增更新；改进线尚无完整部署成绩。
