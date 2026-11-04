# D46 FPO 两条研究线 · 2026-11-04

**最新分析（11月4日）：** [AUG迁移复核与当前FPO退化分析](AUG_MIGRATION_REVIEW.md)。成熟线完整PDMS已从1711步90.35363365回落至2800步90.06854069；Verified400步89.68958784（相同IL+0.19406687，配对置信区间跨零）。目前没有92分结果。下面部署段落保留原08:16时点，实时进度以HTML为准。

**两条正式训练已恢复，共24张GPU。**成熟FPO在 `vlawm-tsinghua-zt5 + vlawm-tsinghua-zt` 使用16卡；Verified改进线在 `vlawm-zt` 使用8卡。2026-11-04 08:16 UTC实测所有rank分别提交至2028、78次Adam更新。动态进度以HTML为准。

D系最高完整单轨迹PDMS为 **90.35363365**（成熟FPO1711），相同IL46基线 **89.49552098** 提高 **0.85811267**。该点是本轮续训起点的评测，不归功于提速改动。1921、2000步的完整NAVTEST已自动运行；Verified尚无完整RL部署分数。目标92+仍待真实结果。

| 正式路线 | 当前执行起点 | 方法与资源 | 自动早评 |
| --- | --- | --- | --- |
| FPO_MATURE | 自身1921/inner1完整状态 | B64/G16/M2，MC4 FPO；两节点16卡 | 1921/2000/2400/2800步 |
| FPO_VERIFIED | 自身47/inner1完整状态 | 相同FPO主目标，加验证教师CFM、真实B1末步监督、历史高分保持；单节点8卡 | 200/400/800步 |

两条路线最初均独立从同一个完整D IL46初始化；Verified未继承成熟线权重或Adam。此次仅迁移各自原生训练状态，保留optimizer、scheduler、RNG、pending rollout和历史教师。原pending更新按旧布局完成，之后的新collection才启用合批实现。工程测试权重不进入正式训练。

原配方LR峰值2e-6，warmup0.08E/hold0.60E/cosine4E/floor0.05，目标6个场景曝光epoch。1/2/3E持续评估，不再因达到3E机械停训。每100次Adam保存完整原生状态，轮换最近10份；额外固定最佳、latest和正在评测的检查点。异常时从完整保存点恢复，未保存更新可能需要重做。

提速实现、实测与限制见[工程报告](ENGINEERING.md)。最新五容器白名单覆盖此前临时授权；没有向名单外节点部署。`vlawm-tsinghua-zt3`的D IL继续运行；`vlawm-tsinghua-zt2`存在无法归属的GPU负载，未抢占。32卡重分片代码尚未部署，不能报告其实际加速或GPU恢复资格。

报告：`/mnt/project/qwen-world-performance-rl-20261103/report.html`；独立页：`/mnt/project/d46-fpo-two-routes-20261104/report.html`。后台自动发现检查点、分片评测和保留恢复点；常规30分钟刷新，有完整新分数立即刷新。B1 K10/shift5/seed2101、12146场景/44物理logs/invalid0及官方评分不变。NAVTEST只作预定评测与模型选择，标签不进入训练。

## 当前报告与两条路线的区别

主报告：`/mnt/project/qwen-world-performance-rl-20261103/report.html`。
独立报告：`/mnt/project/d46-fpo-two-routes-20261104/report.html`。
页面开头展示当前两条训练、全rank共同更新/有效曝光、保存点、新完整成绩、完整训练吞吐与评测队列；两张SVG框图并排，后接逐项差异与后续决策。历史D/AUG/IL70实验在后面默认折叠。常规30分钟、完整结果事件立即更新，保留原唯一report watcher。

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
| 待验证 | 长训是否超过已确认90.35363365 | 好候选能否转为B1改善，且高分退化减少 |

辅助系数是原始损失权重，不是梯度占比或beta50预算。当前16卡与8卡不是同资源消融；不同起点进度的早期分数不用于判定最终上限。所有新分数只接受完整12146场景/44logs/invalid0；历史项目IL70/SafetyEP91.45359756另列，不是D父基线。

[成熟线框图](plots/FPO_MATURE.svg) · [改进线框图](plots/FPO_VERIFIED.svg)


## 本轮首个完整评测

成熟FPO Adam1711（自身原生续训起点）的完整NAVTEST：**PDMS 90.35363365**，12146场景、44物理logs、invalid0。
相同IL46基线89.49552098提高 **+0.85811267**；超过此前D系FPO1614的90.32032058，增量 **+0.03331308**。
对IL46逐场景：修复零分243个，新增零分72个，其中68个来自原高分场景。说明修复与遗忘同时存在，新增保持路线仍需完整评测证明收益。

仅提交[完整分数](results/FPO_MATURE_u001711/SUMMARY.json)与[配对分析](results/FPO_MATURE_u001711/PAIRED_ANALYSIS.json)，不提交开发验证快照。该点为续训起点的完整评测，不能归功于本次1711之后的新增更新；改进线尚无完整部署成绩。
