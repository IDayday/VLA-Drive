# 完整主实验：实际执行状态

状态时间：2026-10-25T18:20:23.992610+00:00。本报告是该时刻快照；
后续实际进度以外部 `campaign/STATE.json`、run journals、COMPLETE 收据为准。

完整框架已经真实接通并启动正式100k训练，**尚未完成IL/RL与最终规划评测**。
不能将工程PASS、训练loss下降或旧native得分写成本方法的规划收益。

## 已完成与正在运行

| 项目 | 实际状态和证据 |
|---|---|
| 完整模型 | 真实3相机Qwen、144/16/32 memory、467M clean-x actor、全部五loss已测试 |
| post梯度 | 真实autograd进入actor attention/FFN、memory、Qwen；final action projection无post梯度符合契约 |
| 部署隔离 | 独立进程strict导出，2真实场景固定noise的raw/physical逐位一致，删除36,852,497训练期参数 |
| 代码回归 | 固定主运行104 passed，离线RGB恢复保护后的交付109 passed；实际多卡ZeRO2、阶段切换、保存/恢复另有真实收据 |
| TRAIN监督 | 90,890唯一场景；教师450shards/303,972,516,000bytes，零重复 |
| DEV监督 | 1,696场景；教师14shards/5,672,102,400bytes，零重复 |
| 真实PDM初始bank | 1,454,240条、每scene16候选；full-record审计及官方复算通过 |
| RGB decoder | 独立2000updates完成，固定640TRAIN场景；固定DEV16重建L1=0.07258202；不作为规划门槛 |
| 正式IL＋辅助监督 | RUNNING，当前journal1428/100000、cursor=45696，已进入joint阶段 |
| 正式DEV | WAITING_CHECKPOINT，按5k…100k固定20端点完整评估，不评profile权重 |
| 正式canonical RL | WAITING_IL_DEV，50k预算；先对选定IL父本做短程真实PDM工程核验 |
| 本方法Navtest | NOT_RUN，只在实际IL/RL policy经DEV锁定后执行 |

## 正式运行绑定

源码起点425eb5420ce1b592786f455ceea9bd6d124d2c54；实际训练代码
`d54617b3ff02d714c067ce809a50f50cd19bceb6`，封存在
`/mnt/project/qwen-world-postcot-20261025/snapshots/formal_main_v2`。
前一封存v1仅运行过等待输入的CPU协调器，GPU子任务为0；资源可见性检查修复后，
已保留其退出与旧source收据。v2与已通过真实训练验收的20个核心文件SHA一致。

实际正式IL运行：`training-vlawm-tsinghua-zt5` 的8张A80080GB，global32、micro4、
accum1、repeat1。永久白名单节点不受本轮9小时临时额度影响。
输出：`/mnt/project/qwen-world-postcot-20261025/runs/IL_full_seed42`；调度及真实进程身份：`jobs/formal_IL_attempt001`。
本run从固定官方Qwen初始化＋全新actor/memory/readouts开始，是独立lineage，
没有将旧1B头切片加载或把profile checkpoint冒充正式初始化。

首个1000步完整恢复点已保存且原生边界归档；实际联合阶段已开始。Qwen的两组Adam
在1000边界初始化，actor历史与全局scheduler保留，实际incoming Qwen梯度FP32。
100k包含1000接口warm start，随后按原100k cosine联合更新Qwen；辅助权重平滑增至
10/1/0.5/0.1/0.2。阶段准确称“轨迹模仿学习＋表征辅助监督＋行动评价辅助监督”，
`uses_metric_supervision=true, uses_rl=false`。下一阶段才是`uses_rl=true`真实PDM RL。

## 数据、评价与比较边界

TRAIN排除了与DEV/NAVTEST共用的物理日志：90,890场景/147日志；DEV1696/15；
NAVTEST12146/44。不是以16或136个segment目录冒充独立物理日志。
625个TRAIN真实future-pose时间gap按waypoint mask处理；838个teacher clip有真实
时序缺口，其future监督全mask。当前DINO及其他有效监督保留，不填造有效未来。
旧Base与公开WA教师可能见过DEV同源数据，不能称完全未见。

每条候选采用锁定canonical NAVSIM v1.1 reference+one-candidate评价。最终源码的
5次小型TRAIN复算也maxabs=0；这只是最终snapshot的接口核验，不是新策略RL结果。
实际evaluator由固定source contract读取两个绝对metric-cache根：
`/mnt/project/flow-safe-improvement-20261013/metric_cache_v1`与
`/mnt/project/constrained-flow-noise-20261014/metric_cache_v1`，maps为`/mnt/navsim/maps`。
RL YAML是整体严格等值校验的source配方；其中legacy相对cache占位字段不是可调
重定位入口，改变它会拒绝，不能把它当成实际读取路径。

旧native参照在本轮结果前已经绑定：IL77k Navtest88.144945、canonical RL50k
90.513164，同12146场景、seed2101、single-K10-shift5；后者父本是IL53k，不能与
IL77k相减宣称其RL增量。新旧架构、执行精度、训练人口和监督预算不同，最终比较
提供方法级结果，不能直接用于逐模块因果归因。

## 后续由后台流程持续推进

IL每1000保存、每5000完整DEV，最高DEV PDMS锁定IL；完整IL/DEV窗口完成后启动
同一模型的真实canonical Flow-GRPO50k。Qwen和student memory冻结，只训attention
LoRA16；真实PDM reward不由post head替代。最后锁定RL checkpoint再做Navtest。
主采样seed2101；锁定权重后另做2102/2103推理波动，不当作三个训练seed。

主score、NC/DAC/TTC/EP/Comfort/DDC、修复与新增失败、物理日志配对bootstrap，
以相同scene/protocol汇总。现在本方法IL/RL PDMS均为UNTESTED。
训练和部署实测见TRAINING_PROFILE；总成本未完成，部分实测收据和估计分别记录。

原native IL/RL、旧Base和IQE22k等权重/任务/ACTIVE均保留。没有修改旧进程源码，
没有删除旧checkpoint，没有运行本轮提前Navtest或外部Scorer教师。

首次正式1000 checkpoint的固定单DEV场景可视化已真实完成，同一预先锁定1473步
RGB decoder下teacher/student重建L1=0.060794/0.145036；同时保存共同PCA basis。
原图与重建仅存私有artifact，不进Git。这是早期表征诊断，非规划得分、非挑checkpoint。
`offline_student_visualization/IL001000_DEV_fixed1/FINAL_VALIDATION.json`保留全部来源。
