# 四种从头IL损失进展与新增Navtest

新锁定七个完整原生checkpoint：原L1控制5k；PHYSICAL9k、TIME_RESIDUAL4k、DELTA7k、HUBER6k；原IL96k、优化轨迹IL27k。硬链接保留全部native/RNG/Adam和身份文件，避免滚动清理影响。每个检查点完整12146场景、seed2101、canonical v1.1、单候选clean-x K10 shift5 ODE；七组共85022次逻辑单候选评测和七次native/export一致性检查，无新增seed/搜索/选轨。

性能比较使用相同步数。四种新损失的5k结果与新增原L1控制5k单独比较；TIME4k只作中期观测，达到5k后原训练owner自动执行完整Test。不同步数的最新点只用于各自学习曲线。已有PHYSICAL5k80.101589、DELTA5k79.186748、HUBER5k72.150853，都是完整12146、invalid0，不是训练reward；目前原L1同预算对照与TIME5k未完成。保留预登记共同10k筛选终点，不据5k临时换规则/提前终止。

本轮沿用全部训练配方、100k scheduler、原始GT和四辅助任务；资格权重未作初始化。训练loss定义不同，数值不可直接比较。原RL三组均50k完成，无新RL checkpoint，本轮不重复历史RL。

允许节点：vla-zt、tsinghua-zt/zt3/zt5，以及临时不限时zt6、rl-zt2和刚授权vla-zt3；tsinghua-zt4仍禁用，其余旧临时节点仍禁用。新增vla-zt3实际八卡空闲已核实，可用于并行评测。进程归属、物理UUID、显存和共享lease继续复核；不抢占未知进程。

原轨迹20k的旧评测因zt4撤权，rank5/6在场景边界停止，已保留11593个正确结果，553个待补；全量汇总正确拒绝不完整分母。新增独立恢复入口只在合法vla-zt3补rank5/6缺失部分，保持原fd2评价源码/identity/seed/导出/评分；旧rank耗时与failure保留，不把中断记作0分。没有重跑其他11593场景。

实际入口：

```bash
python -m tools.qwen_world_postcot.latest_snapshot_owner \
 --root /mnt/project/qwen-world-postcot-20261025/loss_and_latest_NAVTEST_20261029T1425Z \
 --report-dir reports/qwen_world_loss_and_latest_navtest_20261029
python -m tools.qwen_world_postcot.finish_interrupted_label20k --check-only
python -m tools.qwen_world_postcot.finish_interrupted_label20k
python -m tools.qwen_world_postcot.loss_progress_analysis \
 --control /mnt/project/qwen-world-postcot-20261025/loss_and_latest_NAVTEST_20261029T1425Z/CONTROL_IL_005000/evaluation \
 --output /ABSOLUTE/NEW_ANALYSIS_OUTPUT
```

运行时源码保持不可变。最后一个GPU worker退出后唯一publisher写入完整逐点结果、同预算对照、G+/G−、安全转移及物理log配对bootstrap，推送本评测分支；未完成项明确保留待补，不填部分分数。
