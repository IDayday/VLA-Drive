# B-20k终点只读汇总衔接

2026-10-17 13:55 UTC，既定B-20k终点的CPU后续分析已通过14项针对性检查，并在授权vla-zt启动唯一等待进程。此时正式训练约13.9k，20k结果尚未产生。本记录是执行准备与实际等待证据，不是终点效果报告。

复用已验收B-15k衔接的执行账本、输出保护和失败保留逻辑，保留原阶段函数不变。终点条件改为world8/COMPLETE/next_inner0，并等待八份正式rank结束记录、已有独立任务产生的完整训练摘要、精确成本快照、保存批次诊断及DEV/TRAIN的双网格评价。中间READY检查点不能通过终点条件；预算必须为20,000 optimizer更新、5,000份fresh全局rollout、40,000场景曝光和320,000条新候选，科学身份与关闭的调用账本均需匹配。

已有训练预算汇总仍由原`RL_endpoint_summaries_20261016_v1`任务负责，本任务不重复调用它。全部前置证据齐全、16个评价worker按PID和启动时间确认退出后，顺序执行八项现有分析：人口/epoch流、两个评价人口的安全转移、两个评价人口对15k的变化、两种部署网格差、完整DEV曲线、原生optimizer和训练退出核验。分析使用8dbd快照；正式训练225b、数值评价45d、保存批次诊断fd和完整预算汇总d584均保持不变。

终点核验复用B中期的评价/身份/hash/FP32 Adam检查，以及A20k已经执行过的训练进程退出探测。B需480份Adam状态与scheduler均到20k；同时核对原始观测入口、完整评价分母和实际推理次数。训练退出不是由一个结果文件推断，而是额外探测准确PID/start-ticks及该run实际进程。若出现新的训练attempt或未关闭账本，保留证据并拒绝自动宣称完成。

14项已执行检查包括：两个脚本编译、原阶段函数不变、真实A20k COMPLETE通过、真实B10k READY被拒绝、真实A完整预算通过、错误更新数/未关闭账本/不同run身份被拒绝、没有重复训练汇总或已有输出、固定20k对15k比较、当前B未完成时等待、拒绝仍存活的进程以及真实`--once`执行。没有增加推理、评分或optimizer更新。它们不代替未来B20k的真实核验。

实际owner为PID 538976、start-ticks 1368390701；启动时CPU affinity128、load14.43、可用内存约859GiB，CUDA设备对该任务不可见。资源政策hash为`79cca8cfdb98ed7f46af554633c4734e61def4ee02ff656220415013b2082097`。六小时截止只约束这个有限等待任务；超时不意味着训练停止，不会重启或终止任何训练。阶段失败不自动重试，既有输出不覆盖，正常训练和其他任务均不受影响。

已执行的只读检查命令如下。当前已有唯一owner，不得再次启动普通入口：

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/mnt/project/fm-transfer-20261016/source/8dbd2bb2c82026d3a8a0b4b96c9b0ba211248f81 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  /mnt/project/fm-transfer-20261016/resources/B_RL_20k_analysis_handoff_20261017/run.py --once
```

[衔接源码](reproduce_B_RL20k_analysis_handoff.py)、[终点核验源码](reproduce_B_RL20k_completion.py)、[14项检查与实际启动记录](B_RL20k_analysis_handoff_validation.json)。终点结果产生后仍需阅读实际分析、检查图表并发布结论。最终Navtest继续由独立既有控制器在两条20k终点和全部阶段评价完成后启动；没有新增算法、seed或best-of-N。
