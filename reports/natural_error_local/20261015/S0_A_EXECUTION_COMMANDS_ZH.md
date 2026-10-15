# 已执行入口、固定观察和恢复边界

这是正在运行的共享基础阶段记录，不是 N/P/R 结果。实际训练为
`formal_S0_A_seed42_exposurematched_v1`，原模型/训练器 `1493deda…`，包装层
`dce41317…`。报告提交的 SHA 不能替代这两个身份。

## 已执行训练与 profile

完整、无省略的 argv 与配置已保存在：

- [正式训练 argv](S0_A_FORMAL_COMMAND_v1.json)：登记时字段仍为
  `REGISTERED_NOT_YET_LAUNCHED`，不回写该冻结文件；实际启动证据在
  [LAUNCH](S0_A_FORMAL_LAUNCH_v1.json) 和 [真实更新窗口](A_BASE_FORMAL_EARLY_WINDOW_v1.json)。
- [正式配置](S0_A_FORMAL_CONFIG_v1.json)、[曝光与资源登记](S0_A_FORMAL_REGISTRATION_v1.json)。
- [profile 配置](S0_A_PROFILE_CONFIG_v2.json)、
  [80-update 及原生恢复结果](A_BASE_PROFILE_ACCOUNTING_v2.json)：
  `actual_invocation_costs.receipts` 含实际首次/恢复命令及退出码。

正式运行位置为 `training-vlawm-tsinghua-zt2`，GPU0–7；共享代码、数据和检查点全部
位于 `/mnt/project`。当前生效授权每次派发重新核验。vla-zt2 和其它撤销主机不连接。
G2 的既有迁移目的地为 tsinghua-zt4，本轮不改变其任务、源码或评价。

原生恢复会恢复 optimizer、scheduler、各 rank RNG、epoch/offset、曝光和计数。
真实 profile 已从 update4 继续到80。正式 run 当前仍在运行，没有执行正式恢复：

```python
# 仅显示同一原生入口的恢复 argv；不是启动第二份训练的命令。
import json, shlex
from pathlib import Path

root = Path('/mnt/project/natural-error-local-artifacts/20261015')
run = root / 'students/formal_S0_A_seed42_exposurematched_v1'
assert json.loads((run / 'status.json').read_text())['status'] == 'PAUSED'
registration = json.loads((root / 'S0_A_FORMAL_COMMAND_v1.json').read_text())
argv = registration['command'] + ['--resume', '--acknowledge-stop']
print(shlex.join(argv))
```

执行恢复前还需核实原进程已退出、原生 RUN 锁可获取、`checkpoints/latest` 指向
完整同身份 checkpoint、主机当前授权和剩余预算。继续沿用本任务的 CPU/磁盘边界
监测与外层 child CPU 计量，另存 attempt 收据。不能因单个 SSH 观测失败便再启动一份，
不能重置 warmup 或以新 run_id 冒充从零训练。

## 已执行 FP32 检查与固定观察

[4 场景命令](A_NATIVE_CURVE_ENGINEERING_COMMAND_v1.json) 实际从原生 ZeRO FP32 master
恢复 80-update 工程 checkpoint，使用真实 A 内部当前图像和场景绑定 seed42。
[结果](A_NATIVE_CURVE_ENGINEERING4_v1.json) 明确为工程检查，不是全1416场景曲线。
它不提供新底座共同终点成绩，也不能据此判断辅助任务是否有效。

[观察器实际命令](A_INTERNAL_LEARNING_OBSERVER_COMMAND_v1.json) 与
[首次等待状态](A_INTERNAL_OBSERVER_INITIAL_STATE_v1.json) 绑定固定
10680/21360/32040/42047 四个 milestone。观察器：

- 使用已有 `evaluate_a_base_learning.py` 和原 `load_student/predict_action`。
- 只读 A_calibration 的1416场景；GT在预测后读取。
- FP32 master、FP32推理、TF32关闭、原10步、一条q0。
- GPU/shard数量取自派发时可用显存，允许合法共卡；跳过不可见占用。
- 最多同时加载两份 master，推理并行；不停止其他进程，不产生额外权重副本。
- 40小时等待和8 GPU-process hour子额度；不自动重试失败、不启动训练器。
- 不读取 B/开发结果进行底座选择，不运行中间 Navtest。

底座固定终点完成后，已有 `register_shared_a_anchor.py` 才允许注册新锚点。
它会拒绝未达到42047更新/1345101曝光的结果。新的 B compact cache、自然错误
人口、N探针和条件P/R必须绑定这个新锚点；旧S0工程缓存不能冒充正式新数据。

## 成本边界

`WAITED_CHILD_COST_SNAPSHOT_20261015T2010.json` 汇总实际终止调用的外层
`RUSAGE_CHILDREN`，不再次加上被其包含的 worker 或 native summary。
它是选定收据的阶段快照，尚非最终完整成本账本。未计量的早期管理操作没有补造数字，
运行中的底座 CPU 另列为实时下界。训练过程中的 GPU 分配时间与真正 GPU kernel
活跃时间、CPU busy time、墙钟时间分别记录。
