# 将空闲本机接入既定里程碑评价

2026-10-16 23:21 UTC，原36项SFT/RL里程碑队列已由单一新owner接管。已完成18项、待完成18项；本次接管没有启动新策略评价，没有改变训练、评分、seed、预算或观察点。现场A为13,931/20,000 RL更新，B为46,774/100,000 SFT更新，原训练PID及启动时间均保持，日志继续推进。

## 实际资源与执行选择

- `training-vlawm-tsinghua-zt`：接管前两次检查八卡全部空闲，可接下一DEV评价。
- `training-vla-zt`：本机八卡已空闲，加入同一队列，可并行接固定TRAIN诊断；先前本机的其他真实任务没有被本任务停止。
- `training-rl-zt2`：八卡各约1942 MiB占用，NVML进程在容器内不可见，准入判定为false；保留并跳过，不假定是占卡脚本。
- `tsinghua-zt2`的Base训练、`tsinghua-zt3/zt5`约70 GiB/卡的不明占用及其他任务均未修改。A/B各自的八卡训练主机保持。

三个候选评价主机有24个不同物理GPU UUID。仍按实际资源二次采样、可见owner、显存及授权检查派发，每台最多一个既定八shard任务。本次没有发出压力worker或训练进程终止信号；不存在为腾资源停止其他真实工作的操作。

## 最小调度修改与已执行验收

旧控制器对所有目标使用SSH；本机`training-vla-zt`没有可解析的自SSH alias。现在只有目标等于`require_dispatch()`核定的当前主机时，直接用参数数组执行同一命令。其他主机继续原SSH路径，所有路径仍先检查授权。

本机子命令使用中性工作目录`/`。这是为了防止新controller checkout覆盖明确指定的`PYTHONPATH`：真实CPU检查已确认派发模块仍来自不可变`225b9af660cb0210208e2c46fc4216d0f151275b`，而不是当前工作区。没有新增shell拼接或修改SSH配置。

7项针对性测试通过，包括既有队列去重、活动任务排他、二次资源检查，以及新增本机argv/cwd、远程SSH保持、撤销授权拒绝。实际本机resource probe读取了8卡UUID；本机模块路径检查通过，0新worker/候选。完整`--once`执行正确重读18个已完成结果、保持18项待办，未重复启动任何评价。

修改仅在[CPU控制器](../../scripts/analysis/fm_transfer/milestone_controller.py)及[对应测试](../../tests/rl/test_fm_transfer_parallel_milestones.py)，源码提交为`a28249336531a2908eaca35d40cbbaeacc01b6f3`。

## 单一owner与源码身份

旧owner PID193325/start1362183928通过现有`STOP_AFTER_TICK`在无active/ready任务的边界退出。退出状态为`STOPPED_CONTROLLER_ONLY`，其文件锁已释放，历史目录与日志完整保留。之后才运行新控制器的`--once`验收，再启动持久owner PID322120/start1363146860；实际`/proc`源码目录和排他锁均核验。

运行目录：`/mnt/project/fm-transfer-20261016/milestone_controller_v4`。旧目录`milestone_controller_v3`保持停止，不应恢复为第二owner。

| 组件 | 继续使用的源码 |
|---|---|
| 正式RL/派发 | `225b9af660cb0210208e2c46fc4216d0f151275b` |
| 原始观测推理/评分 | `45d2318d2dec265583860a26f891d8b79bb6068a` |
| CPU评价控制器 | `a28249336531a2908eaca35d40cbbaeacc01b6f3` |

完整36项job定义与旧owner逐项一致。配对分析器、原完整评价分析器及FM合同文件相对旧controller source的hash相同。没有热修改任何运行快照，也没有改已完成的轨迹、score或摘要。

机器证据：[接管执行摘要](local_evaluation_handoff_execution.json)。完整私有准入与owner收据保留在`/mnt/project/fm-transfer-20261016/resources/milestone_local_migration_20261016/`。当前仍无新的最终Navtest结果。

## 实际启动命令

以下命令先加`--once`实际执行完成，再按原样启动持久controller。重复执行前必须先检查现有PID/启动时间和owner锁；不能与当前owner并行启动。

```bash
cd /mnt/project/fm-transfer-20261016/source/a28249336531a2908eaca35d40cbbaeacc01b6f3
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD" /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.milestone_controller \
  --evaluation-source /mnt/project/fm-transfer-20261016/source/45d2318d2dec265583860a26f891d8b79bb6068a \
  --dispatch-source /mnt/project/fm-transfer-20261016/source/225b9af660cb0210208e2c46fc4216d0f151275b \
  --host training-vlawm-tsinghua-zt \
  --additional-hosts training-vla-zt training-rl-zt2 \
  --include-rl \
  --control-dir /mnt/project/fm-transfer-20261016/milestone_controller_v4
```

下一次实际派发仍会重新检查资源；23:21的空闲快照不当作持续授权或持续空闲的证据。
