# 实际执行命令

以下命令取自启动/完成收据，均已执行。控制器与恢复监控已在后台运行，不要并行启动重复实例。所有 checkpoint 与模型/评分合同保持原值。

G0 立即启动（此轮已完成）；主机 `training-vlawm-tsinghua-zt`：

```bash
/root/miniconda3/envs/navsim/bin/python -B -u /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_dispatcher_v3/run_navsim_navtest.py --serve --request-key G0_NO_FUTURE_LABEL_0050000 --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225 --inputs /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_current_v1_0c54 --routes /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/RUNTIME_ROUTES.json --cadence-policy /mnt/project/structured-world-fgtr-round1-artifacts/20261008/NAVSIM_NAVTEST_CADENCE_20261012.json --score-workers 32 --canonical-python /root/miniconda3/envs/navsim/bin/python
```

收据：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/G0_CONTROLLER_LAUNCH.json`。

G1 原输出恢复核验；主机 `training-vlawm-tsinghua-zt`：

```bash
/usr/bin/python3 -B /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/frozen_recovery_v1/run_navsim_navtest.py --recover-infrastructure --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225 --inputs /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_current_v1_0c54 --request-key G1_FULL_UNIFORM_0070000 --recovery-evidence /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/G1_COMPLETED_SSH_RECOVERY_EVIDENCE.json
```

收据：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/G1_RECOVERY_COMMAND.json`。

G1 已评分记录汇总；主机 `training-vlawm-tsinghua-zt`：

```bash
/root/miniconda3/envs/navsim/bin/python -B -u /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_dispatcher_v3/stream_navsim_navtest_pdms.py --predictions /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G1_FULL_UNIFORM_0070000 --index /mnt/project/local-interaction-mask-v2-artifacts/20260927/navtest_cache_index.json --devkit /mnt/project/local-interaction-mask-v2-artifacts/20260927/reference_sources/navsim-v1.1 --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G1_FULL_UNIFORM_0070000/scores --workers 8
```

收据：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/G1_SCORE_FINALIZATION_LAUNCH.json`。

G1 完整配对分析；主机 `training-vlawm-tsinghua-zt`：

```bash
/root/miniconda3/envs/navsim/bin/python -B /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_analysis/analyze_navsim_navtest.py --scores /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G1_FULL_UNIFORM_0070000/scores --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/G1_70K_COMPARISONS.json --reference same_G1_60k=/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G1_FULL_UNIFORM_0060000/scores/q_final/scenes.csv --reference same_G1_50k=/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G1_FULL_UNIFORM_0050000/scores/q_final/scenes.csv --reference same_step_G3_70k=/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0070000/scores/q_final/scenes.csv
```

收据：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/G1_70K_ANALYSIS_COMMAND.json`。

G0 完整配对分析；主机 `training-vlawm-tsinghua-zt`：

```bash
/root/miniconda3/envs/navsim/bin/python -B /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_analysis/analyze_navsim_navtest.py --scores /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G0_NO_FUTURE_LABEL_0050000/scores --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/G0_50K_COMPARISONS.json --reference same_step_G1_50k=/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G1_FULL_UNIFORM_0050000/scores/q_final/scenes.csv --reference same_step_G3_50k=/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0050000/scores/q_final/scenes.csv --reference same_G0_25k=/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G0_NO_FUTURE_LABEL_0025000/scores/q_final/scenes.csv
```

收据：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/G0_50K_ANALYSIS_COMMAND.json`。

已接续的常驻队列控制器；主机 `training-vlawm-tsinghua-zt`：

```bash
/root/miniconda3/envs/navsim/bin/python -B -u /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/frozen_dispatcher_v4/run_navsim_navtest.py --serve --request-key G0_NO_FUTURE_LABEL_0050000 --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225 --inputs /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_current_v1_0c54 --routes /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/RUNTIME_ROUTES.json --cadence-policy /mnt/project/structured-world-fgtr-round1-artifacts/20261008/NAVSIM_NAVTEST_CADENCE_20261012.json --score-workers 32 --canonical-python /root/miniconda3/envs/navsim/bin/python
```

现有压力恢复监控；主机 `training-vlawm-tsinghua-zt`：

```bash
/usr/bin/python3 -B -u /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_pressure_monitor/navtest_idle_pressure.py --policy /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/idle_pressure/training-vlawm-tsinghua-zt/POLICY_V3.json
```

现有压力恢复监控；主机 `training-vlawm-tsinghua-zt2`：

```bash
/usr/bin/python3 -B -u /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_pressure_monitor/navtest_idle_pressure.py --policy /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_g0_immediate_20261013T0120Z/idle_pressure/training-vlawm-tsinghua-zt2/POLICY_V3.json
```

当前任务 worktree 中已执行的 CPU 检查（35 项通过）：

```bash
PYTHONPATH=tools/structured_world /usr/bin/python3 -B -m unittest tools.structured_world.test_navtest_completed_ssh_review tools.structured_world.test_navtest_scoring_startup_review tools.structured_world.test_navtest_load_oom_review tools.structured_world.test_navtest_temporary_hosts
```

汇总完整结果的已执行命令：

```bash
python3 reports/structured_world/execution_20261013/navtest_g0_immediate_20261013T0120Z/collect_results.py
```
