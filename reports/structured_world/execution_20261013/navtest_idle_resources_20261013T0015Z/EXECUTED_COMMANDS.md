# 本轮实际执行命令

以下来自已执行的启动收据。当前 dispatcher 已在运行，不需要再启动第二个。代码和状态位于共享 `/mnt/project`；Python 环境为每台机器已验证的隔离副本。

当前完整 Navtest 队列（G1 70k 后继续 G0 50k）；主机 `training-vlawm-tsinghua-zt`：

```bash
/root/miniconda3/envs/navsim/bin/python -B -u /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_dispatcher_v3/run_navsim_navtest.py --serve --request-key G1_FULL_UNIFORM_0070000 --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225 --inputs /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_current_v1_0c54 --routes /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/RUNTIME_ROUTES.json --cadence-policy /mnt/project/structured-world-fgtr-round1-artifacts/20261008/NAVSIM_NAVTEST_CADENCE_20261012.json --score-workers 32 --canonical-python /root/miniconda3/envs/navsim/bin/python
```

收据：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/CONTROLLER_V3_LAUNCH.json`。

G3 100k 替代 CPU scorer（现已完成）；主机 `training-vlawm-tsinghua-zt`：

```bash
/root/miniconda3/envs/navsim/bin/python -B -u /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_dispatcher_v2/stream_navsim_navtest_pdms.py --predictions /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0100000 --index /mnt/project/local-interaction-mask-v2-artifacts/20260927/navtest_cache_index.json --devkit /mnt/project/local-interaction-mask-v2-artifacts/20260927/reference_sources/navsim-v1.1 --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0100000/scores --workers 24
```

收据：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/SCORER_REPLACEMENT.json`。

G3 100k 完整配对分析（现已完成）；主机 `training-vlawm-tsinghua-zt`：

```bash
/root/miniconda3/envs/navsim/bin/python -B /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_analysis/analyze_navsim_navtest.py --scores /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0100000/scores --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/G3_100K_COMPARISONS.json --reference same_G3_80k=/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0080000/scores/q_final/scenes.csv --reference same_G3_90k=/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0090000/scores/q_final/scenes.csv --reference same_G3_70k=/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0070000/scores/q_final/scenes.csv
```

收据：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/G3_100K_COMPARISONS_COMMAND.json`。

tsinghua-zt 空闲恢复监控；主机 `training-vlawm-tsinghua-zt`：

```bash
/usr/bin/python3 -B -u /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_pressure_monitor/navtest_idle_pressure.py --policy /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/idle_pressure/training-vlawm-tsinghua-zt/POLICY_V2.json
```

收据：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/idle_pressure/training-vlawm-tsinghua-zt/LAUNCH_V2.json`。

tsinghua-zt2 空闲恢复监控；主机 `training-vlawm-tsinghua-zt2`：

```bash
/usr/bin/python3 -B -u /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/frozen_pressure_monitor/navtest_idle_pressure.py --policy /mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/idle_pressure/training-vlawm-tsinghua-zt2/POLICY_V2.json
```

收据：`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_resources_20261013T0015Z/idle_pressure/training-vlawm-tsinghua-zt2/LAUNCH_V2.json`。

已执行的 CPU 回归检查与技能校验（当前任务 worktree）：

```bash
PYTHONPATH=tools/structured_world /usr/bin/python3 -B -m unittest tools.structured_world.test_navtest_idle_pressure tools.structured_world.test_navtest_temporary_hosts tools.structured_world.test_navtest_load_oom_review tools.structured_world.test_navtest_scoring_startup_review
python3 /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/navtest-parallel-eval
python3 reports/structured_world/execution_20261013/navtest_idle_resources_20261013T0015Z/collect_snapshot.py
```

`collect_snapshot.py` 仅收集完整结果汇总与当前资源/训练快照；重跑会刷新本目录中的观察记录，不启动模型或新的实验。初始 unittest 的模块路径错误保存在 CPU_IMPORT_PATH_FAILURE.log；随后 conda Python 缺少 pidfd API 的失败保存在 CPU_PYTHON_PIDFD_FAILURE.log。最终正确的 system Python 命令输出保存在 CPU_VALIDATION.log。
