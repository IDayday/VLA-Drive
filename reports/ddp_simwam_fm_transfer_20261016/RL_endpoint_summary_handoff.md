# 完整RL终点的预算汇总衔接

2026-10-16 22:46 UTC，单一CPU等待进程已实际启动，等待A/B各自完整20k RL终点。它只连接已有`simwam_recipe.summarize_training`，不启动训练、推理、评分或Navtest，不修改配方和运行源码。

启动前完成11项有限准入检查，包括真实未完成A/B目录的`--once`：等待成功，0次汇总派发。检查覆盖缺失rank结果、部分rank完成、预算错误、资格模型、错误phase、过期progress、尚未关闭和不一致的成本账本。mock只验证等待条件；真正原生文件hash、日志连续性和成本统计仍由原汇总器验证，最终模型/数据/预算/里程碑仍由已有final lock再次核验。

必要边界：原trainer先写`result_rank*.json`，再关闭相应invocation账本。因此等待任务同时要求八rank结果为正式20k、5000 fresh collections、40000场景曝光、320000候选，原生checkpoint为COMPLETE/inner0，progress一致，且八份对应invocation均关闭到20k。它不在结果文件刚出现但成本尚未关闭时发布最终摘要。正式trainer没有改动。

每组只运行一次现有汇总命令，输出`analysis/<variant>_RL_complete_training.json`，之后用已有`validate_rl_summary`校验。已存在的合法摘要可验证复用；不覆盖旧摘要，不自动重试失败，不把截止时间到达当作训练停止。等待期限24小时，每30秒检查，文件锁和attempt身份阻止重复owner。失败或超过期限保留证据；后续重新启动需明确检查并另建attempt。

调用的不可变汇总源码为`d58485b08078a6319b2835793d8c79ffbbd3e9d5`，其汇总器SHA256为`5ec72ecbe5200c1317dc0afcfd96fa6efcc652c9a8bac73ab0d7bbaa3db09d1e`。新等待脚本SHA256为`17f1278849cf8f4b9899c6d4c77531c19540bb74c4783270ceb4f864d03ed5cf`。八rank成本关闭检查加入前的未部署资格版本及证据保留在私有attempt目录，正式只运行本页对应版本。

实际启动命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/mnt/project/fm-transfer-20261016/source/d58485b08078a6319b2835793d8c79ffbbd3e9d5 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  /mnt/project/fm-transfer-20261016/resources/RL_endpoint_summaries_20261016_v1/run.py
```

后续只有在完整终点出现时才调用：

```bash
python -m scripts.analysis.simwam_recipe.summarize_training --run /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_R_SOURCE_seed42_v1 --output /mnt/project/fm-transfer-20261016/analysis/A_DDP_TIME_RL_complete_training.json
python -m scripts.analysis.simwam_recipe.summarize_training --run /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1 --output /mnt/project/fm-transfer-20261016/analysis/B_SOURCE_ACTION_DIT_RL_complete_training.json
```

这两条终点命令在本页发布时**尚未执行**。等待进程实际PID/启动时间核验通过，A/B训练继续，Navtest仍受原四模型终点及36项里程碑准入控制。

证据：[脚本副本](reproduce_RL_endpoint_summary_handoff.py)、[有限资格检查](RL_endpoint_summary_handoff_qualification.json)、[实际启动与等待](RL_endpoint_summary_handoff_execution.json)。源码副本提供复核，不应在已有owner存活时另行启动。
