# B 组：用户终止训练与 checkpoint 保留

2026-11-03 UTC。根据用户指令停止 B 组 IL，并只保留最新及最高分 checkpoint。范围仅为 B；D 与其他实验未停止、未迁移、未修改训练源码。

## 实际停止

- 原生训练在 **70,139** 次更新、cursor **2,244,448** 原子保存，正常退出，returncode=0；远端未发现剩余匹配 B 训练进程。
- 保留全量实验日志、训练与评测结果。关闭 B 自动恢复和新评测登记；原已运行 B70k 评测正常完成。
- 唯一 campaign 仍使用原科学快照 `5a2d5b50335e85fd84049ab1f1e5ebe34f537b4e`，控制器仅将架构列表从 A/B/D 改为 A/D；未向 GPU learner 发送信号。
- B 的八卡训练保护和专用 lease 已释放，供既有授权队列使用；D 的保护保持。核查时 D 八个 rank 均已推进到 56,366（状态快照）。

## 保留的权重

| 用途 | 更新 | 资产 |
|---|---:|---|
| 最新完整恢复 checkpoint | 70,139 | `runs/ARCH_B/IL_seed42/checkpoints/update070139` |
| 最高分完整恢复 checkpoint | 70,000 | `runs/ARCH_B/IL_seed42/checkpoints/update070000` |
| 最高分部署权重 | 70,000 | `periodic_NAVTEST_ARCH_B_070000_20261103T080320Z/B_IL_070000/export` |

资产根：`/mnt/project/qwen-world-four-frameworks-20261029`。保留 native checkpoint 的模型、optimizer、scheduler、RNG 和数据游标。70,139 未单独执行新评测，不套用 70k 分数。

B70k **canonical NAVSIM v1.1 PDMS=88.2487377802**；此前 B60k=88.1990759994。完整人口 12,146 scenes / 44 physical logs / invalid=0，单候选 B1、K10/shift5 ODE、seed2101。这里只陈述当前已测最高点，不将它解释为额外训练复现。

完整分项与全部历史曲线继续由[既有动态报告](https://github.com/IDayday/VLA-Drive/tree/reports/il-bd-navtest-live-20261103/reports/il_bd_navtest_updates_20261103)发布。原 HTML 单一 owner 已请求刷新，没有新建第二个发布器。

## 清理与验收

- 实际删除 **152** 个 B 中间 checkpoint / export payload，撤销 **24** 个对应完整标记；所有元数据先归档校验。
- 按最后一个 hardlink 被删除时的 `st_blocks × 512` 计，实际释放 **498,669,080,576 bytes = 464.4218 GiB**。不使用总逻辑文件大小或共享磁盘 df 差值冒充释放量。
- **34** 份既有 B 评分摘要在清理前后 SHA256 完全相同；评分缓存、请求/模拟轨迹、source、teacher/data cache 不在删除清单。
- 保留的两份 native inventory 均有 18 个文件且完整 seal 有效；最新 native、最高分 native 和最高分 export 的模型文件已执行实际 SHA256 核验。清理后两份 native 的 execution lineage 验证通过。
- 上一轮 11 月 2 日清理已删除旧 16,948 ancestor payload 及原 COMPLETE。本轮从原始审计归档恢复 **314 bytes 的原 COMPLETE 元数据**，SHA 与 migration receipt 完全相同，以保全最新 checkpoint 的迁移身份验证。旧 ancestor 仍明确标为 `LINEAGE_METADATA_ONLY / directly_resumable=false`，未恢复或伪造任何旧权重。

[最终校验](FINAL_VERIFICATION.json)、[实际删除记录](DELETION_RECEIPTS.csv)、[执行统计](EXECUTION_RESULT.json)、[停止核验](STOP_VERIFIED.json)。完整本地事务及原始元数据归档：`operations/B_user_termination_20261103/cleanup_apply_01`。

执行工具为本目录的 `retire_intermediate_payloads.py`，复用此前审计的 `20261102_selected_weight_cleanup/cleanup_lib.py` 原文件，SHA256=`27a15a56d1c37218bc968c719e4d06334f21cbb6530c189fa8a48c232849d3b7`。实际执行命令：

```bash
env PYTHONPATH=/mnt/project/qwen-world-il-research-20261030/maintenance/bd_completion_20261103/publisher_runtime_v1  /usr/bin/python3 /mnt/project/qwen-world-four-frameworks-20261029/operations/B_user_termination_20261103/retire_intermediate_payloads.py  --phase cleanup_apply_01 --apply
```

该 phase 是已完成的审计身份，禁止重复执行；不得重启用户已终止的 B 实验。
