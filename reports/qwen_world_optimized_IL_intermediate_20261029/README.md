# 优化轨迹标签IL：13k/15k Navtest登记

已锁定同一个完整100k预算优化标签IL的13k、15k原生检查点，全部18个payload库存以硬链接保留；导出器仍验证完整内容hash。10k已有完整Navtest79.90427443202655，直接复用。原controller登记的20k/40k/60k/80k/100k评测保留，本次不重复。

协议：完整12146场景、canonical NAVSIM v1.1、seed2101、单候选clean-x K10 shift5 ODE。新点分别与同一优化标签IL10k及前一观察点配对。不与旧Qwen-World40k混作同预算标签对照；不使用历史训练标签98分作为部署成绩。Test历史已反复观察。

最多2个模型并行；仅固定五主机，按实时UUID/显存/已识别进程/共享lease准入。用户已撤回的临时主机禁止派发。源码只修改CPU评测恢复和派生分析，不改模型/采样/评分。逻辑新增24192场景和2次native/export一致性检查；实际共享GPU worker费用保留在资产根。

实际启动入口：

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u -m tools.qwen_world_postcot.latest_snapshot_owner --root /mnt/project/qwen-world-postcot-20261025/optimized_labels_intermediate_NAVTEST_20261029T0330Z --report-dir reports/qwen_world_optimized_IL_intermediate_20261029
```

本页为结果前登记；完整评测且worker退出后，owner自动替换为正式结果并推送同分支。
