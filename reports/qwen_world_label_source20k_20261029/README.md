# 原GT／优化标签20k：同预算Navtest补测登记

优化标签20k已完整12146场景，canonical v1.1单候选clean-x K10 shift5 ODE、seed2101，PDMS84.12627395485266，invalid0且native/export PASS。本次只补原GT20k，原生完整checkpoint已硬链接锁定；不重复优化标签推理或评分。

两组初始化、模型与主配置、前20k场景顺序/LR/optimizer计数一致。前15k实际候选bank相同；15001起原GT使用自身10k刷新bank，优化标签使用自身10k刷新bank，八rank每个有5000次不同bank曝光。因此20k比较包含此下游辅助监督差异，不能说成纯主轨迹标签梯度的单因素对照。完整核验见matching_audit.json。

逻辑新增12146场景评价、1次native/export一致性。固定五主机，仅当前授权显存和共享lease准入；无新训练、seed或候选筛选。完整结束后按44个物理log配对bootstrap，报告优化−原GT及NC/DAC/TTC/EP等分项。

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u -m tools.qwen_world_postcot.latest_snapshot_owner --root /mnt/project/qwen-world-postcot-20261025/original_vs_optimized20k_NAVTEST_20261029T0740Z --report-dir reports/qwen_world_label_source20k_20261029
```

这是结果前登记，owner完成全部worker并验证后会自动替换结果并推送。Test历史已反复观察。原100k训练不修改、不重启。
