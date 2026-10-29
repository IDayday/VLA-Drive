# 原轨迹与优化轨迹标签：Qwen-World 同预算IL对照登记

当前还缺原轨迹10k/15k的完整Navtest，不能用原轨迹70k88.792910与优化标签15k81.359718推断标签优劣。本次只补原轨迹10k/15k，优化标签10k79.904274和15k81.359718已有全量结果直接复用。原轨迹13k不在原生保留目录，优化标签13k66.369502未以异步数检查点冒充匹配控制。

两组均为官方通用Qwen＋随机World memory/actor初始化、seed42、global32、100k总预算、原L1及相同四辅助任务。初始化/模型/数值/optimizer参数组/teacher/ZeRO合同一致；resolved config仅各自candidate registry路径不同。实际八rank的1..15000更新逐点核对场景ID、LR、cursor、optimizer计数、跳步、辅助曝光和bank一致，sigma_mean差0；没有发布原始场景ID。仅根据sigma统计一致不宣称完整噪声张量逐值验收。

源码并非相同SHA：增加可选标签overlay、坐标诊断、资源身份和显式RL父模型分支等。已核对两组IL所走的有效L1/模型/优化器路径，未启用坐标损失分支。已有真实八卡CONTROL数学/执行等价收据保留。初始化合同相同，不额外声称保存过逐字节零步全模型对照。

前15k实际使用同一个初始候选bank。后续各自异步候选刷新仍需按日志核对，不能将此匹配性外推到完整100k。优化标签98分为历史标签评分，不能代替模型部署分数。

评测锁定完整12146 Navtest，canonical NAVSIM v1.1，seed2101，单候选clean-x K10 shift5 ODE。两个新原始IL点新增24192条逻辑场景评价、2次native/export一致性；优化标签既有结果复用并保留历史费用。固定五主机，实时显存/UUID/可见owner/共享lease准入；临时主机撤权保持。Test历史已被观察。

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u -m tools.qwen_world_postcot.latest_snapshot_owner --root /mnt/project/qwen-world-postcot-20261025/original_vs_optimized_labels_NAVTEST_20261029T0500Z --report-dir reports/qwen_world_original_vs_optimized_labels_20261029
```

单一owner；完成后自动生成优化−原轨迹的PDMS/原生分项、新增失败/恢复、G+/G−、物理log配对区间，并在worker退出后推送。此登记未启动新训练或改变旧实验。
