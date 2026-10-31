# 磁盘清理后：七组真实训练恢复

用户要求恢复异常中断任务。恢复范围仅仍获授权的A/B/C/D、TIME_RESIDUAL、HUBER和PPO_SAFETY_EP；V12、RAW已完成训练、PHYSICAL/DELTA筛除组及关闭的RL变体均未恢复。

## 已执行结果

下表是2026-10-31恢复确认时的八rank共同更新下界，不是启动进程数。全部任务已有至少4次新有效optimizer update。

|任务|实际节点|完整恢复边界|已确认共同更新|
|---|---|---:|---:|
|FOUR/A|vlawm-tsinghua-zt6|19000|19058|
|FOUR/B|vlawm-tsinghua-zt|15000|15040|
|FOUR/C|vlawm-tsinghua-zt4|17000|17049|
|FOUR/D|vla-zt|11000|11035|
|TIME_RESIDUAL|vla-zt3|46000|46138|
|HUBER|vlawm-tsinghua-zt2|26000|26137|
|SAFETY_EP|vlawm-tsinghua-zt3|28000|28111|

六组IL继续原100k预算；SafetyEP继续原50k总预算，当前执行段仍至35k。没有从头训练、二次warmup、换LR、重置Adam/参考策略或改变评分。

## 原生状态与日志

- 原科学源码保持不变：四架构formal_fdaff80与ops_fcd43eb；两条损失IL的既有loss-from-init源码；SafetyEP原v7_memorylr_121aa27aa99c。
- 验证旧worker/torchrun/rank退出和rank锁，核对COMPLETE、文件清单、八rank状态和源hash，再由原生restore严格检查实际模型/优化器payload、scheduler及RNG。
- 最近完整checkpoint后的旧日志移入各任务`aborted_journals`，保留失败/物理成本。由相同原生状态重新计算未提交尾部，逻辑更新不重复计数。已移除的832条派生报告曲线记录另存，未删除原始失败证据。
- 全部七组八个rank的首个重算更新均衔接恢复边界；样本/scene、cursor和实际LR中可对照字段与原中断前记录匹配，optimizer计数合法、跳步0。见[真实学习证据](real_learning_confirmed.json)。
- 对盘满导致的末行截断，只允许在已完整验证的提交前缀之后保存原始损坏尾部；提交前缀损坏仍拒绝恢复。

## 资源协调与控制器

首次恢复检查遇到新Navtest占卡，三个CPU恢复helper在模型启动前拒绝，0新增训练/评分；没有驱逐评测或未知进程。第二次只在可见、已登记评测及足够显存下采用成熟shared admission。原生actor、batch/chunk和精度不变。A从不可安全辨识占用的rl-zt2转至获授权清华zt6；其他实际位置以表格和工作收据为准。

D继续vla-zt八卡。CPU guardian为本次shared-slot native invocation保留primary/research槽，现有评测结束后维持独立八卡；不向任何模型发信号。所有主控制器已接替活跃invocation，原RL固定节点评测和独立每2k cadence也已恢复，不复活关闭变体。

## 执行入口与验证

实际执行的入口：

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u -m tools.il_report.disk_recovery --mode four
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u -m tools.il_report.disk_recovery --mode fresh --arm TIME_RESIDUAL
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u -m tools.il_report.disk_recovery --mode fresh --arm HUBER
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u -m tools.il_report.recover_safety_disk
/usr/bin/python3 -m unittest discover -s tests/il_report
```

42项针对性测试通过；实际验收另包含七组全部八rank新更新、原生状态与重算batch验证。入口拒绝重复派发，当前仍在训练，不要再次启动同名helper。

工作收据根：`/mnt/project/qwen-world-il-research-20261030/maintenance/disk_recovery_20261031/`。每组PREPARED、NEW_INVOCATION、RESTORED_OWNER、REAL_LEARNING_CONFIRMED及汇总RESTORED_ALL_SEVEN均真实存在。初次容量拒绝、损坏metadata、原始失败、回滚日志和CPU接管历史保留。新训练仍在进行，不宣称完整100k/50k已经结束；此次恢复不新增PDMS方法结论。
