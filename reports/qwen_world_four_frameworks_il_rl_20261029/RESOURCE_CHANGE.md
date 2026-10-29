# 2026-10-29 资源变更与无损迁移

按本轮最新用户指示，撤销 `vla-zt3`、`tsinghua-zt6`，恢复 `tsinghua-zt4`。全局调度策略及四架构、旧World、旧loss三个实际活跃任务grant已原子更新，旧授权收据保留。新的四架构代码和配方未改变。

当前可用主机：`vla-zt`、`tsinghua-zt`、`tsinghua-zt3`、`tsinghua-zt4`、`tsinghua-zt5`、`rl-zt2`。授权不等于空闲：派发仍检查实际GPU UUID、占用归属、显存、CPU/RAM及共享租约。

撤销后只进行了管理性退出核查，两个撤销节点均为本项目进程0、NVML GPU owner0，没有向它们新派实验或重试。四架构的有限profile只在 `rl-zt2` 和 `tsinghua-zt` 运行。

旧TIME_RESIDUAL在撤销时从完整update15449保存退出，已在仍授权的 `tsinghua-zt3` 与已知本项目V12 IL共享八卡恢复。八rank真实首步均15450、cursor494400；后续至15455无gap/duplicate/skip，首批32场景、学习率、已消费bank及原生Adam/RNG/数据状态一致。原科学source9aa、operations e125未改，未重启主owner。该迁移不改变新四架构正式学习任务的整机独立准入。

本次同时定位旧TIME/HUBER CPU控制器失败：arm目录缺少只读`candidates`资产别名，10k候选refresh已生成但registry发布失败。补齐指向原bank的别名后，CPU finalize attempt002真实退出0，旧失败RESULT、日志和状态仍保存，没有将原FAILED伪写为COMPLETE。

旧HUBER实际训练已在撤销指令之前因rank6 SIGKILL失败，最后完整checkpoint为14k；日志中的后661步尚未提交。它与本次撤销/别名修复无时间因果关系。该训练未被自动重启，失败证据与完整恢复点保留。

`tsinghua-zt4`现已可派发，但GPU3实际有约5483MiB used、无可归属NVML owner（driver reserved另为768MiB）；因此不能将其视为八卡空闲整机。没有对未知占用发信号或重置GPU。其余空闲卡仍可按现行准入用于后续独立评测。

完整管理及迁移证据位于实际资产目录 `resources/host_change_20261029T232711Z`，包括REQUEST、授权前后摘要、退出核查，以及`legacy_fresh_loss_recovery/TIME_RESIDUAL/TIME15449_MIGRATION_CONFIRMED.json`。资源变更不代表本轮IL100k、RL50k或Navtest已完成。
