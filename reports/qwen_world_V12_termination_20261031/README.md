# V12优化轨迹实验：用户终止

2026-10-31用户明确终止优化轨迹实验。对应独立Qwen-World V12 IL，已登记`USER_TERMINATION.json`，关闭训练、自动恢复、候选刷新和后续Navtest。没有改动其他实验的预算或配方。

核查时原worker已经FAILED/returncode1，最后八rank日志68,084步，最近完整native68k；原失败先于用户终止，不宣称本次实施了无损合作停止。清华zt3实际NVML无GPU owner，原训练进程和CPU owner均不存在；未向任何native模型发信号。所有已完成分数保留。

共享394TB文件系统已满（可用0），终止登记首次写入失败EDQUOT。按用户此前允许清理已终止/完成实验恢复状态的授权，只移除V12旧5k..65k checkpoint的104个优化器分片，原模型权重、原checkpoint seal和全部评测结果未改。单链接实际释放约209.58GiB；有其他硬链接的文件不计作已释放物理空间。67k/68k完整native保持，旧checkpoint加RECOVERY_RETIRED旁路标记，不能再当成完整恢复点。逐文件删除计划/大小/原hash与执行收据在运行目录保留。

周期评测跳过已终止run，并取消其未开始的pin；已完成评测不重复。当前V12无PENDING/RUNNING评测。报告显示“用户已终止”，健康检查禁止自动恢复。报告与cadence两个CPU owner原先已经退出，已采用生命周期检查后恢复，其他worker未发停止信号。37项针对性测试通过。

本次磁盘事件还暴露了其他IL任务的FAILED或状态待核实，见[disk incident snapshot](disk_incident_observation.json)。它们不属于本次用户终止，不可混记为主动取消或直接盲目续训。恢复需要核查各自合法native边界与失败原因；本报告没有宣称这些任务已恢复。损坏的实时RESULT元数据在健康检查中明确标为UNREADABLE_METADATA并报警，严格checkpoint/recovery解析保持拒绝，未把损坏当作成功或零分。

实际运行收据：`/mnt/project/qwen-world-postcot-20261025/optimized_labels_IL_20261028/user_termination_20261031/`；长期规则同步`/mnt/project/AGENTS.md`和dispatch policy的精确run lifecycle hold。当前实验状态见现有IL动态HTML，不恢复V12以填满空闲GPU。
