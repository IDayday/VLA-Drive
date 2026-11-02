# 实际运维脚本归档

这些文件是实际执行源码的逐字节归档，用于复核，包含本轮日期和 NAS 路径。当前运行的 CPU owners 与 guardian 仍固定各自冻结源码；不要以归档替换正在运行的进程或重放已结束的资源窗口。

`archive_s7_permanent_capacity_wait.py` 只在旧 I 自然完成10k且实际资源释放后，为 S7 追加长期槽位；它不派发、不发信号，仍服从 NativeOwner 的显存、最多两已知 native 和优先级规则。

`archive_s7_native_qualification_driver.py` 是已完成的有限三阶段工程资格，所有产物独立于正式 fresh-IL70 训练。

`archive_close_window_20261103.py` 仅负责本次六机实际硬截止时的调度禁入，不终止或修改其它任务授权。
