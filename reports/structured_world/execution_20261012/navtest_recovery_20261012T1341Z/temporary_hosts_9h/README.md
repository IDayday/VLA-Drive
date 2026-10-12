# 两台临时服务器的九小时授权

用户本次授权 `training-vlawm-zt` 和 `training-vla-zt3` 在九小时内使用，要求先记录原占卡脚本、用完恢复。按该消息首次登记时间计，窗口为 **2026-10-12 14:25:43 至 23:25:43 UTC**。这是本次 G 系列 Navtest 的临时资源授权，长期 allowlist、vlawm-zt 的原停用记录和训练合同均未改写。授权内容与截止时间保存在 [AUTHORIZATION.json](AUTHORIZATION.json)，共享调度登记使用独立的 `temporary_task_authorizations` 条目。

截至本目录 [STATUS.json](STATUS.json) 的记录时间，两台 SSH 均可连接，但 NVML compute-apps、pmon 和容器 `/proc` 中没有实际占卡进程；每卡分别约占用 12,221 MiB 和 36,979 MiB。扫描可见进程的 NVIDIA 文件描述符，只找到标准 `nvitop` 监控入口，没有发现可确认为占卡脚本的运行命令。磁盘上存在 `gpu_stress.py`，不等于该脚本正在运行。不能据此生成替代命令或把监控工具当作原脚本终止。

最初状态为等待实际占卡进程的启动入口。当时没有停止原进程，也没有启动新的 GPU worker。

用户随后明确改为从外部处理占卡脚本，并用其他监控程序负责恢复，**无需本代理恢复**。此最新指令见 [RESTORATION_HANDOFF.json](RESTORATION_HANDOFF.json)，优先于最初恢复安排。本代理始终没有终止或创建占卡脚本。

15:00 UTC实测两台机器16张GPU都已清空。15:13起，G3 90k全量评估在这16张独立物理GPU上运行，同时在原canonical环境用16个CPU进程评分。两台新主机均通过4个真实场景的q0/q_final/out_of_range逐位对照；FP32模型进程峰值reserved显存为13.078125 GiB。native参考在控制器交接前已生成，随后复用；每台新主机的实际GPU命令保存在qualification的launches记录中。

路由源码为`6fed371d77f3ddc6bcd574c7cca0fa98e61ad546`，模型/评分源码保持`0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`。每条临时主机GPU命令都在该主机重新计算绝对剩余时间，由GNU timeout在到期前60秒发TERM、必要时10秒后KILL；SSH启动延迟不会延长窗口。到期授权即失效，历史allowlist也不能绕过本次截止时间。九个CPU测试覆盖哈希、任务/主机隔离、九小时上限、过期、启动延迟和只结束自有命令；既有路由、OOM审阅、恢复测试亦通过。

此次只在所有旧评估子进程退出、旧参考完成后，停止空闲的自有评估控制器并切换到支持窗口的版本；没有停止训练或修改运行中的源码。共享调度中的临时独占记录在完整G3结果和worker退出证明产生后解除；到期恢复先前训练主机的路由资格，保留其他任务的新修改。占卡恢复由用户外部监控负责。

At15:22UTC, vla-zt3 disconnected; its eight original worker identities were verified gone after SSH was repaired, on the same physical GPU UUIDs. Separately, the IQE controller paused the other eight G3 evaluation workers and our controller atabout15:30 under its recorded user-authorized60k priority lease. Its guards restored those exact processes at15:46:31/32UTC after IQE inference completed. G-series formal training was never signalled by this task. Successful G3 prediction/score records remain unchanged. The unused vla-zt3 host reservation was released at15:47:52 after exit proof; pressure handling remains external. Full G3 completion still requires12146 validated records and explicit review of the original SSH failure.

Completed16:08UTC: both trajectory variants coverall12146 scenes with no score failure. Original SSH255 and partial worker reports remain preserved under the reviewed CPU-only closure. Claims were released and prior routing restored at16:09:30; the obsolete own claim watcher was stopped only after explicit verified cleanup. Registered cadence restarted at16:10:15 using unchanged frozen6fed371 dispatcher. The original nine-hour deadline is unchanged and enforced by its absolute remote timeout. No assistant pressure actions occurred.
