# Huber终止、检查点清理及B/D训练保护

2026-11-01，依据用户连续两条指令执行。不改变任何IL数学、精度、数据或评价协议。

## Huber已终止

- 根目录：`/mnt/project/qwen-world-postcot-20261025/fresh_loss_IL_20261029/HUBER`。
- 写入root/run绑定的`USER_TERMINATION.json`及全局实验生命周期hold：禁止新训练、新评测和自动恢复。
- 只停止经PID/start_ticks/命令核对的Huber CPU owner，对其八个训练rank发USR1边界退出请求。原生worker实际`EXITED/returncode=0`，最终有效update **48,288**。
- 所有Huber评测receipt同时检查大小写结果文件，无未结束的评测worker。其他实验进程未收到信号。
- 在owner和八个rank writer lock释放后，删除该组原生训练/资格检查点、`.saving`及未完成46k评测持有的原生硬链接。按文件块核算约 **479.05 GiB**，逻辑文件量514.38 GB；不是用共享磁盘全局空闲变化估算本任务回收量。
- 保留日志、配置、源码、逐场景请求与模拟轨迹、全部评分、报告和部署export。47个保留的metrics/SUMMARY/export metadata文件删除前后hash一致。原生小型inventory/seal已归档，LATEST/progress标记`checkpoint_deleted_by_user=true, recoverable=false`，不再宣称可恢复训练。

保留20个完整Navtest观察点，已有最高分为 **28k：87.1387804119**；44k为85.9641668776。停止不代表完成100k，也不补造46k或最终48,288分数。

实际执行与清理证据：根目录`termination_20261101/{NATIVE_STOP_REQUEST,CLEANUP_PLAN,CLEANUP_COMPLETE,RETAINED_NAVTEST_INDEX}.json`。只清理用户明确授权的本组检查点，其他实验和共享数据未改。

## B/D的后续资源优先级

用户要求后续任务分配尽可能保护B、D IL。实际训练仍是：

| 组 | 主机 | 原生槽 | 对新增任务的保护 |
|---|---|---|---|
| B | training-vlawm-tsinghua-zt | primary，8卡 | 新CPU guardian已获得其余3个namespace共24个空槽锁 |
| D | training-vla-zt | shared，8卡 | 保留原四namespace guardian，其余24个锁已持有 |

复用已验证D guardian的被动机制：已有cohost自然结束后才获取释放的锁，不驱逐、停止或迁移任何正常实验。CPU guardian不导入CUDA、不创建GPU上下文、不向模型发信号。当前原生invocation结束、主机权限撤销、明确RELEASE或guardian正常退出时释放锁。

B guardian：`/mnt/project/qwen-world-four-frameworks-20261029/resources/BD_IL_priority_20261101`；实际PID2947399/start1493812145，`B/STATE.json`为HELD24/24。绑定原生B invocation `execution_fast_fast_5a2d5b5_B_resume016948_attempt004`。D原guardian2628986未修改。

固定调度规则已写入`AGENTS.md`及`server_dispatch_policy.json`的`placement_preferences.qwen_world_IL_B_D`：

1. 新GPU任务、重CPU评分/缓存及IO任务优先使用其他已授权节点。
2. 不为资源重排暂停或迁移正常B/D训练。
3. 不因有空闲显存就继续给B/D增加cohost；先用其他节点。
4. 保留当前其他项目和真实作业，资源权限没有扩大或撤销。
5. 原生invocation变化时重新核对保护边界，不能保留无主的永久锁。

新通用入口`tools/il_report/hold_native_training_slots.py`已执行语法检查和实际24锁接管；未改现有D脚本或正式trainer。实际启动的是按脚本hash冻结的副本。原IL HTML owner保持，已请求其刷新终止状态与任务分配。
