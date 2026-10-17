# 20k终点辅助任务：修正阶段判定，训练不变

2026-10-17 12:17 UTC，在B仍正常训练、尚未产生20k检查点时，发现新登记的两个终点辅助任务错误沿用了中期的`READY`判定。正式训练在20k使用`COMPLETE`，因此原辅助任务会拒绝正常终点；不是正式checkpoint、优化器或算法错误。

影响仅限`B_RL_20k_saved_batch_dispatch_20261017`中的保存批次诊断等待器和精确成本捕获等待器。它们尚未启动GPU分析、产生候选或更新模型。已核对PID、启动时间、完整命令及等待状态，停止这两个确切进程；原脚本和状态全部保留，并追加`SUPERSEDED.json`。其他训练、15k等待器、里程碑评价和最终Navtest控制器保持原进程。

替代目录为`/mnt/project/fm-transfer-20261016/resources/B_RL_20k_saved_batch_dispatch_20261017_v2`。最小修改只有：

- 发布目录改为独立v2，避免覆盖原尝试。
- 20k检查点要求`phase == COMPLETE`；updates20k、world8、next_inner0、文件hash和其余身份检查不放宽。

保存批次仍使用已验证的`fd440275edc3fdf40b2aaa46c38da1d038068902`诊断源码，只读取collection5000；训练源码仍为225b，数值推理源码仍为45d。没有修改现有科学配方、评分、轨迹或训练预算。

两个脚本各执行4项谓词检查：实际A-20k COMPLETE接受，实际B-10k READY拒绝，伪造20k READY拒绝，world错误拒绝；共8项通过，AST编译通过。检查使用实际原生seal和文件清单hash，不是通过修改正式检查点制造成功。新等待器身份存活且处于等待状态；旧两个进程已退出。完整diff、hash与PID回执见[校验记录](B_RL20k_auxiliary_phase_validation.json)。

真实20k诊断仍需等B到达终点后执行，不能把这8项阶段判定检查当作未来诊断已完成。此前SFT、RL5k/10k、A终点及已有评价结果均不受影响。
