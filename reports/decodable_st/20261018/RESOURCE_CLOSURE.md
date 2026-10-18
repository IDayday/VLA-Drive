# 成本、资产与资源收尾

按当时live dispatch policy使用rl-zt2、Tsinghua-zt2、Tsinghua-zt6，教师、学生、独立解码和验证并行。没有根据历史报告扩大授权，没有连接停用容器，没有停止无关任务或恢复压力脚本。资源共享许可只用于有明确用途的有限任务，没有为占满GPU而放行未验收长训。

本轮P0/P1预算登记上限16 GPU-hours、并行上限6卡；后补两rank恢复占用4卡、预留1.5 GPU-hours，仍在同一总预算中。所有实际run的阶段/源码/更新上限和日志在受控共享目录。最后三个使用过的远程主机进程检查均无本任务Python/torchrun；原始收尾记录为RESOURCE_CLOSURE.json。没有后台campaign自动续训队列。

## 实测而非推算的费用

RESULTS.json按原始receipt汇总**可核实GPU-hours下界约3.147**，包含教师前向、容量试验、tiny训练、解码、失败的prefix一致性探针、恢复对照和两步规划工程链。它不是完整账单：部分早期import/hash/load、下载、首次短段及导出/CPU渲染没有完整独立计时，不补造总成本。与共享的历史S0训练483.481 GPU-hours分开列。

主要单段：RS第203→1000段2062秒，之前2→203段889秒；Rfrozen1000步1671秒；real DDP连续/重复/2+2恢复合计.3125 GPU-hours。单段包含启动与真实执行，不能将除以updates的商当稳定step吞吐。RS早期两步的独立原RESULT已被当时后续分段收据替代，未伪造可验证原始文件，故不计入下界。

RS训练峰值约28.8–29.1GB allocated；Rfrozen约14.8GB。precision与恢复工程另列，不把BF16算子测试当主训练精度。吞吐未经200步多窗口性能实验，本轮没有对训练速度作稳定倍数承诺。

## 缓存和存储

- native ViGT训练64 + 独立开发16 full context约5GiB。全训练dense估算约6.2TiB，未构建。
- 64场景稀疏feature/精确native logits为602112 bytes/scene，预计全量约57GiB；构建9.96秒、顺序读取.092秒。FP16特征最大绝对误差.000945，logits保持FP32。tiny本体仍使用有界完整FP32 context。
- 12GB基础student导出只保留一份实际bytes；1000步表征导出hardlink复用，加strict delta。
- 每个恢复试验只保留最新rank状态，另保存2步边界的receipt/hash；没有为四更新测试生成多份大S0模型。
- 32个固定teacher/压缩面板及32个teacher/两个学生面板留在受控共享存储，公开只交索引、hash及不含原图的数值曲线。

原S0 checkpoint、旧teacher/data/cache、原失败报告和其他worktree没有被覆盖。今后恢复本轮只能使用相同源码/config/基础身份；跨配方继续属于新warm-start，不能冒充exact resume。
