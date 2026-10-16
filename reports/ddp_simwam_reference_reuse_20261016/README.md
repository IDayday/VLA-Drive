# 固定 SFT reference 的四次 inner 复用

本项只减少同一保存链上的冻结reference重复前向，不改变CURRENT、源SDE、概率clamp、优势、anchor或Adam更新。原参考每份collection执行12次DiT前向；目标为3次，并在后续三个inner复用相同detach结果。

缓存仅保留一份条目，绑定输入tensor身份、version、layout/device/dtype、时间/窗口、采样配置和全部冻结reference参数/buffer版本；换链自动失效。CURRENT路径始终重算并保留autograd。缓存不进入state_dict、optimizer或恢复checkpoint；恢复后冷启动重算。

CPU边界检查已执行，见[cpu_qualification.json](cpu_qualification.json)。初次仅因输出父目录缺失而无法落盘，已补齐并以新attempt重验；[失败证据](cpu_harness_failure.json)保留，正式训练未受影响。

真实八卡验收已登记，尚未执行或尚无结果；不得仅以CPU通过声明无损提速。需比较零/非零LoRA、两种真实随机窗口、四次实际inner更新、下一collection原behavior重现以及冷缓存。固定96次资格optimizer更新，0新策略采样、0新评分；资格权重不用于任何正式初始化。[登记](registration.json)

现有LAST_ODE及SAFETY25正式训练源码快照保持不变。是否采用只由真实数值对照和耗时确定；正常运行不会为安装补丁中断。
