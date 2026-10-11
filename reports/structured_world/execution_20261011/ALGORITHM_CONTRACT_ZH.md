G3 沿用 planning-local sampling for structured scene supervision。sample_queries 同时为道路、当前占用、未来语义与车身关系监督产生查询，因此 G3−G2 改变的是整体结构化场景监督分布。既有组名和结果数字保持不变。

FGTR 读取逐时间 Bt latent。scene_semantics 是训练与诊断读出，在部署删除。监督通过共享空间表示塑造 Bt，再由 FGTR 参与修正；预测道路/占用数值并没有以硬约束直接输入执行规划。

真实原 10 步 q0 使用独立 proposal RNG、no_grad 和 eval/dropout 状态，并恢复原状态。位置和条件 detach；refine loss 经过 H_A、Bt、FGTR 回传，不经过 q0 采样链。这个分离梯度边界是本轮固定方法定义。

工程分支不分开 current/future 查询，不改变 K、采样比例、时刻数、网格、分辨率、相机、FM repeat、参数组、microbatch 或 AdamW 衰减语义。不解除 loss_preserving_v1 正式 guard。

同容量当前修正器、语言/action 表示依赖、未来语义与当前语义各自局部采样的影响仍属于研究控制；本次工程优化不替这些实验作因果结论。
