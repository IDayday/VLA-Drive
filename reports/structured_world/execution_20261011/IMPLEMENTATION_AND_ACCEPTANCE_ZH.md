本分支保持 G 系列算法与曝光合同。G3 的规划局部采样同时改变道路、当前占用、未来语义和车身关系的查询分布；FGTR 使用共享 Bt latent，语义数值不作为硬约束输入规划。真实十步 q0 使用独立噪声与 eval/no_grad，修正梯度经过 H_A/Bt/FGTR，不经过采样链。旧 no-recompute 正式拒绝记录与 guard 保留。

基准训练源为 0c54f1c49e7ed2a76b22826b654d6c72625f5b6d，报告归档 267701f3b2c6cf36a97ff3a5e6bfb535e5aab232。实际执行源和控制器源分别登记，报告提交不替代训练身份。所有性能工作在新的隔离 worktree 中进行，没有热修改正式 G0/G1/G2/G3。

P0 engineering_fast 已实现实例内 Qwen 视觉长度 tuple 共用，真实当前 CPU processor/prompt/grid/mRoPE 准备及受检只读载荷，一次 pinned 标签传输与 microbatch view，20-update GPU 标量日志窗口，100-update FP32 master 证据，TokenFuser/MLN 公共计算复用，以及标签候选、车身模板和离散时间读取的复用。Qwen visual 输出和 DeepStack 仍在线计算；没有大规模视觉离线缓存。

P1 engineering_timevec 将无 dropout/BN 的时间 decoder 与融合合入 batch，训练期 FGTR 可变形 attention 的实际 dropout 0.1 保留逐时调用，只合并确定性坐标 MLP。precision_fast 单独为几何 R50/FPN/BEV 卷积开启 BF16 autocast，投影、DepthNet、grid、loss、主权重、optimizer 和梯度通信仍 FP32，canonical 推理明确绕过 BF16。后者属于 E3，不能与 P0 数学保持执行混称。

CPU 控制与初步完整 GPU 数值记录已保存；正式采纳尚待交替稳定窗口、native optimizer/moments/BN、短期继续训练、真实中断恢复和开发评价。完整链路的严格梯度测试在 A/A、A/B 都失败，不能将原阈值放宽后称通过。中期 P0 前向、q0/q_final、所有 loss、RNG 与 BN 都逐位一致，A/B 梯度误差处于 A/A 波动范围，但尚不证明长期更新一致。

最终性能测试登记为 NAVSIM G1/G3 × 早期/中期 × 两种执行 × 三交替窗口，每窗口 20 预热 + 200 稳定 optimizer updates。两种执行主计时统一为所有 rank 的 step20→220 墙钟最大值，日志摊销与窗口事件作补充记录；NCCL 其他 stream、父子范围、CPU synchronize 不作可相加节省量。窗口省去一次性的临时终点权重，始终保留只读 native origin，数值与恢复试验正常保存。任何已有 checkpoint、标签和几何准备均未覆盖或重做。

当前已执行筛选和最终验收结果将据真实记录追加；此文件不是验收或正式部署许可。
