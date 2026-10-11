本分支保持 G 系列算法与曝光合同。G3 的规划局部采样同时改变道路、当前占用、未来语义和车身关系的查询分布；FGTR 使用共享 Bt latent，语义数值不作为硬约束输入规划。真实十步 q0 使用独立噪声与 eval/no_grad，修正梯度经过 H_A/Bt/FGTR，不经过采样链。旧 no-recompute 正式拒绝记录与 guard 保留。

基准训练源为 0c54f1c49e7ed2a76b22826b654d6c72625f5b6d，报告归档 267701f3b2c6cf36a97ff3a5e6bfb535e5aab232。实际执行源和控制器源分别登记，报告提交不替代训练身份。所有性能工作在新的隔离 worktree 中进行，没有热修改正式 G0/G1/G2/G3。

P0 engineering_fast 已实现实例内 Qwen 视觉长度 tuple 共用，真实当前 CPU processor/prompt/grid/mRoPE 准备及受检只读载荷，一次 pinned 标签传输与 microbatch view，20-update GPU 标量日志窗口，100-update FP32 master 证据，TokenFuser/MLN 公共计算复用，以及标签候选、车身模板和离散时间读取的复用。Qwen visual 输出和 DeepStack 仍在线计算；没有大规模视觉离线缓存。

P1 engineering_timevec 将无 dropout/BN 的时间 decoder 与融合合入 batch，训练期 FGTR 可变形 attention 的实际 dropout 0.1 保留逐时调用，只合并确定性坐标 MLP。precision_fast 单独为几何 R50/FPN/BEV 卷积开启 BF16 autocast，投影、DepthNet、grid、loss、主权重、optimizer 和梯度通信仍 FP32，canonical 推理明确绕过 BF16。后者属于 E3，不能与 P0 数学保持执行混称。

CPU 控制与初步完整 GPU 数值记录已保存；正式采纳尚待交替稳定窗口、native optimizer/moments/BN、短期继续训练、真实中断恢复和开发评价。完整链路的严格梯度测试在 A/A、A/B 都失败，不能将原阈值放宽后称通过。中期 P0 前向、q0/q_final、所有 loss、RNG 与 BN 都逐位一致，A/B 梯度误差处于 A/A 波动范围，但尚不证明长期更新一致。

最终性能测试登记为 NAVSIM G1/G3 × 早期/中期 × 两种执行 × 三交替窗口，每窗口 20 预热 + 200 稳定 optimizer updates。两种执行主计时统一为所有 rank 的 step20→220 墙钟最大值，日志摊销与窗口事件作补充记录；NCCL 其他 stream、父子范围、CPU synchronize 不作可相加节省量。窗口省去一次性的临时终点权重，始终保留只读 native origin，数值与恢复试验正常保存。任何已有 checkpoint、标签和几何准备均未覆盖或重做。

当前已执行筛选和最终验收结果将据真实记录追加；此文件不是验收或正式部署许可。

截至 2026-10-11 11:36 UTC，最终性能矩阵已完成 13/24 个窗口。G1 早期和中期分别完成每模式三个交替窗口、每窗口 200 个稳定更新；早期吞吐提高 11.98%，中期提高 13.16%，固定曝光 GPU 成本分别降低 10.70% 和 11.63%。G1 中期共同边界墙钟为 IO 3.2292 秒/更新、P0 2.8537 秒/更新。所有 rank 的峰值 allocated 从约 31.90 GiB 降到 30.58 GiB。完整结果及实际进程成本见 `performance/INTERIM_FINAL_WINDOWS_V4.json`；该记录明确是中途快照，不能替代尚未结束的 G3 矩阵或数值验收。

时间向量化没有进入最终性能候选：真实 G1 中期同输入验证中，Bt 的 842117/2097152000 个元素未满足预先冻结的 1e-6 absolute / 1e-4 relative 容差。虽然 q_final 最大位置差仅约 3.8 微米，仍保留原失败结果与阈值，不将其标为通过。当前最终候选只有 P0 `engineering_fast`。局部 BF16 的 60-update 筛选约 2.608 秒/更新，属于 E3 数值变化；500-update 配对、恢复和开发结果尚未完成，不能正式承接。

G1 中期 A/AA 各 500-update 连续训练已完成，B 正在执行；五个观察位置均使用原生 ZeRO checkpoint。G3 中期排在同一空闲授权 8 卡主机，四组早期及中断恢复另在性能矩阵完成后执行。没有切换或中止正式 G0/G1/G2/G3。

2026-10-11 12:23 UTC 更新：G1 A/AA/B 各 500 更新已经完成，共 15 个 native 观察点。zt3 在 11:42 UTC 后 SSH 拒绝连接；G3 A 仅确认完成第一次更新，50001 的写入尚无 COMPLETE，进程状态和完整成本未知。G1 全量检查点比较仍为 0/5，不将 metadata、调度推导或预算当成数值/曝光/成本实测。另一台 zt4 的 24-window 性能矩阵仍在执行，正式 G1/G3 在原 0c54f1c / io_preserving_v1 源码下正常推进，未热修改。

P0 单次前向观察器单独冻结于 6d16abf，模型/训练源仍为 be5e722；12 个新观察槽（四个 origin 各三个独立 A/AA/B 重复）尚未执行。原严格结果和 exit=1 保留，完整失败观察可记为完成但不可改为 strict pass 或正式验收。恢复 V2 只澄清实际 +16 加载与 +32 比较终点，沿用旧数值/dev 阈值；CPU 34 测试和 controls 路由 7 测试通过，实际恢复仍待运行。zt4 环境追加登记已只读核验的 flash_attn 2.7.4.post1，不改旧 qualification、依赖、backend 或运行环境。
