G1/G3 native capture CPU证据复核（2026-10-11T12:21:32.903421+00:00）

G1 A/AA/B 各续训500步，累计1500步，各global endpoint25500。断连前归档确认15个native captures、每个17文件inventory。此前producer CAPTURE成功已记录；完整receipt与原始all500/all8 counts目前仅在无法访问的zt3本地。

Exposure不能将1500步直接当作48000 scenes。归档origin为epoch7/offset2775/exposure799944；实际冻结epoch_batches的CPU推导显示，每角色499个32-scene批次和1个24-scene尾批，预期新增15992，三角色47976，各endpoint815936（epoch8/offset100）。这仍是调度推导，实测全程totals待原始counts/metrics独立验证。

G1各PROCESS_EXIT wall/GPUh与native保存/加载/日志成本原记录当前无法读取，保留UNKNOWN。每角色5GPUh、总15GPUh仅为manifest计划reserve，不是实际cost；不得借用其他窗口/资格实验成本。

G3 A归档最后更新50001；8 ranks单更新counts/scene partition一致，新增32 scenes。首metric在启动114.636031113秒后到达，分配8卡观察到的成本下界为0.254746735807GPUh，完整process cost未知。metric的351.651709837384GPUh是继承累计值，不能记为本次cost。checkpoint仅writing_milestone_0050001，无COMPLETE/PROCESS_EXIT/TRAINING_COMPLETE；进程生死、OOM/restart或中断原因均不确定。

G1独立完整payload hash/数值比较0/5；跨A/AA/B完整counts/RNG/scheduler/groups/master/moments/BN精确验收PENDING。863.7GiB先前可用RAM不能用于OOM归因；部分G1 origin副本保留、未验证。

E3 endpoint500绑定仍为 1e3f0ddda2578ce748663bc2cc13b1525178e343248ce3e22f3456135c96d497；冻结be5、共享每组1bank/46GiB cap、100GiB物理余量，planned-only。原binding/阈值/strict/P0 guard均未修改。

本轮一次只读SSH重试仍connection refused。无GPU、恢复、native数值比较、新复制、信号或source编辑；恢复访问先告知parent，不自动开job。审阅只完成现有可访问证据的可提交整理，不宣称实验验收完成。

详细证据：SUBMITTABLE_CAPTURE_EXPOSURE_COST_REVIEW_V2.json，SHA 02398d13b1ccd0b5ecb33202a5876cce4a2d4aa5424d8dee81ef2d7d3bf5fa11。
