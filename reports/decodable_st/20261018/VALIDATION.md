# 测试范围与原失败

相关CPU回归共13项通过，不做全仓历史审计。原始日志哈希在证据清单；这些单测不是模型能力PASS。

| 方面 | 实际验证 | 限制 |
|---|---|---|
| 输入边界 | AllowedObservation拒绝targets/ID；原数据拆分；原生teacher前缀来源检查 | 未完成全开发人口逐字段破坏实验 |
| query/mask | 左右padding、group隔离、mRoPE数学、初始化不改RNG | 不宣称可复用跨请求KV缓存 |
| Qwen实际执行 | batch1/2/4、混合image grid、DeepStack/mRoPE，同批原prefix最大差0 | 跨batch原生舍入差异保留 |
| 教师完整权重 | ViGT559keys、SG1119keys严格加载；两个head分别加载 | 权重完整不等于教师质量 |
| 原生拆分 | 各64真实场景forward与hidden/context再decode差0 | SG采用披露的合法当前prompt，原能力分布尚未复现 |
| 原生梯度 | 固定decoder输入梯度、FP32/BF16有限，参数/buffers不改 | 没有PPU硬件测试 |
| SG布局 | 合成可定位轴张量、每次fresh shapes、不混当前/未来权重 | 真实horizon未核验 |
| 几何 | 变换代数、ego补偿、timestamp、ROI/unknown与射线语义 | 真实发射器原点与高度未核验 |
| loss | 有效scene归约、某rank全空、全局全空、累计分母；2rank Gloo对单batch梯度 | tiny完整模型只用已具备所有教师目标的64场景 |
| reader | 零residual等价、实际student memory梯度 | 正式P3尚未运行 |
| 实际DDP | 两卡真实图像/原DINO/native ViGT/AdamW；所有rank最终参数哈希一致 | 单步执行证据不能代替完整训练验收 |
| 恢复 | 单卡连续4步、2+2独立进程、另一次连续4步 | 只覆盖早期warmup；不保证长程bitwise |
| 导出 | 两个真实导出、独立进程阻断teacher/cache，固定噪声max差0 | 一张固定输入；可信本地Torch版本锁定 |

## 原始严格恢复结果保留

| 比较 | 离散状态错误 | 不逐位相同的浮点tensor | max abs | 最大归一化L2 | 最小cosine |
|---|---:|---:|---:|---:|---:|
| A/A连续重复 | 0 | 33 | 1.82074e-7 | 1.02587e-4 | .9999999947 |
| A/B连续／恢复 | 0 | 33 | 1.72993e-7 | 8.72734e-5 | .9999999962 |

step、sampler、scheduler、optimizer组、参数清单及各RNG状态精确；模型与moments严格逐位失败不删除。A/B在这次短测中没有扩大到超出A/A量级，但这个观察不构成长程误差上界。没有临时放宽原assert来写PASS。

另已完成真实两rank四步连续／重复／独立进程2+2恢复；单独结果写入DISTRIBUTED_RECOVERY.json，不覆盖上述旧测试。每个rank的step/RNG/sampler/optimizer/scheduler等离散状态精确；所有执行内rank参数哈希一致。A/A每rank225个浮点tensor不逐位相同，max abs1.04588e-6、最大归一化L2 .0006199；A/B为1.36509e-6和.0006688。A/B略大于这次A/A，保留INCONCLUSIVE，不用放宽阈值宣称exact resume。

该执行探针使用unit-scale feature loss及从第一步激活LoRA，不能把它当正式tiny配方或LoRA800步后的恢复证据。连续/重复使用同两卡；独立中断/恢复使用同主机另两卡，物理设备差异已记录。正式多阶段和长期漂移仍未验证。

## 真实失败与修复

- 最初整序列追加query改变SDPA形状，原prefix严格测试失败；改为同一次forward的native-prefix KV尾部调用，保持backend，真实同批差0。
- 原始PCD按文本读取失败；复用现有二进制LidarPointCloud读取，不把空读取当无障碍。
- 首次导出遇到Transformers训练hook closure；移除仅训练用hook后，独立导出通过。
- 初版来源锁未区分SGDrive源码/权重许可证；v2修正，原锁保留。
- 第一patch容量run用了当时未提交的本任务overlay；记录并用干净冻结SHA复跑，两个结果都保留。

正式完整训练、多阶段切换、完整开发/Navtest、三训练seed和真实teacher质量仍未完成；对应状态为BLOCKED/NOT_RUN/INCONCLUSIVE，而不是由上述工程测试替代。
