# A组SFT完整终点与RL前置筛选

2026-10-16 19:21 UTC左右，A_DDP_TIME完成登记的20,000次监督optimizer更新。八个rank均写出`COMPLETE`结果，原生终点为`update020000`、phase READY、cursor 640000。模型、Adam、scheduler、八rank RNG和数据进度均按既有原子保存协议封存；没有将中期最高DEV分数替代终点。

| 核算项 | 实际结果 |
|---|---:|
| 逻辑optimizer更新 | 20,000 |
| 逻辑场景曝光 | 640,000 |
| 实际访问的不同TRAIN场景 | 101,592 / 101,592 |
| DEV/Navtest更新场景 | 0 |
| 每场景FM独立噪声重复 | 8 |
| FM插值训练样本数 | 5,120,000 |
| SFT阶段RL候选/奖励调用 | 0 |
| rank0记录训练wall | 6081.059秒，约101.35分钟 |
| 八rank累计worker wall | 48648.046秒，约13.51 worker小时 |
| 派发至最后训练边界，近似 | 6104.693秒 |

八rank更新序列均为连续1–20000，无重叠或遗漏；每步cursor、四个本rank场景及TRAIN成员资格均经过实际日志核查。FM重复没有算成独立场景，也没有把八rank并行更新算成160,000次逻辑optimizer更新。

rank0累计数据取得1031.545秒，forward/backward/optimizer更新4792.631秒；其余256.883秒含保存、hash、同步和日志开销，不能全部命名为保存时间。逐次保存耗时被旧progress字段覆盖，未单独恢复精确总值；模型加载与rendezvous/setup也未完全分离。worker wall不是独占GPU计算时间。完整来源与限制见[A组SFT成本账本](A_SFT_complete_budget.json)。

完整原生动作头SHA256：`ac52e9346080ce0e6bf27441935ac9aa6acbff239330b316bee36b560486579a`。终点files.json SHA256：`25abe375f069f60cbdfd8ea0f94d7acef1cff85e962402ce2bf7ae8ff6c71fbf`。原入口导出权重SHA256：`8da410eb75c1560c0c935214b946a45a65e16cd744b674d6df2d78d65e0d9005`。终点DEV/固定TRAIN双网格评价已进入原评价队列，本记录发布时尚无完整20k规划结果。

## 已启动的下一阶段

24个全量TRAIN筛选worker已实际派发并核对PID、启动时间、命令及工作目录：

| 主机 | 全局分片 | 实际GPU数 |
|---|---|---:|
| vla-zt | 0–7 | 8 |
| rl-zt2 | 8–15 | 8 |
| tsinghua-zt4 | 16–23 | 8 |

各分片固定4233场景，总计101592；使用自己的20k SFT、native单候选seed2101、原始动作解码和canonical v1.1独立评分。每次派发均重新核对当时授权和显存；全部保留一场景batch1。数值worker固定为`225b9af`，分片调度器为`b80cdad`，不存在中途热改actor。19:23 UTC检查时24个worker均处于RUNNING，已记录2560个完成场景；这只是进度快照，不是完整候选清单。

全量筛选输出：`/mnt/project/fm-transfer-20261016/screens/A_DDP_TIME_SFT_update020000`。实际派发回执：同目录`dispatches/`及`/mnt/project/fm-transfer-20261016/resources/A_screen_partition_launch/`。

只有全量筛选及payload校验结束，才生成固定canonical<0.9困难清单。随后执行自己的完整SFT真实八卡四次inner/中断恢复/合并/原部署验收，再从20k SFT重新初始化正式LoRA训练。资格权重不作为正式初始化。当前不能把SFT完成或筛选启动写成RL已完成。

B组100k SFT继续运行，已完成的10k结果见[独立ActionDiT中期报告](interim_B_SFT_update010000.md)。旧追加实验保持暂停，未恢复新方法矩阵；本轮Navtest尚未启动。
