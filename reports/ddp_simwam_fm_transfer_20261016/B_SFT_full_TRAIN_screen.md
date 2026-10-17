# B100k SFT：完整TRAIN筛选、固定困难人口与成本

2026-10-17完成B_SOURCE_ACTION_DIT自身100k SFT的完整TRAIN单候选筛选。人口为101,592场景、1,176 logs，使用seed2101、原生均匀十步ODE、canonical NAVSIM v1.1。冻结condition来自已与原观测入口核对的缓存；这是每场景一个固定样本，不是G8组统计、best-of-N或Navtest。

**完整TRAIN PDMS为94.233394，按固定score<0.9得到21,836个困难场景、1,056 logs。** 24个worker全部完成并经PID/start-time核实退出；101,592份评分与轨迹payload的身份/hash逐条核验通过。没有缺失场景、评分异常填零、复用或重派候选。B自己的原始观测完整DEV基线另见[100k终点报告](final_B_SFT_update100000.md)。

## 单候选人口与安全分项

| 事件 | 场景数 | 占完整TRAIN |
|---|---:|---:|
| PDMS严格等于1 | 38,673 | 38.0670% |
| 0.9≤PDMS<1 | 41,083 | 40.4392% |
| 0<PDMS<0.9 | 21,284 | 20.9505% |
| PDMS严格等于0 | 552 | 0.5433% |
| NC/DAC/TTC全部完全通过 | 100,731 | 99.1525% |
| NC未完全通过 | 17 | 0.0167% |
| DAC未完全通过 | 543 | 0.5345% |
| TTC未完全通过 | 325 | 0.3199% |

安全事件可重叠，NC=0.5不视为完全通过，三个违规计数不能相加为联合违规数。此处单seed通过不代表所有噪声均安全。也不能将38.0670%的单轨迹满分率称为G8满分组比例。

| 官方指标，0–100 | 完整TRAIN | 固定困难清单 |
|---|---:|---:|
| PDMS | 94.233394 | 83.509845 |
| NC | 99.987204 | 99.940465 |
| DAC | 99.465509 | 97.513281 |
| TTC | 99.680093 | 98.511632 |
| EP | 87.279445 | 65.631165 |
| Comfort | 99.871053 | 99.400073 |
| DDC | 98.608158 | 98.589485 |

## 固定困难清单

困难清单占完整TRAIN的21.4938%。其中20,975个场景的NC/DAC/TTC全部通过，占困难清单**96.0570%**；861个有安全分项未完全通过。类型如下：

| 困难类型 | 场景数 |
|---|---:|
| 安全分项通过、质量/进度较低 | 20,975 |
| 仅TTC未通过 | 301 |
| 仅DAC未通过 | 535 |
| NC与TTC未通过 | 16 |
| DAC与TTC未通过 | 8 |
| 仅NC未通过 | 1 |

本清单仍主要偏向安全但进度/质量不足的人口。不能由此预设新ActionDiT的RL已经解决安全问题，也不要求每个fresh G8组都存在非零优势。正式训练中的等分、正优势类型及安全候选比例必须由真实当前策略采样统计。

清单来自B自己的100k终点，不能借用A的清单。A有21,367个困难场景，两者交集20,690，A独有677，B独有1,146。两组RL的科学配方相同，但自己的困难人口、SFT初始化与监督预算不同，因此整体路线对比不是仅改变动作头结构的纯因果消融。完整TRAIN点估计中，B与A的PDMS接近；B零分/DAC/TTC未通过数较少，但NC未通过17个，略多于A的14个，不能称所有安全分项均改善。

阈值、成员、scene来源和SFT身份已固定，不随当前RL分数改变。全部来自RL_TRAIN，无DEV/Navtest。私有清单保留逐scene的token、score及payload身份，仓库只提交聚合结果和hash。

## 实际成本

- 24个全局分片，每片4,233场景。逻辑评分101,592次，实际新推理101,592次，复用0次，未发布评分请求0次，24份调用账本全部关闭。
- 端到端筛选3,324.331秒（55.406分钟）；累计worker wall 76,518.545秒（21.255 worker-hours）。后者不是独占GPU compute，也不能与并行job wall混用。
- 累计模型加载438.225秒，生成66,511.017秒，评分等待21.853秒，worker评分2,995.106秒。生成计时含condition读取、十步ODE及CUDA同步，不是纯GPU kernel时间；异步评分时间不应再次加到端到端wall上。
- 全量payload核验246.797秒，8个CPU线程；随后只读汇总154.459秒。两者没有新增模型推理或评分。
- 运行主机为tsinghua-zt6、zt4和zt；后两者与其他真实任务共享GPU。本次实际检查没有可释放的占卡脚本，没有向这些真实任务发送信号。A筛选约28.94分钟，资源占用与执行时段不同，不能把这次耗时比单独归因为模型结构。

B的100k SFT训练成本另见[原生终点与预算](B_SFT100k_completion_and_handoff.md)。筛选历史成本保留，不能因为后续读取已完成清单而把它写成零成本。

## 身份、执行与后续

- SFT语义身份：`63b2da609f3ab3b225f056a5399050fc217989d25b937876a5a5051a29575af7`。
- SFT原生模型SHA256：`4e71b1fcf963d38456174de1f8d17762a9a4ccf1597c22b7cec262026d9201b7`。
- 困难清单内容fingerprint：`681efb6c5958a7b45c80d48cb8ebf22d1ec00e0c06abe8010b43982407adc4e3`。
- 困难清单文件SHA256：`cc6017f0740c080dfae73f36ddc24bb657ae2996a8440b7c035ad4821612ec4a`。
- 筛选及清单核验数值源码：`225b9af660cb0210208e2c46fc4216d0f151275b`；只读汇总执行源码：`d8e4c6c7cdf11eed2a1f1ef8c3b84230b7e46999`。修复后的CPU汇总实际退出0，没有改变评分或成员规则。

[完整统计与成本](B_SFT_full_TRAIN_screen.json)、[24个进程退出与物理账本](B_SFT_full_TRAIN_screen_execution.json)、[清单/汇总/身份核对](B_SFT_full_TRAIN_screen_validation.json)。发布核对比较了独立执行账本与官方汇总、全部类别分母、清单成员数和hash；未重新评分。

以下入口由现有控制器实际执行，分别在上述锁定源码目录运行。现有结果不可覆盖；需要复核时使用独立输出身份，不重复启动原worker。

```bash
python -m scripts.analysis.fm_transfer.build_hard_manifest \
  --screen /mnt/project/fm-transfer-20261016/screens/B_SOURCE_ACTION_DIT_SFT_update100000 \
  --output /mnt/project/fm-transfer-20261016/hard_manifests/B_SOURCE_ACTION_DIT_SFT_update100000.json

python -m scripts.analysis.fm_transfer.analyze_screen \
  --screen /mnt/project/fm-transfer-20261016/screens/B_SOURCE_ACTION_DIT_SFT_update100000 \
  --hard-manifest /mnt/project/fm-transfer-20261016/hard_manifests/B_SOURCE_ACTION_DIT_SFT_update100000.json \
  --output /mnt/project/fm-transfer-20261016/analysis/B_SOURCE_ACTION_DIT_SFT_update100000_full_TRAIN_screen.json
```

此报告发布时，B自身端点的真实八卡RL验收已启动，正式20k RL仍等待验收。不会使用验收后的adapter作为正式初始化；继续从相同100k SFT新建R_SOURCE。完整TRAIN与DEV均不能替代最终锁定Navtest成绩。
