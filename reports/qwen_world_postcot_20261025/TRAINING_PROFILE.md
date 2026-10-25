# 完整训练测速

当前有效native基线已在独立A800八卡完成32次完整更新，预热4次、稳态28次。
加载原IL78k的真实全模型和Adam状态，原100k学习率时间轴；输出隔离，不改原run。
使用本轮固定合法TRAIN的前1024个实际sampler场景，micro1×accum4、global32，
确定性/TF32等设置继承native。无profiler干扰该吞吐窗口。

| 路径 | mean s/update | p50 | p95 | scenes/s | peak allocated |
|---|---:|---:|---:|---:|---:|
| native实际78k权重+Adam | 4.44600 | 4.29374 | 4.47355 | 7.19748 | 75,023,986,176 B |
| 新完整模型joint、正确v3教师、正式完整bank | 1.94502 | 1.93309 | 2.01150 | 16.45228 | 22,888,364,032 B |

数据：`/mnt/project/qwen-world-postcot-20261025/profiles/native32/SUMMARY.json`。
checkpoint不包含在上述正常update计时；本对照没有重测native checkpoint摊销。
旧正式节点约13.7秒的历史退速不能由此单次隔离测速归因给某个因素，也不把历史
数字当本次新模型实测。新模型是完整新架构和训练定义，倍率不能叫作纯执行无损收益。

新模型也完成32次完整joint更新、预热4、稳态28；micro4×8、accum1。
teacher有效1017/1024场景，另外7个真实timestamp gap正确mask；DINO1024全有效，
post6144个真实有效分项。五loss与总loss均实测FP32，参数/梯度/master/AdamFP32，
20层投影K/V为BF16。reserved峰值29,234,298,880B，约27.23GiB。
相对native为2.286倍整体吞吐对比，包含不同actor/memory/监督及ZeRO2执行变化，
不是同模型纯调度无损收益。数据：`audits/FINAL_V3_TRAINING_PROFILE.json`。

曾有v2仅36.8%真实future有效的诊断，明确作废其完整监督资格；最终表只用修正
两个互补挂载后的v3。较早micro2两步2.66/2.71秒也不作为正式速度结论。
真实ZeRO2保存耗时约61.75秒，单列、不混入正常update；按每1000步摊销约0.06175
秒/步，仅为该次保存的条件外推。正常更新100k约54.03小时，另加保存、DEV、
共享资源竞争及迁移；这是估计，最终总成本应以完整journal计。

独立warm4+2step trace已实际导出，路径
`profiles/full_joint_micro4_v3_trace2/trace_rank0.json`（其余rank同目录，实际目录
见V3_PROFILE_PIPELINE.json）。包含CPU图像decode、教师缓存、候选读取、processor，
以及模型区域/反向/optimizer/NCCL；不把多stream inclusive时间直接相加。

该两步trace中，Qwen forward/backward设备工作量约188.49/617.78ms/步；
NCCL活动536.14ms，其中287.75ms与计算重叠，248.39ms没有同时计算。后一项
是时间线描述，不等于移除通信后能节省同样wall time。CPU图像decode164.11ms、
processor145.91ms、H2D区域93.38ms均为各自inclusive范围，不能与CUDA工作量
直接累加。主actor和辅助actor设备前向分别10.67/6.46ms，不能据此代替完整update。
完整分解：`audits/V3_TRACE_RANK0_BREAKDOWN.json`；本轮不再扩展性能候选。

## 单轨部署

同一A800、batch1、单候选K10shift5，48个配对真实场景、预热8、计时40，交替执行
native/新模型以控制漂移。包含真实图像decode、processor、H2D和CUDA同步。

| 路径 | mean / p50 / p95，秒 | 单模型独立峰值allocated |
|---|---|---:|
| native78k | 0.87418 / 0.87475 / 0.88923 | 14,099,073,536 B |
| 完整模型部署（资格权重） | 0.56746 / 0.56644 / 0.57855 | 10,757,213,184 B |

计算forward均值分别0.80712/0.50029秒，其余约0.067秒为当前图像/processor/H2D。
新模型资格权重仅用于架构时延，不代表规划效果；部署不含任何教师或辅助readout。
两模型同时resident计时，独立峰值另在加载第二模型前/删除第一模型后测，未将
双模型总显存误当单模型峰值。完整记录：`audits/DEPLOYMENT_LATENCY.json`。

## 正式运行：完整接口阶段与联合阶段

验收时间2026-10-25T18:07:13.440319+00:00，同配方已在永久ts5八卡真实跑过1000接口更新及123联合更新。
接口稳态996次：mean/p50/p95=1.66662/1.65917/1.77390秒；联合稳态119次：
**2.60730/2.47468/2.64285秒**，12.27323 scenes/s，peak allocated22,860,822,528B，
reserved28,118,614,016B。两阶段分别预热4次；不要将冻结Qwen的接口速度当联合速度。

正式联合窗口中包含真实7–8秒尾部更新，没有为加速结论删除。它们主要落在data准备
之后，但未有足够trace证明根因，不归因于磁盘、NCCL或GPU温度。正式节点此窗口
比先前隔离短测1.945秒慢；目前不能把后者当持续正式速度或断言发生了同阶段退速。
后续长窗以journal继续统计，不扩张本轮性能候选。

每步计时为data+同步forward/backward+engine.step的最慢rank；后续journal/metadata
记录与checkpoint在该区域外。首次1000步从worker开始到完整保存的实际wall1880.68秒，
包含初始化和保存；checkpoint独立70.5195秒，按1000频率条件外推约0.07052秒/步。
按当前正式窗口估计，完整IL含保存约70–75小时，资源竞争/DEV另计；这不是完成成本。
首个完整阶段checkpoint已无新增payload硬链接归档，避免rolling删除其研究证据。
证据：`audits/FORMAL_FIRST_STAGE_AUDIT.json`、`audits/FIRST_FORMAL_BOUNDARY_ARCHIVE.json`。

后续持续运行快照（2026-10-25T18:20:23.992610+00:00）已到1428更新；联合阶段1005–1428共424次计时：
mean/p50/p95=**2.58765/2.47955/2.62230秒**，12.36642 scenes/s，显存峰值与上表相同。
它包括真实尾部等待，未用短测值或接口阶段速度替代。当前证据见
`audits/FORMAL_CONTINUOUS_JOURNAL_TIMING.json`，更新后的长期速度仍由journal跟踪。
