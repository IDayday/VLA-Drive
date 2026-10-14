# 训练执行优化：保持同一真实链与更新结果

本记录只处理执行效率。两条正式训练线、完整动作头、G16/K10、两个 inner epochs、原始场景顺序、canonical v1.1、五项安全成本、乘子、replay、参考保持及 ODE 部署协议都沿用已锁定设置。它不解决零优势组缺少学习信号的问题，也不将训练加速当作规划性能提升。

## 已测出的时间去向

[原正式训练最近 64 个完整 collection](training_cost_20261014T1705Z.json)，使用八个 rank 的实际日志和已封存候选，未额外生成训练样本：

| 环节 | C_SDE | N_FLOW_NOISE |
| --- | ---: | ---: |
| 候选生成 | 37.15% | 32.06% |
| 等待实际评分 | 40.23% | 46.67% |
| 动作头更新 | 18.38% | 16.92% |
| reference | 1.24% | 1.35% |
| 其余 IO / 同步等 | 约 3% | 约 3% |

C 窗口为 1,664 个 raw scenes / 2,258 秒；N 窗口为 1,592 / 1,479 秒。墙钟来自相邻原子候选归档时间，文件系统时间精度有限；按每次 first16 / refill8 调用取最慢 rank 分解，不能解释成精确逐 kernel trace。CPU worker 时间之和不等于墙钟。CPU 评分时间中约 70% 用于车身道路/对象几何余量，约 24% 用于 canonical PDM。

旧收集器在一个 rank 上执行“GPU 生成一组 → CPU 评分并等待 → reference → 下一组”，尽管每个 rank 已有两个评分 worker，仍没有同时提交第二个场景。完整采样还反复将同一冻结 BF16 Linear 权重转为 FP32；GPU profiler 显示这部分转换有明显开销。

## 三项执行改动及证据

1. **GPU 生成与 CPU 评分重叠。** 生成第一组后提交原官方场景评分任务，再运行 reference 和下一组；按输入顺序收回结果。first16/refill8、组内全部 16 条轨迹及 dual 的首批人口不变。所有评分 future 在返回 collector 前完成，不能跨 behavior 更新或 checkpoint。基础设施失败仍中止更新，不变成零 reward。
2. **相同 GEOS 几何操作向量化。** 仍使用实际模拟轨迹的 41 个车身、同时间对象、可行驶区域并集、孔洞和相同边界；不减少物理时间点，也不改成中心点近似。静态几何上下文仅作有界缓存。
3. **缓存冻结 behavior 的 FP32 Linear 权重转换。** 保留原 BF16 参数及 dtype 语义，只缓存其精确 FP32 副本，每次 `load_state_dict` 后刷新稳定地址。当前可训练网络不走该缓存；不缓存 hidden、qwen_proj 输出、sigma 或跨版本输出。额外显存约为该冻结动作头 Linear 参数的一份 FP32 副本。

各项单独测量，不能相乘当成整体加速：

| 测量 | 原实现 | 优化后 | 范围 |
| --- | ---: | ---: | --- |
| 512 条真实轨迹几何余量 | 50.034 秒 | 3.809 秒 | 几何环节 13.14×；全部结果逐项一致 |
| C_SDE 两场景采集，仅重叠 | 12.299 秒 | 10.357 秒 | 采集约 1.19× |
| N_FLOW_NOISE 两场景采集，仅重叠 | 14.585 秒 | 9.146 秒 | 采集约 1.59× |
| 冻结权重转换缓存 | — | 采样约 1.4–2.0× | 两模型、SFT/已训练权重、固定 TRAIN 场景；不是全训练加速 |

证据：[几何及 CPU profile](vector_geometry_qualification_20261014T1713Z.json)、[仅重叠配对采集](collection_overlap_qualification_20261014T1710Z.json)、[冻结权重转换与刷新](frozen_cast_qualification_v1.json)、[真实 GPU 采样 profile](rollout_kernel_profile_20261014T1715Z.json)。小样本计时有共享资源负载影响。所有 warmup 和重复官方评分计入资格测试成本；它们不进入正式训练人口或预算。

## 等价验收与执行身份

原 83 个训练/评价源文件保持正式 seal 不变。新入口位于独立 analysis 命名空间，复用原生 trainer、checkpoint 和评分 worker；单独保存执行源文件 hash、源码快照和 acceptance。执行身份进入原子 checkpoint，而不是事后修改已封存恢复点。

重叠 + 几何的第一套八卡验收已经全部通过：[聚合证据](overlap_geometry_full_acceptance.json)。C/N 两组实际完整动作头学习、原始 replay dropout、KL、dual、全部八分片 FP32 optimizer、每 rank RNG、数据与 inner 游标、最终完整 ODE 导出均与原实现逐值一致。N 另验证第一 inner 更新后中断并原生恢复；所有保存链、请求、模拟反馈、成本、优势、组选择一致。仅时间字段和明确的执行 provenance 不参加相等比较。

加入冻结权重转换后的第二套组合验收与正式接续结果在获得完整证据后补充，不预先记为通过。

保留的失败：最初包装入口使用 argparse REMAINDER，torchrun 吞掉 `--` 后在解析参数阶段退出，没有执行任何训练。已改为只解析包装入口选项并保留原生参数的 `parse_known_args`；失败源码和日志保留在私有 `efficient_execution_qualification_v1`。这是启动接口问题，不是算法结果。

## 实际命令与恢复

从本分支 worktree、已授权 Tsinghua 主机和原 Python 环境执行。八卡验收及正式接续需要足够空闲显存，不占用未知进程。

```bash
PY=/mnt/project/flow-safe-improvement-20261013/venv/bin/python
RUN=/mnt/project/constrained-flow-noise-20261014

# 已执行：原训练时间分解；该命令拒绝将重叠后的时间当串行时间相加。
$PY -m scripts.analysis.constrained_flow_noise.profile_training_cost \
  --run-root "$RUN" --collections 64 --output "$RUN/training_cost_20261014T1705Z.json"

# 已执行：八卡真实学习与 inner 中断恢复；输出目录不允许覆盖。
$PY -m scripts.analysis.constrained_flow_noise.qualify_efficient_training \
  --run-root "$RUN" --output "$RUN/efficient_execution_qualification_v3" \
  --adapter-module scripts.analysis.constrained_flow_noise.train_efficient_cast
```

正式接续先向本任务八个 trainer rank 发 SIGUSR1，等全部保存同一原生完整恢复点并退出，随后 `resume_efficient` 校验源码、acceptance、checkpoint 全部 hash、八 rank 停止状态、运行锁和空闲显存。不会重置 Adam、lambda、KL controller、LR 阶段、数据/RNG 或已保存链。恢复命令与实际 checkpoint 必须取已生成的执行 receipt，不从权重文件单独续训。

优化后的 `scoring_seconds` 是提交到收回结果的延迟，包含与 GPU 的重叠；不能再与 rollout 相加。新 `compare_execution_throughput` 使用完整连续归档间隔比较 raw scene 吞吐，另报重启/冷启动成本。优化前后连续数据窗不是同场景配对，生产吞吐变化需要与固定场景 ABBA 测量分开解释。

## 仍存在的问题

- 零最终优势组和 refill 后超过 16 组的有效 surplus 仍有采样/评分成本。直接丢场景、减少 G/K、跳过零组 reference/dual、改变组选择都会改变本轮实验合同，未作为“无损优化”实施。
- C_SDE 在原实现 update1850 后触发 fresh-ratio 一致性检查，正在独立从 update1024 精确重放恢复。执行加速不构成该数值故障已修复的证据；未放宽概率检查或 clip。
- 当前 DEV 显示平均 PDMS 有收益，但 TTC 安全保持仍未合格；更快执行不能替代最终单候选 ODE 评价。两条正式预算终点仍未完成。
