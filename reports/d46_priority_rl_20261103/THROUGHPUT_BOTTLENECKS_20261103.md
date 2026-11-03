# D46 当前吞吐瓶颈：完整 collection 与一分钟资源采样

证据生成于 2026-11-03T15:04:47.998852+00:00。本报告只读日志、冻结源码和 GPU/CPU 状态；没有运行模型、评分、profiler、抢锁或信号操作。完整数据见 `/mnt/project/d46-priority-rl-20261103/analysis/THROUGHPUT_BOTTLENECKS_20261103.json`。

## 实测端到端周期

每条路线均读取全部 rank 最近 40 个完整、连续、同 invocation 的 M2 collection；每个 collection 为全局 B64/G16、2 次 Adam。稳态统计排除前一边界实际保存 checkpoint 的间隔。端到端以同一 rank0 的相邻完整 collection 结束时间之差计算。

|路线|world / update窗口|完整/稳态轮数|整轮均值s|基础生成s|基础评分s|两次更新s|其余残差s|场景/s|
|---|---|---|---:|---:|---:|---:|---:|---:|
|D0|16 / 2310→2390|40/40|21.154|9.121|0.839|10.401|0.794|3.025|
|D3_FPO|16 / 1492→1572|40/39|33.141|9.475|0.815|19.485|3.366|1.931|
|D2|8 / 1758→1838|40/39|75.381|23.952|1.924|31.298|18.208|0.849|

基础生成计时包含 G16 采样及当前/冻结参考 W3 重算；**不包含**随后 FPO MC4 旧能量准备、D2 nominal B1 与自适应扩展。残差还含 finalization、behavior 保存、控制与跨 rank 等待，不能全叫 CPU 或评分成本。不同路线、曝光点、world 与场景不同，本表不是严格匹配的算法吞吐因果对照。

D2 这 40 轮实际增加 2,560 个 B1 和 5,312 个扩展候选查询；基础评分字段漏记这些查询，因此不能据 1.924s 断言 D2 的全部评分只占 2.6%。D0/FPO 速度计时没有这类额外查询。

## GPU 与 CPU 的真实采样

四节点并行采样 60 秒，每 5 秒一次、每节点 13 次。只读已有 RUNNING 中严格确认的各 8 个 rank PID/start；没有输出未知任务命令或环境变量。

|节点|所属路线|GPU util均值|功率均值W|每个已知rank CPU核均值|
|---|---|---:|---:|---:|
|training-vlawm-zt4|D0|34.1%|147.9|0.944|
|training-vlawm-tsinghua-zt5|D0|34.7%|152.8|0.936|
|training-vlawm-tsinghua-zt3|D3_FPO|37.9%|128.1|0.957|
|training-vlawm-tsinghua-zt4|D3_FPO|37.0%|127.3|0.954|

这些是整卡瞬时 NVML 数据；TSH3/TSH4 上还可能有已知受保护的 IL 评测，不能当作单训练进程占用。rank CPU 接近一核、GPU util 约 34–38%，与 Python 调度/同步/小 kernel 瓶颈相容；没有 profiler，不能据此断言有某个百分比可直接追回。PDM 子进程 CPU 未混入 rank CPU。

## 优先级与可省什么

1. **先消除重复 CUDA→主机标量判定。** `qwen_flow_adapter.py:28/75`：28 层的 K/V 共 56 项，再加 valid/next_positions/reasoning，共 59 次 `_select`。G16 单场景广播的每次 `DPrefix.index_select` 会执行 59 次 `bool(torch.all(...))`、59 次 `int(indices[0])` 及 16 次 names 的 `int(i)`，共 134 次潜在主机同步取值。D0 每 rank 每轮 20 次 prefix 构造，静态可达 2,680 次；FPO 为 24 次/3,216 次。此为源码调用计数，**不是同步耗时测量**。将合法 scene 选择及广播信息在 CPU 上验证一次，复用给所有 K/V；保留 stride0、候选顺序、边界和 autograd 语义。

2. **减少重复 prefix/批量执行实际模型计算。** 日志 D0 每 rank/轮为 current velocity 76、reference 12、current prefix16、reference prefix4；FPO 为76/12/20/4。FPO 的4次旧能量与2×4次训练 MC denoise 未计入 velocity counter，实际每 scene 增12次 forward 调用；训练 MC 还需反向。可以测试等价 MC/候选批量化，不能把减 MC 或去掉 W3 anchor 包装为纯工程优化。采样与 W3 重算处于同一 θ，可先研究共享无梯度 current prefix；跨 Adam 更新不可缓存 trainable C/P。冻结 reference 本窗口已经每 scene 只构造一次，没有证据称它被反复重建。

3. **评分与数据先精确拆时，再决定流水化。** `RewardService.score = submit(...).result()` 串行阻塞；下一场景 GPU 生成与已生成候选 CPU 评分可研究有界重叠，但必须保持版本/RNG/query顺序和 pending checkpoint 可恢复。冻结视觉预处理已 LRU 缓存，本窗口每 rank/轮耗时仅 D0约.552s、FPO约.548s、D2约1.357s，且包含 decode/processor/encode 与同步；不能重复加到已计生成时间中。

## 端到端收益边界

|假设（不是实测收益）|D0|FPO|D2|
|---|---:|---:|---:|
|已计基础评分全部免费|1.041×|1.025×|1.026×|
|两次更新区间成本减半|1.326×|1.416×|1.262×|
|已计基础生成区间成本减半|1.275×|1.167×|1.189×|

以上 Amdahl 情景只作用于明确计时区间，假定其余不变，彼此不能相加；评分免费不可能完全实现。D2 的“基础评分免费”不涵盖额外查询。`DPrefix` 同步节省上限目前没有可辩护的百分比，应先在独立合法卡做短微基准，而非承诺 GPU util 提到100%或若干倍提速。

## 可用于后续资格微基准的只读候选

采样时 WM2 GPU0/1 各 free81,153MiB；RL2 GPU1/7 各 free81,218MiB；四卡 util0、compute owners为空。UUID 与时间见 FACTS。未抢锁、未启动任务；实际运行仍须实时授权、UUID/lease/headroom/未知 owner 检查，GPU6 未纳入。

## 可复现与来源

- `results/throughput_bottlenecks/FACTS.json`：紧凑数值、文件 SHA、来源路径。
- 同目录 `build_throughput_bottlenecks.py`：有限日志窗口/冻结 source 聚合；重新运行会选当时最新完整窗口。
- 同目录 `sample_throughput_resources.py`：一次60秒只读采样，重新运行需明确授权，不自动循环。
- NAS 主 JSON 保留120条 collection 的逐 rank 时长、40个 journal 读取片段 SHA 与源码 SHA；未读取图像、模型或 optimizer tensor。

本轮不改任何在线科学 source、配置、owner、预算或任务，不 stage/commit。
