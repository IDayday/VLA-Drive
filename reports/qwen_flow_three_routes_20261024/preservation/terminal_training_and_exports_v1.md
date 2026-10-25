# P0–P5固定初筛终点、纯部署导出及成本

记录时间：2026-10-25T08:57:13.373251+00:00。六组均完成登记的256次optimizer更新、64份fresh collection、1024次scene曝光和16384条新候选评分；合计1536次更新、384份collection和98304条新候选。这里是本轮有限初筛终点，不是历史50k预算。仅训练seed42。

六组终点均已CPU严格读取完整checkpoint，核对每组480个Adam参数状态的step全部为256，FP32 moments有限，scheduler.last_epoch为256，pending batch为空、next_inner=0。全部匹配模型worker已有退出记录；暂停迁移的exit0与最终COMPLETE状态分别核对，原P0 OOM的失败记录和190.397秒费用保留。

## 已完成的核验与部署包

384份已提交collection的payload全文件SHA、scientific identity、固定chain/old logprob/reward摘要、最终优势逐值重算全部通过；P4/P5逐collection验证投影dual算术和只更新一次的计数。四次inner只计一份fresh数据，迁移和P0原链恢复没有增加虚构的新逻辑样本。

现有immutable345的`export_adapters`导出P0–P5终点包，并用`read_adapter_package`复核。每包只含480个adapter张量和封存元数据；不携带optimizer、dual、archive、teacher或reference网络。推理时仍由已验收的FP32 CPU `B@A*2`合并进普通ActionDiT。此次没有启动GPU、修改GLOBAL_FINAL锁或资源lease。

本地统一索引为`/mnt/project/qwen-flow-three-routes-20261024/P_FINAL_EXPORT_INDEX.json`，seal identity：`a8af421ad7db8d8996f2fba0fda7d7daa48623285afa0e1744557e63f27d3258`。各包位于`preservation/exports/P{0..5}_seed42_terminal256`；这些大张量不进入git。训练源码保持immutable97；导出源码保持immutable345。

## 终点训练机制与实际费用

|组|有效PG组|安全分项有差异的组|inner2–4 clip率|额外存档监督scene曝光|共享worker占用小时|
|---|---:|---:|---:|---:|---:|
|P0|99.707%|75.195%|0.456%|0|8.581|
|P1|99.805%|75.781%|0.174%|0|8.991|
|P2|99.707%|75.293%|0.343%|4096|9.994|
|P3|99.707%|75.391%|0.334%|4096|9.957|
|P4|99.707%|73.633%|0.398%|0|9.173|
|P5|99.707%|73.828%|0.414%|4096|9.935|

早期100%有效PG是中期快照。完整终点中P1有2/1024个零优势组，其余每组3/1024；所有组仍各完成256次含有效PG的RL更新，reference/replay-only更新均为0。progress-only组占比均为0。第一inner平均ratio精确为1、clip率0；后续inner仅少量真实更新触发clipping。当前窗口不支持“主要被无优势组或PPO clip阻断”的解释；这也不否定原ODE采样中的饱和现象。

累计共享worker占用为56.631小时。已计入原OOM、原主机执行段、合法迁移后的执行段及加载/退出；不是独占GPU小时，也不是六组串行端到端时间。ledger中的update约占73.5%，generation约22.9%，score等待约0.61%。评分等待不是CPU总仿真工作量；checkpoint保存和CPU评分worker时间未独立完整计时，不能把未归属差额统称保存耗时。不同组跨主机/共享环境不同，因此这张实测成本表不能把所有时间差严格归因于算法。

P2/P3/P5的4096次正权重FM存档监督曝光是额外训练成本，不产生新on-policy链、不附加伪造old logprob。三组current速度网络函数调用均为53248次，其余49152次；reference均49152次。函数调用带有共同固定chunk布局，不能等同于单样本数量。

## 必须保留的校准限制

- P1仅用TRAIN做过两次LR校准，最终LR为2.5e-5。漂移匹配仍是`NOT_MATCHED`：最大相对误差49.105%，超过20%容差。该量是共同root请求轨迹漂移代理，不是输出KL；不能据此声称已完全排除“只是更保守/学得更慢”的替代解释，不继续补调参。
- replay系数锁为1并触及预设上限。固定4scene校准中的replay/PG梯度比为5.287%，低于目标25%；PG/replay梯度cosine约+0.1845。这一小批次结果不代表全训练人口，也不能把无收益直接解释为多轨存档概念无效。
- 固定archive来源为共同ODE K32候选及5条独立参考，435/512个TRAIN场景有合格条目。P2单最佳与P3/P5多轨从同一初始库构建；没有合格目标的场景不制造成功标签。
- NC/DAC/TTC成本预算来自同核初始source-SDE的128×16条未筛选真实候选，分别为0.036376953125/0.1318359375/0.12353515625，epsilon=0；不能用ODE候选的更低风险冒充SDE预算。
- P4/P5 dual均只更新64次；终点λ分别为`[0.997500,0.979336,0.998906]`和`[0.997754,0.979063,0.999414]`。变化幅度最多约2.1%，当前差异主要含初始为1的成本项，不能夸称自适应乘子已经学出很强的新权衡。
- P4仍有155条违规候选得到正最终优势，其中151条同组有全安全替代；P5相应为167/163。平均成本的相对优势surrogate不是逐候选硬筛选，也不是逐场景安全保证。是否减少真实部署退化必须看固定全Navtest。

共同资产成本需单列：TRAIN的32候选+5独立参考共18944次评分，连同原DEV诊断为28416次；初始同核SDE预算校准另有2048次评分和12个shadow optimizer updates，影子权重没有进入正式初始化。存档和缓存复用不等于免费新数据，不将同一共享成本重复乘六。终点sampler审计、完整Navtest、资格验证的费用继续独立登记。

## 当前研究判断与复跑入口

本报告证明固定初筛训练、恢复祖先和纯部署导出已闭合，不证明方法提高PDMS。P0–P5的效力判断仅依据已登记的完整Navtest：12146场景/136logs、固定五个推理seed、单候选source shift5 ODE/canonical v1.1；不使用DEV删组，不使用SDE均值、candidate最高分或训练loss替代。当前路线标签仍为“证据不足，等待终点全Navtest”。随后由公共统计器做P3−P2、P5−P1及archive×constraint因子比较，并明确P1未匹配的限制。

CPU核验/导出命令（实际已执行）：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  /mnt/project/qwen-flow-three-routes-20261024/preservation/tools/export_terminal_adapters_cpu_v1.py

CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  /mnt/project/qwen-flow-three-routes-20261024/preservation/tools/summarize_committed_v1.py \
  --require-complete --verify-all-payload-hashes \
  --output /mnt/project/qwen-flow-three-routes-20261024/preservation/analysis/terminal_complete_v1
```

导出工具遇到已有同一终点包时校验复用，不覆盖异身份结果。历史汇总保留已有输出；再次复算需指定新的输出目录。聚合证据为同目录`terminal_training_and_exports_v1.json`；逐collection CSV和大payload仅保留在私有运行资产中。
