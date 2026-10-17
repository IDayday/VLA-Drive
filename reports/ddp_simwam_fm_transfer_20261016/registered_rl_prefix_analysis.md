# 固定RL观察点的训练与成本汇总入口

2026-10-17，在B继续原定20k训练期间，将已实际执行的B-5k汇总整理为
`scripts.analysis.fm_transfer.summarize_rl_prefix`。该入口仅读取既有数据，
支持已登记的5k/10k/15k中间边界；20k仍使用现有完整训练汇总。
训练、采样、评分、部署和正在运行的controller源码均未修改。

原5k汇总及复现脚本保留。新入口复用原`read_journal`、`validate_rows`和
`summarize_window`，不重写统计公式，也不把每份rollout的四次inner当作
四倍新候选。新增参数化入口避免为每个观察点复制并手工替换预算数字。

## 实际执行与对照

在无GPU的CPU执行中，重新读取B的原生`update005000_inner0`、八rank日志、
5k精确progress和invocation成本快照，写入独立资格目录。
与已发布5k摘要逐字段核对：全部累计统计、十个500-update窗口、场景/log
覆盖、checkpoint身份和全部成本值**精确相同**。仅采集时间、分析脚本hash、
本次读到的活跃日志长度允许不同。

核算仍为5000次全局更新、1250次fresh全局rollout、10000次场景曝光、
80000条候选。此次分析新增推理、评分、optimizer更新均为0。
证据见[实际对照与边界检查](registered_rl_prefix_validation.json)。

实际执行的入口如下；原输出保持，复核应选择新的输出路径：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.analysis.fm_transfer.summarize_rl_prefix \
  --run /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1 \
  --updates 5000 \
  --output /mnt/project/fm-transfer-20261016/qualification/registered_rl_prefix_cli_v1/B5k_recomputed.json
```

## 必须满足的输入边界

- 原生checkpoint已经封存，八rank文件hash及READY/inner0/更新与采样计数一致。
- 正式配置仍为world8、每rank一场景、G8、inner4和20k总预算。
- 使用该观察点捕获的progress与八rank成本快照，不从更晚的活跃账本猜历史成本。
- 当前入口只接受同一从0开始、未经恢复的invocation；恢复后的累计成本应走已有
  invocation账本汇总，不能用累计更新除以新进程时间。
- 后续已经完成的日志可以存在，但不进入被请求的固定前缀；重复update仍被拒绝。
- 输出禁止覆盖，未封存的观察点不写结果；20k明确交给完整终点汇总。

实际检查已证明：重复写5k被拒绝且原文件hash不变；尚未封存的10k被拒绝且
没有生成结果文件；20k被中间入口拒绝。10k/15k统计仍需等待其真实边界，
不能把入口验收当作这些训练阶段已经完成。

八rank时间是重叠的worker elapsed，不是独占GPU小时；scorer elapsed不是
CPU核秒。共享GPU、模型准备、保存和训练循环wall继续分别报告。保存于
观察点后的活跃invocation候选计数不参与该观察点逻辑预算。
