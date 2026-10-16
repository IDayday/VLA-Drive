# 固定 SFT reference 的四次 inner 复用

2026-10-16。两个真实八卡保存链对照已通过：在已测零/非零LoRA、两个随机窗口上，loss、192个LoRA张量的归约梯度、参数、Adam、scheduler和RNG逐项一致。每份collection的reference DiT前向由12次降至3次，更新区间耗时实测降低20.40%和21.12%。这是有限执行等价性与吞吐证据，不是新的规划收益。

现有LAST_ODE和SAFETY25正式run继续使用各自原不可变源码；本补丁尚未装入这些健康运行的训练。没有改变20k终点、数据人口、seed、SDE、损失、优化器或评价协议。当前原ODE Navtest最好为X1的89.3817，92目标仍未达到；本次没有新增Navtest或算法组。[已有完整研究证据](https://github.com/IDayday/VLA-Drive/blob/a93f80a01abc8ad885049a115d6f0e641d8935ee/reports/ddp_simwam_x_anchor_20261015/flow_design_review/multi_source_analysis_20261016.md)

## 1. 改动与边界

源loss在四个inner中用永久冻结SFT、相同condition、相同保存的随机状态求reference均值。三个随机步骤每inner各一次DiT前向，共12次。本补丁只在第一次求值，之后三次读取同一detach结果；CURRENT仍在每次inner重新前向、反传和更新。四次实际Adam更新及old链、old概率、优势固定的语义不变。

| 项目 | 实现与约束 |
|---|---|
| 缓存对象 | 一份保存链的reference logprob、mean/velocity/std及原概率审计结果 |
| 不缓存 | CURRENT、LoRA可训练激活、采样结果以外的新状态、reward、优势、optimizer |
| key | 输入tensor对象及存储身份、version、shape/stride/offset、dtype/device；time/window；sampler设置；全部reference参数/buffer版本 |
| 准入 | 原`no_grad`上下文；所有reference子模块eval；所有reference参数冻结；输出无autograd graph |
| 失效 | 换链、输入原地修改、窗口/配置、参数/buffer或layout变化均重新计算 |
| 输出保护 | 缓存输出被原地修改则拒绝继续；持有输入引用，避免对象/指针复用误命中 |
| 恢复 | process-local临时状态不进入checkpoint；新进程冷启动重新计算 |
| 开关 | `--reuse-frozen-reference`，默认关闭；开关进入execution identity，不改变scientific identity |

实现：[reference_reuse.py](../../starVLA/rl/simwam_recipe/reference_reuse.py)、[trainer_adapter.py](../../starVLA/rl/simwam_recipe/trainer_adapter.py)、[train.py](../../scripts/analysis/simwam_recipe/train.py)。复用的是同一reference计算，没有用近似模型、低精度结果或另一条教师轨迹替代。张量版本检查适用于正常PyTorch更新；不授权绕过版本计数的`.data`写入。

原数学入口仍为vendor中的SimWAM loss和SDE；创新clamp、逐维relative clamp后dimension-mean、全局raw reward std、源rank-mean、beta50 anchor、实际gradient clipping=0、DDP固定bucket等均保持。缓存不改变源部分随机步骤的经验代理边界，也不会把它变成精确整链policy gradient。

## 2. 已执行的保存链等价性验收

CPU共16项覆盖输出/RNG、输入和reference版本失效、layout/sampler变化、输出篡改、冷缓存、train mode/trainable reference及autograd上下文拒绝，见[cpu_qualification.json](cpu_qualification.json)。默认execution字段不变、启用开关被身份绑定且scientific字段不变，以及真实TRAIN清单和CLI预检见[execution_option_qualification.json](execution_option_qualification.json)。

GPU验收使用完整DDP动作头、原LoRA、原AdamW和真实world8梯度归约：

| 对照 | 起始权重 | 随机窗口 | 实际资格更新 | 新候选/新评分 |
|---|---|---|---:|---:|
| LAST_ODE | 原SFT零LoRA及该run的update4900 | [6,7,8] | 48 | 0 / 0 |
| SAFETY25 | 原SFT零LoRA及该run的update1000 | [7,8,9] | 48 | 0 / 0 |

每组零LoRA分别走一次uncached/reused；非零LoRA走AB/BA两种次序以及inner中清空缓存路径。每个非零路径重放两个已提交collection、各四次更新。除比较loss/metric和全部192梯度hash外，还比较每步LoRA及终点Adam/scheduler/全部RNG；进入第二份collection前必须准确重现原正式behavior的adapter身份，不能仅证明两条新路径彼此一致。每个rank均验证冻结参数未变、固定chain摘要与第一inner ratio。

全部比较通过，current/reference调用计数符合预登记值，见[LAST八卡证据](LAST_ODE_world8.json)、[SAFETY八卡证据](SAFETY25_world8.json)。这些资格权重不用于任何正式初始化；保存链重放没有额外策略采样或官方评分。总量为96个全局optimizer updates，不能乘八卡后称为768次独立训练。

GPU数学对照源码为`94b400f6553f844e5d92e0b014adc75d15cabe9e`；训练开关及恢复集成源码为`01ab0800632680135098b4b00e8410b6c0b6bc08`。后者没有修改已通过的reference复用或actor代码。

## 3. 实测速度与成本

计时仅使用非零LoRA的AB/BA路径。每次update取八rank计时的最大值，再对16次更新汇总；计时包括current/reference、loss、backward、梯度norm、Adam和scheduler，排除模型/保存链加载和证据hash。共享GPU，不是独占GPU核算。

| 配方 | 原更新区间秒/update | 复用后秒/update | 时间减少 | 更新吞吐倍数 |
|---|---:|---:|---:|---:|
| LAST_ODE | 0.74208 | 0.59070 | 20.40% | 1.2563× |
| SAFETY25 | 0.74449 | 0.58728 | 21.12% | 1.2677× |

AB/BA分别得到LAST的21.30%/19.49%、SAFETY的21.21%/21.02%时间减少。原12次reference降至3次意味着**reference前向减少75%**；它不代表总训练时间减少75%。rollout、old probability、真实评分、保存、加载等不受该优化直接加速，尚无正式run端到端提速结果。

两个成功资格job wall分别为92.07/91.97秒，累计八worker wall为691.15/690.73秒；峰值PyTorch allocated显存每rank约6.6303GiB。该内存不是驱动全部占用或共享其他进程的总显存。完整原始计时、归约方式与成本在[profile_results.json](profile_results.json)。

保留两次GPU启动失败：第一次torchrun将`--run`视为歧义launcher参数，1.384秒；第二次管理脚本替换attempt编号时误改formal run路径，9.039秒，在模型载入前因身份缺失退出。修正worker参数分隔和显式run白名单后才成功。两次均0更新、0采样、0评分，原run未写入。CPU首轮输出父目录缺失的失败单独保留于[cpu_harness_failure.json](cpu_harness_failure.json)，没有把失败开销隐藏到成功吞吐中。

## 4. 真实进程恢复验收

另行登记了实际训练入口的连续8步与inner2退出后恢复至8步，使用同一真实TRAIN资格清单和SAFETY25固定6+2人口。两条路径总计16个全局资格updates、4份fresh rollout、256条真实候选和评分；这是额外验收开销，不能继续写成全部工作0评分。见[native_recovery_registration.json](native_recovery_registration.json)。

实际连续/恢复路径已通过：八rank最终LoRA/Adam/scheduler/RNG、全部loss/梯度norm/ratio/LR、下一份fresh链/old probability/advantage均准确一致。八rank的192个LoRA张量实际改变，冻结参数hash均未变，DDP没有重建bucket。恢复后第一个inner的cache命中数归零并重新计算reference，随后正常复用；连续路径两份collection各rank共6次reference前向，跨进程路径共9次，额外3次正是冷恢复计算，不通过保存旧cache伪装一致。

原生比较见[native_recovery.json](native_recovery.json)，完整invocation、有效样本、参数合同及成本核对见[native_integration_audit.json](native_integration_audit.json)。总job wall为171.02秒，24份rank-invocation的准备和执行时间合计1204.39秒，为共享worker累计wall，不能称为端到端wall或独占GPU时间。实际新生成/评分各256条，与预登记一致。资格launcher与supervisor已退出，GPU占用回到先前共享基线，未停止其他进程。

这是实际进程退出和native恢复；前节保存链测试中的`cache.clear()`仅验证临时状态可丢弃，二者分别记录。没有扩展为新的长训或完整故障矩阵，也没有用资格权重启动正式训练。

## 5. 使用与恢复边界

CPU入口已实际执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m scripts.analysis.simwam_recipe.qualify_reference_reuse cpu --output "$NEW_CPU_RECEIPT"
```

完整模型保存链验收入口已实际执行；`--`用于明确区分torchrun参数和worker的`--run`：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 --module -- \
  scripts.analysis.simwam_recipe.qualify_reference_reuse real \
  --run "$EXISTING_READONLY_RUN" --start-update "$SEALED_READY_UPDATE" --output "$NEW_GPU_RECEIPT_DIR"
```

真实训练/恢复验收入口：

```bash
python -m scripts.analysis.simwam_recipe.qualify_recovery_run \
  --root /mnt/project/simwam-reference-reuse-20261016 \
  --name eight_gpu_reference_reuse_v1 \
  --config configs/flow_rl/simwam_canonical_x1_safety25_ddp.yaml \
  --manifest /mnt/project/simwam-safety-exposure-20261016/qualification/real_TRAIN_tokens.json \
  --interrupted-after 2 --reuse-frozen-reference
```

已用输出不可覆盖；重验要使用新name和独立预算记录。所有GPU命令只能在实时授权、实际有足够显存的主机执行，环境继续使用原数值合同。当前formal运行不可直接追加新开关或替换源码；跨源码恢复需绑定明确父checkpoint、旧/新完整身份、通过的等价性证据和migration receipt。没有为现有健康run签发迁移receipt或强制中断。

训练日志中的reference reuse计数是本次invocation内累计值，恢复后重新计数；它不是累计scene或有效RL数据量。已有source hash检查、事务恢复、日志归属与科学人口断言继续有效。

## 6. 正式任务保护与适用结论

只读核查时LAST_ODE为6900更新、SAFETY25为2700更新，均处于合法READY边界；两run各24个正式source文件与原快照及run内副本hash全部一致，见[formal_source_preservation.json](formal_source_preservation.json)。两个正式launcher和四个milestone/final controller均按PID/start-time/实际命令核实存活，[交付前进程复查](runtime_snapshot.json)另存实际时间和进一步进度；状态有时间戳，不是实时承诺。两者仍按原固定观察点及20k终点执行。

本项支持在经过明确源码迁移的后续合法恢复中采用reference复用；尚未提供长期正式run端到端加速或任何新增单候选规划分数。当前优先完成已登记的随机窗口与安全失败人口曝光对照，由其固定ODE结果判断下一步方法，而不根据本项性能测试更改算法。
