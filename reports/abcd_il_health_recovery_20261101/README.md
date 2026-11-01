# ABCD IL、检查点评测与tsinghua-zt4撤回：2026-11-01

用户询问检查点评测和ABCD训练是否正常，随后撤回tsinghua-zt4。已实际核验进程、八个rank日志、原生checkpoint及独立Navtest队列，并恢复了中断的C组。

## 当前结果

恢复后一次实际核验：

| 组别 | 八个rank共同有效更新 | 最新完整Navtest checkpoint | canonical v1.1 PDMS |
|---|---:|---:|---:|
| A | 40,359 | 40k | 86.742934 |
| B | 33,767 | 32k | 86.504104 |
| C | 33,019，已恢复 | 32k | 85.552021 |
| D | 26,578 | 26k | 89.047748 |

四组最后记录的loss均有限、optimizer计数与有效更新一致、skipped均为0；日志新鲜度约1–3秒。它们属于不同训练预算的中间点，不是匹配终点的架构排名。“训练进程正常”也不意味着分数单调提高。

四个分数各为完整12,146场景、invalid0，单候选clean-x K10/shift5 ODE、seed2101。A40k在本次核验期间由运行中变为完整，未使用部分场景均值。当前ABCD已登记Navtest任务没有FAILED或待处理积压，原每2k检查点队列owner保持运行，未创建第二个owner。

## C组故障与实际恢复

初次核验发现：C日志停在33,224，约5小时未更新。原任务在07:07收到SIGTERM；worker、torchrun及原rank进程已不存在，但旧RUNNING.json没有终结记录，旧CPU总控仍将它当作运行中。现有日志没有该次退出由数值错误引起的证据，也无法确认外部SIGTERM的发送来源。

中断时保存的`.saving/update033224.*`仅有不完整model文件，没有合法COMPLETE、完整Adam分片和rank状态，未将其提升为恢复点。最后合法checkpoint是33k，18个原生文件齐全，inventory seal有效；正式loader继续执行完整文件hash及严格identity校验。

处理如下：

1. 保留原失败目录、SIGTERM记录、旧worker收据和未完成保存文件。
2. 八个rank的原始journal通过hardlink完整归档，逐rank验证有效前缀恰为1…33,000，未提交尾部恰为33,001…33,224。新有效journal保留33k前缀，224步将重做。原始计算成本保存在归档及`journal_rollback.json`，未按最新日志假装这段计算没有发生。
3. C迁到仍授权的**tsinghua-zt3八卡**，保持world8、global32、micro2/accum2、全部七项目标、BF16前向、FP32参数/Adam、100k预算及全部训练源码不变。按原生33k完整恢复Adam、scheduler、RNG、builder及scene cursor，未重置优化器。
4. 实际完成加载后，八个rank均已更新到33,019、optimizer计数一致、skipped0，故恢复状态写为`RECOVERED_ACTUAL_UPDATES`，并非仅“命令已启动”。

该恢复需要重做224个未封存更新，不能称中断前所有计算均无损保留。

## 运行归属与最小工程补丁

科学源码仍为原immutable `runtime/fast_5a2d5b5`。新任务入口只修改外围job worker：

- 原legacy owner识别保留。对于已有Safety F/G cohost，仅在实际模块为该研究的train/evaluate、指定父模型严格为`qwen-world-rl-optimization-20261030/IL_PARENT`且包含对应研究运行路径时识别；单凭新增路径前缀不能放行未知进程。
- 保留原GPU UUID、可见owner、2048MiB未解释显存阈值、实际headroom、现有primary/shared lease及授权检查。没有驱逐cohost、增加lease命名空间或绕过未知显存检查。
- 增加至多1800秒的pre-child容量等待；每次重试重新检查授权，释放不完整lease集合，CUDA子进程仅在完整准入后创建。默认等待0秒，维持原调用行为。

5个针对性CPU测试通过，涵盖合法/非法owner、错误父模型、路径前缀不足、有限等待释放partial locks、默认零等待及撤权立即拒绝；另有真实八卡恢复与更新证据。

实际运行的外围worker固定在`17c23cd`，工作区`VLA-Drive-abcd-il-c-recovery-20261101`已detach到同一SHA，保持代码不变。文档在独立publication worktree提交。训练子进程cwd/PYTHONPATH仍指向原fast源码，不使用本分支actor实现。

原CPU总控只对其精确PID/start身份做了接管，未向A/B/D或其他GPU learner发信号；旧C条目依据实际缺失进程标为中断，新C收据被显式采纳。当前唯一CPU总控为3184659，原ops源码`68eb482`保持不变，启动阶段显式为`il evaluate`，尊重此前用户终止ABCD RL的指示。

**另一个独立问题仍保留：旧完整DEV编排存在历史失败状态，不能用其架构级FAILED字段判断仍在运行的IL进程。**本次没有清除这些历史错误或绕过review，未宣称该DEV父模型选择链已修复。当前定期Navtest使用独立、正常工作的已验证队列。已终止ABCD RL、Huber及暂停T系列均未恢复。

## tsinghua-zt4撤回

已按用户最新指示为精确主机`training-vlawm-tsinghua-zt4`登记新时间戳的global dispatch fence，并从当前allowed host/aliases移除；未改不可变历史grant。实际检查ABCD与Safety两个任务的授权函数均已排除该主机，禁止新任务、评测和失败重试。

最终只读核验该主机：本项目可见GPU进程0、相关科学计算进程0。最后一个评测在核验前自然结束，无需kill。原C进程本已退出；未重启该机训练。对未能解释的宿主显存占用未做GPU reset或操作无归属进程。

## 资产、源码与复现

- 本分支：`fix/abcd-il-c-recovery-20261101`。
- 恢复资产：`/mnt/project/qwen-world-four-frameworks-20261029/operations/C_recovery_20261101`，含LAUNCH、原始/恢复journal收据、CPU接管、严格恢复与最终健康核验。
- 实际C invocation：`/mnt/project/qwen-world-four-frameworks-20261029/jobs/C_native_resume033000_20261101_attempt001`。不允许直接重跑该invocation。
- 撤权资产：`resources/TSINGHUA4_WITHDRAWAL_LATEST.json`指向本次REQUEST、旧policy及最终核验。
- 此目录保存不含原始场景的机器可读[健康结果](final_training_and_eval_health.json)、[恢复证明](recovery_complete.json)、[journal归属](journal_rollback.json)和[撤权证明](tsinghua4_withdrawal.json)。权重、缓存、完整journal与私有场景不进git。

```bash
# 已实际执行：仅针对外围资源准入，不启动模型
/usr/bin/python3 -m unittest discover -s tests \
  -p test_c_recovery_owner_admission.py -v
```

真实恢复的精确argv位于本地`operations/C_recovery_20261101/LAUNCH.json`，继续使用原训练入口、33k父checkpoint及原execution receipt。需要再次恢复时，应重新检查实际owner和授权、选最新合法封存点、归档超出恢复边界的旧journal，并创建新attempt；不能复用已经存在的worker目录或改用不完整33,224保存。
