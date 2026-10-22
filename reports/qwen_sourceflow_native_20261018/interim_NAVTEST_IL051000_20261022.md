# 51k IL全量Navtest与49k配对结果（2026-10-22）

**51k IL单候选canonical NAVSIM v1.1 PDMS为86.675474，相对49k的86.810426变化-0.134953分。** 未达到92，仍差5.324526分。 这仍是IL学习过程，不是Flow-GRPO收益。

## 本轮实际状态与评价合同

2026-10-22 23:08 UTC启动核查：正式IL八个rank到51,745/100,000步，最近完整恢复点为51k；训练日志持续增加。正式RL仍未开始，DEV队列仅20k/40k完成，继续等待60k/80k/100k，原父模型选择规则未改。

完成后于23:30 UTC核查：八个rank一致到 **51,845/100,000步（51.845%）**，累计1,659,040次scene曝光，最新完整恢复点仍为51k。最近500个完整update取各rank最大耗时后平均为13.627秒/update；按此速度，剩余update计算约182.3小时，另需保存和后续评价时间。该估计不包含未来资源变化，不是完成时间承诺。[进度记录](progress_NAVTEST_IL051000_20261022.json)。

用户再次明确要求新的Navtest，因此在新结果出现前固定最新完整51k恢复点。模型、optimizer、原生身份和seal通过同文件系统hardlink保留；不是改milestone清单，也不把51k纳入正式RL父模型选择。

- 全量12146场景/136 logs；seed2101；单候选10步source shift5 ODE；canonical NAVSIM v1.1。
- 当前左/前/右三视图、导航与合法当前ego；完整原生Qwen＋源ActionDiT8×3及原后处理。无搜索、GT选轨、scorer选择、best-of-N或额外seed。
- 与49k使用完全相同的d3a7ce8不可变评价快照，输入、精度、评分协议和所有source_files hash相同；本轮没有模型、训练、推理或评分代码改动。直接使用默认严格配对，无需新增源码兼容例外或重复数值矩阵。
- 新中期Navtest不参与训练、调参或IL父模型选择。20k/40k/49k/51k已被观察，后续不能称完全盲测。

## 全量成绩（0—100）

| 检查点 | Navtest PDMS |
|---|---:|
| IL20k | 80.370064 |
| IL40k | 85.786557 |
| IL49k | 86.810426 |
| IL51k | 86.675474 |

| 分项 | IL49k | IL51k | 变化（百分点） |
|---|---:|---:|---:|
| NC | 98.007575 | 97.838795 | -0.168780 |
| DAC | 95.372962 | 95.175366 | -0.197596 |
| TTC | 94.105055 | 93.759262 | -0.345793 |
| EP | 81.047307 | 81.467936 | +0.420629 |
| Comfort | 99.942368 | 99.967067 | +0.024699 |
| DDC | 98.518031 | 98.386300 | -0.131731 |

51k−49k按log聚类、保留scene权重的配对bootstrap（2000次）95%区间为 **[-0.702964, +0.461589] PDMS点**。仅表示固定训练/推理seed下的场景抽样不确定性，不包括训练或推理seed不确定性。

本次总分均值略降，区间跨0，不能据此认定持续退化或已进入平台期。EP提高与NC/DAC/TTC均值下降、新增零分并存；这次新增训练没有表现为净规划收益，也不能用进度分项的上升宣布整体改善。

## 恢复与新增失败

- 零分场景：781 → 826；零→非零258，非零→零303。
- NC/DAC/TTC联合完全通过：10925 → 10851；原联合安全人口新增违规421，原不安全人口联合恢复347。NC=0.5不计完全通过。
- G+贡献2.838980个PDMS点，G−贡献2.973932点，净变化-0.134953点。
- 任意提高/下降：5405/2881场景；达到0.005原始score幅度的提高/下降：3855/1826。任意负差不等于严重事故。

| 安全项 | 49k违规 | 51k违规 | 新增 | 恢复 |
|---|---:|---:|---:|---:|
| NC | 253 | 275 | 82 | 60 |
| DAC | 562 | 586 | 235 | 211 |
| TTC | 716 | 758 | 213 | 171 |

## 请求轨迹与实际模拟

| 描述统计 | IL49k | IL51k |
|---|---:|---:|
| 请求终点前向位移（米） | 19.050653 | 19.024906 |
| 请求分段平均速度（米/秒） | 4.962538 | 4.962779 |
| 请求分段jerk RMS代理（米/秒³） | 4.740840 | 4.471431 |
| 模拟前向位移（米） | 19.453055 | 19.676961 |
| 模拟原始进度（米） | 20.056434 | 20.352365 |

分段jerk是请求点的描述代理，不是官方Comfort；原始米数不是逐场景归一化EP。均值变化与新增/恢复需共同判断，不把个别代理量或均分当成全部场景改善。

## 资源、成本和异常边界

使用vla-zt、tsinghua-zt3、tsinghua-zt4各8个不同物理GPU，共24卡。只释放逐PID/UUID核对过的原压力worker，原argv/environment保存在服务器私有目录；未停止真实/未知任务。

根据当前全局授权，tsinghua-zt2、tsinghua-zt6、rl-zt2均排除；Qwen-Flow IL主机显存不足不参与，zt5无法确认归属的活跃占用也未使用。
24个worker全部exit0；逻辑单候选12146，实际推理12146，复用scene0。完整分母、每条payload hash和匹配协议均由原入口校验，没有将基础设施错误填0或删场景。
worker实际时间跨度998.644秒，控制进程wall1092.389秒；累计worker wall19848.384秒，其中模型加载2149.711、前向13372.685、评分等待1562.209秒。累计worker时间不能当成端到端wall或独占GPU核计算。
准备阶段的一次redacted压力捕获回执输出发生变量名覆盖异常：第一个主机的自有暂停已生效，但尚未启动任何模型或评分。保留原私有capture与错误记录后，按精确marker接管暂停并完成其余主机释放；不重新捕获为另一份原始状态。所有评价worker退出后恢复三台原30分钟空闲监控策略，不创建新压力脚本、不强占真实任务卡。

完成后再次按PID/start-time核对：24个原评价进程全部退出，三台原监控均解除本任务暂停，保留原1800秒窗口。原监控恢复不代表独占GPU；后来进入的真实任务仍受保护。

## 结论与接下来的既定任务

本次仅增加2000个IL optimizer updates，相对49k的变化不能当成算法更换或RL提升；是否有稳定的后训练增益仍待两组正式RL。保留100k IL上限、五个DEV父模型选择、同父模型CANONICAL/SOURCE各50k，不依据本次Navtest临时换配置、追加seed或延长预算。

## 实际入口

本次单一operator：`/mnt/project/qwen-sourceflow-native-20261018/interim_navtest_IL051000_20261022/run_evaluation.py`；实际24份worker命令、source和物理GPU身份位于该目录`jobs/rank*/spec.json`，已有attempt拒绝重复启动。

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.qwen_sourceflow.analyze_il_milestones \
  --evaluation /mnt/project/qwen-sourceflow-native-20261018/evaluations/IL_update051000_NAVTEST_interim_20261022 \
  --reference /mnt/project/qwen-sourceflow-native-20261018/evaluations/IL_update049000_NAVTEST_interim_20261022 \
  --output /absolute/path/to/new_paired_summary.json
```

证据：[完整均值](IL051000_interim_NAVTEST.json)、[51k−49k配对](IL051000_vs049000_NAVTEST.json)、[执行成本](IL051000_NAVTEST_execution.json)、[检查点固定](IL051000_checkpoint_pin.json)、[评价登记](IL051000_interim_NAVTEST_registration.json)。没有上传权重、原始图片、场景数据或巨大cache。
