# 53k IL全量Navtest与51k配对结果（2026-10-23）

**53k IL单候选canonical NAVSIM v1.1 PDMS为87.016025，相对51k的86.675474变化+0.340551分。** 未达到92，仍差4.983975分。 当前是IL阶段，不是Flow-GRPO收益。

## 实际进度与固定评价合同

启动核查时正式IL八个rank一致到53,496/100,000步，最近完整原生恢复点为53k。此次按用户要求，在评价结果出现前固定53k权重、完整optimizer及RNG资产；同文件系统hardlink保留原生文件和seal，不修改训练源码或父模型选择规则。

完成后进度快照：2026-10-23T06:13:56.081504+00:00，八个rank一致到 **53,594/100,000步**，累计1,715,008次scene曝光。最近500个完整update取各rank最大耗时后的均值为13.648秒/update；按此实测速度，剩余update计算约175.9小时，另需保存、评价，且不包含未来资源变化。

正式RL尚未开始。20k/40k DEV已完成，既定60k/80k/100k DEV继续排队；RL父模型仍只在登记的五个DEV候选中选择。53k不是新增父模型候选，不能利用此次Navtest调参或选权重。

- 全量12146场景/136 logs；seed2101；单候选10步source shift5 ODE；canonical NAVSIM v1.1。
- 当前左/前/右三视图、导航、合法当前ego；原生Qwen完整token接口和源ActionDiT 8×3输出。
- 与51k采用同一d3a7ce8不可变评价快照，所有source_files、输入、精度、后处理和评分合同相同。直接使用严格同源码配对，没有新增兼容例外，也没有重复无关数值验收。
- 无搜索、GT选轨、scorer选择、best-of-N或额外seed。中期Navtest已多次观察，后续不再称完全盲测。

## 完整Navtest成绩（0—100）

| IL检查点 | PDMS |
|---|---:|
| 20k | 80.370064 |
| 40k | 85.786557 |
| 49k | 86.810426 |
| 51k | 86.675474 |
| 53k | 87.016025 |

| 分项 | 51k | 53k | 变化（百分点） |
|---|---:|---:|---:|
| NC | 97.838795 | 98.089906 | +0.251111 |
| DAC | 95.175366 | 95.274164 | +0.098798 |
| TTC | 93.759262 | 94.055656 | +0.296394 |
| EP | 81.467936 | 81.722655 | +0.254719 |
| Comfort | 99.967067 | 99.950601 | -0.016466 |
| DDC | 98.386300 | 98.411000 | +0.024699 |

53k−51k的scene加权log-cluster bootstrap（2000次）95%区间为 **[-0.240971, +0.922310] PDMS点**。区间跨0，不能据这一相邻检查点差值断言持续改善或持续退化。 不包括训练/推理seed不确定性；bootstrap重采样不算新增训练seed。

## 失败、恢复与正负收益

- 零分：826→784；原零分恢复298，原非零新增零分256。
- NC/DAC/TTC联合完全通过：10851→10905；原联合安全人口新增违规357，原不安全人口联合恢复411。NC=0.5不视为完全通过。
- G+贡献3.048145点，G−贡献2.707593点，净变化+0.340551点。
- 任意提高/下降4210/3917场景；达到0.005原始score幅度的提高/下降2863/2722场景。任意负差不等于严重事故。

| 安全项 | 51k违规 | 53k违规 | 新增 | 恢复 |
|---|---:|---:|---:|---:|
| NC | 275 | 244 | 49 | 80 |
| DAC | 586 | 574 | 220 | 232 |
| TTC | 758 | 722 | 170 | 206 |

## 请求轨迹与模拟后果

| 描述统计 | 51k | 53k |
|---|---:|---:|
| 请求终点前向位移（米） | 19.024906 | 19.275085 |
| 请求分段平均速度（米/秒） | 4.962779 | 5.014946 |
| 请求分段jerk RMS代理（米/秒³） | 4.471431 | 3.948379 |
| 模拟前向位移（米） | 19.676961 | 19.782191 |
| 模拟原始进度（米） | 20.352365 | 20.434188 |

请求点的jerk是描述代理，不是官方Comfort。模拟米数也不是归一化EP，不用单个代理量代替真实PDMS。

## 资源与实际成本

仅使用vla-zt、tsinghua-zt3、tsinghua-zt4各8张不同物理GPU。初次资源快照仍有GPU占用，随后两次admission采样均显示三台24卡空闲；本任务未停止、暂停或迁移任何既有进程，也未改原压力监控策略。

tsinghua-zt2、tsinghua-zt6、rl-zt2按当前授权排除；主IL机器显存不足不参与，zt5仍有无法确认归属的高GPU活动，不使用。

24个worker全部exit0，且按PID/start-time再次核实已退出。逻辑单候选12146，实际推理12146，复用scene0；没有把异常填0或从分母删除。所有token/log成员、payload hash和协议均由既有完整汇总入口校验。

worker时间跨度1011.885秒，operator wall1093.925秒。累计worker wall19644.267秒，其中模型加载2088.497、前向13047.761、评分等待1564.280秒；累计worker时间不是端到端wall，也不是独占GPU核计算。没有新增评分/数值矩阵或重复使用先前checkpoint的结果充作本次推理。

中途控制服务重启使原前台receipt owner退出，模型worker保持独立运行。恢复前核对：8个worker已成功退出，其余16个PID/start-time匹配且仍在原评价模块；没有重新派发任何模型worker或重算已完成scene。新owner只接续原receipt收集、全量校验与配对汇总，保留原owner起始时间和恢复收据，控制中断不从总wall中扣除。该事件没有改变权重、trajectory、评分、seed或评价人口。[恢复收据](IL053000_controller_recovery.json)。

## 解释边界与既定后续

本次相对51k的总分回升伴随NC/DAC/TTC和EP均值改善，Comfort小幅下降；新增零分256个、恢复298个，净减少42个零分。固定seed下的配对区间仍跨0，不能称为稳定提升。相对历史49k，总分均值+0.205599点，NC +0.082332、DAC -0.098798、TTC -0.049399个百分点，零分784个仍略多于49k的781个。因此本次超过49k总分均值，也不意味着安全分项全面超过49k。49k比较为历史完整汇总的描述统计，不替代预先固定的53k−51k配对分析。

这仍是继续IL训练后的单候选结果。两次相邻检查点只相差2000个optimizer updates，不能据此归因新的RL算法，也不能代替正式RL相对自身IL的实验。保留100k IL上限、五个DEV选择规则和同父模型两条50k RL，不按中期Navtest更改配方、增加seed或追加预算。

## 真实执行与证据

首次operator：`/mnt/project/qwen-sourceflow-native-20261018/interim_navtest_IL053000_20261023/run_evaluation.py`。控制中断后由同目录`recover_evaluation_owner.py`取得原互斥锁，只接续收据汇总，没有重复派发。每个GPU的实际命令/source/UUID位于该目录`jobs/rank*/spec.json`，重复attempt被拒绝。

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.qwen_sourceflow.analyze_il_milestones \
  --evaluation /mnt/project/qwen-sourceflow-native-20261018/evaluations/IL_update053000_NAVTEST_interim_20261023 \
  --reference /mnt/project/qwen-sourceflow-native-20261018/evaluations/IL_update051000_NAVTEST_interim_20261022 \
  --output /absolute/path/to/new_paired_summary.json
```

证据：[完整分数](IL053000_interim_NAVTEST.json)、[53k−51k配对](IL053000_vs051000_NAVTEST.json)、[执行与成本](IL053000_NAVTEST_execution.json)、[固定检查点](IL053000_checkpoint_pin.json)、[预先登记](IL053000_interim_NAVTEST_registration.json)、[训练进度](progress_NAVTEST_IL053000_20261023.json)。只提交汇总，不上传权重、原始数据或巨大cache。
