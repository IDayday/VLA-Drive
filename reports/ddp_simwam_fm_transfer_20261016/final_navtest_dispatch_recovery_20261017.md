# 固定A/B终点Navtest：资源准入失败后的显式恢复

2026-10-17 23:41 UTC起，A的SFT20k/RL20k、B的SFT100k/RL20k共四个固定端点，已在授权的training-vla-zt八卡启动单候选完整Navtest。每个模型两个独立scene shard，各评价原生均匀10步与shift5。每网格仍为12146场景/136 logs、seed2101；总计97168条逻辑模型预测，额外入口资格检查单列。

这次只是恢复资源调度，没有修改训练、权重、推理网格、模型输入、概率/损失或评分代码。官方v1.0补评分读取同一已保存请求轨迹，不为换评分版本重新采样。该记录表示已派发，不表示评价完成。

## 失败边界与保留内容

原final_handoff_v1在两条RL20k和全部36项DEV/TRAIN阶段观察完成后，成功发布了四个固定端点的LOCK。随后第一次本机派发遇到GPU可用状态变化，`admit_devices`在模型worker启动前拒绝资源准入。该命令约12.3秒；所有失败日志、原路由计划和LOCK保留。

恢复前实际确认：原owner已经退出；没有final worker收据、模型预测或有效scene输出；唯一已创建的A_SFT输出目录为空。不能把原计划的15张卡当作已部署资源，也没有将此次失败算作模型/算法失败。

独立的final_handoff_v2_capacity_retry_20261017只显式复用已完成的LOCK阶段与同一配置，保存父失败与锁定文件hash，不复用未执行路由。单owner文件锁保持，原ad73控制器、d584派发/分析、45d推理快照不变。所有旧计划与失败证据没有覆盖。

## 实际资源与占卡脚本

重新对八台授权主机做两次实际观察，按物理UUID去重。当前只有本机八卡满足足够显存、可见owner和CPU余量检查；其他主机的GPU owner不可见或容量不足，保持原任务不动。临时主机、vla-zt2及其别名没有使用。

本机GPU4–7的四个owner准确对应`/tmp/zt-idle-pressure-v1/agent.py --worker <UUID>`。读取脚本后，核对PID/start time、argv、GPU UUID、脚本hash、工作目录和无子进程，私有环境仅在本机受限目录保存。只向这四个已确认占卡worker发送SIGTERM并确认退出，没有向真实或未知任务发送信号。

现有压力监控进程保持运行。按当前授权文件的规则，手工终止后由它重新积累30分钟负载观察，再判断是否恢复；未立即重建fillers，也没有设置永久暂停。该规则优先于旧skill中立即恢复原填充进程的通用建议。

| 模型 | 主机 | GPU | shards |
|---|---|---|---:|
| A_DDP_TIME_SFT | training-vla-zt | 0,1 | 2 |
| A_DDP_TIME_RL | training-vla-zt | 2,3 | 2 |
| B_SOURCE_ACTION_DIT_SFT | training-vla-zt | 4,5 | 2 |
| B_SOURCE_ACTION_DIT_RL | training-vla-zt | 6,7 | 2 |

实际worker身份、原失败时长、锁定hash及恢复链见[机器记录](final_navtest_dispatch_recovery_20261017.json)。不把共享worker wall写成独占GPU时间；推理完成后以所有scene payload、rank退出和成本账本共同验收。

## CPU版本评分与汇总

原CPU owners已到有限等待截止时间并退出：三个历史模型全部补评分完成，五份独立/配对分析完成，四个当前模型尚未派发。没有运行中的CPU子任务需要终止。

新的scoring_dispatch_v2_final_retry只接续四个从未启动的当前模型，仍使用b506评分快照；analysis_handoff_v3_final_retry只接续八份未完成汇总，仍使用d8bf分析快照。两个私有调度脚本均原样复用，改变的只有待办清单、调用身份、截止时间和已完成收据的显式继承。原完成评分、候选和摘要没有重跑或覆盖。

已运行的既有入口仍是：

```bash
python -m scripts.analysis.fm_transfer.final_handoff --control /mnt/project/fm-transfer-20261016/final_handoff_v2_capacity_retry_20261017 --hosts training-rl-zt2 training-vla-zt training-vlawm-tsinghua-zt training-vlawm-tsinghua-zt2 training-vlawm-tsinghua-zt3 training-vlawm-tsinghua-zt4 training-vlawm-tsinghua-zt5 training-vlawm-tsinghua-zt6
```

该控制器已经在运行，不能重复执行。若再次异常，应先核对当前owner和逐worker状态，再显式接续合法边界，不能把本记录当作无条件自动重试授权。

正式最终结论等待完整Navtest；不使用当前部分人口均分，不用DEV、历史控制组或两个网格中较高值代替四个固定端点的结果。
