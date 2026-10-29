# 2026-10-29 16:25 UTC：最新完整Navtest与TIME原生迁移

所有数值为canonical NAVSIM v1.1、12146场景、invalid0、seed2101、单候选10步shift5 ODE。训练seed42。未完成模型不发布部分均分。本目录是阶段快照；完整来源及文件hash在scores.json。

| IL版本 | 已完整检查点 | PDMS |
|---|---:|---:|
| 原始Qwen-World L1，最新 | 96k | 88.618573 |
| 原始Qwen-World L1，已测最高 | 70k | 88.792910 |
| 优化轨迹V12，最新已完整 | 22k | 83.323871 |
| PHYSICAL物理加权 | 9k / 10k | 83.599990 / 81.299422 |
| TIME_RESIDUAL时间分布加权 | 5k | 79.107640 |
| DELTA增量损失 | 7k | 78.496596 |
| HUBER | 6k | 78.000602 |

V12 27k正在评测。不同训练步数不能直接作为损失方法的匹配排名。5k原L1=79.866345，PHYSICAL=80.101589，TIME=79.107640，DELTA=79.186748，HUBER=72.150853。PHYSICAL 10k=81.299422相对原L1 10k=80.531938提高0.767484点，但低于自身9k，不能宣称稳定优势。四臂共同10k筛选规则不变，没有依据5k擅自终止。

| Qwen-World RL训练范围 | 已测最高检查点与PDMS | 固定50k终点 |
|---|---|---:|
| 动作attention LoRA | 20k：90.580893 | 89.905612 |
| 动作attention LoRA＋scene/predictive memory，Qwen冻结 | 33.3k：90.741310 | 90.628892 |
| Time MLP/AdaNorm LoRA | 40k：89.838968 | 89.508000 |

本次没有新增RL分数。三组已完成50k。实际共同IL父模型是40k=87.972382，Memory峰值/终点相对该父模型分别+2.768928/+2.656510点。不能用更晚70k IL作为这些RL的真实父模型。峰值不等于固定终点；所有RL曲线均非单调，尚未达到92。

更早Qwen-Flow是独立模型族，已核实IL峰值80k=88.157880、RL峰值24.6k=90.689893，不能与当前Qwen-World混名。

## TIME_RESIDUAL迁移验收

用户要求从拥挤节点迁移提速。原训练在tsinghua-zt3与V12 IL共享，保存原生update005390后正常退出；vla-zt3八卡从5391继续，保留全部模型、八rank Adam/scheduler/RNG、scene cursor、辅助采样器与训练源码9aa888c11426a673a0b9d21f45e7ff44a42b287c。global batch32与100k scheduler未改。核对八rank连续无重复/缺步、optimizer未skip、下一批scene与保存cursor相符、LR连续；原生父检查点完整硬链接保留。

新CPU接管包装器最初漏returncode属性，导致CPU调度器退出；GPU原生训练继续。旧失败证据归档保留，补丁e125640ac4903ea5bb756e4d55f684d0ffe1f399只补CPU调用合同并恢复原任务接管，没有重启GPU或改科学源码。主/TIME调度器恢复，其余三臂owner未重启。有限迁移验收PASS详见time_migration.json。

迁移前160步平均4.759068秒；恢复后跳过前5步的30步平均2.999412秒，中位2.743030秒，短窗口吞吐约1.59倍。场景不完全相同且资源共享，不能当长期固定加速保证。恢复核对完成时已经运行到5614，不以单步最快值估算总工期。

旧评测、未完成评测和V12训练均保留；未占用撤销授权的tsinghua-zt4。没有新增seed、训练方法或评分协议。
