# 统一调度的较长 RL 对照（2026-11-01）

本轮解决的是短训练、不同调度与方法效果混在一起的问题。历史九组试验保留，不能将新的成套配置与旧恒定学习率差异直接归因为 cosine 的独立收益。当前没有新的确认提升，也没有达到92。

## 为什么需要更长

当前每次采集8个新场景，每场景8个候选，复用4次参数更新。1k/3k仅见2000/6000个场景，约全量103288的1.94%/5.81%。10k为20000场景（19.36%），50k为100000场景（96.82%），完整第一遍需51644次更新。覆盖比例不是已学会比例，组内候选数也不增加新场景覆盖。

旧3k完整NAVTEST12146、invalid0：Control91.013711，Dense90.948854，Saturated90.911018。Dense相对Control−.064858，配对95%CI[−.186583,.040992]。它们尚未证明优于对照，也不能由这么短的曲线断言长期无效。

## 四条独立路线

| ID | 设计 | 要回答的问题 |
|---|---|---|
| SAFETY_G8_CONTROL | 原SafetyEP W3，统一新调度 | 相同预算下的可信基准 |
| STANDARD_INVERSION_G8 | 只抵消有真实较低分正优势见证的倒置优势对 | 已找到安全高分轨迹后，是否存在可纠正的错误信用排序 |
| STANDARD_COMBINED_G8 | 三类互斥场景分别应用DenseDAC、成功饱和信用、倒置抵消 | 全失败同代价、成功组无学习信号和真实排序倒置能否同时改善 |
| STANDARD_ENDPOINT_LR02_G8 | ODE前缀、ramp AR1终点探索、条件joint24 PPO，峰值LR为W3的0.2 | 更好的候选覆盖能否在较小、较充分的更新下转化为部署收益 |

组合信用全部根据原Safety优势判门，只允许每场景激活一支，保留原批量std。Dense使用真实越界时长，为失败轨迹提供相对指导，不能把它当作已经找到成功轨迹。饱和分支只处理全可行EP1而canonical仍有差异的组。倒置分支保留没有较低分正优势见证的正常相对负优势。终点方法仍有明显负面先验：历史1k未胜对照，因此这是一项有明确机制问题的检验，不能预先宣称有效。

共同起点是完整World IL70、Adam/RNG42从零开始；全量103288均匀首遍、G8、8rank、4inner；可训练52,707,840参数（动作LoRA5,242,880 + 场景/预测memory47,464,960），Qwen特征冻结，current memory每次inner重建。不是只训练最后轨迹输出层。

## 固定调度与评测

按已完成optimizer更新u计时：0到1000线性从峰值的0.001升到1；其后按cosine在u50000到0.1。W3峰值action5e−5/memory5e−6；终点方法显式使用1e−5/1e−6。两组参数使用相同乘子。第一次更新用u0倍率，指标记录更新前实际LR和更新后next LR。原生恢复保留Adam、所有RNG、场景游标、未完成rollout、scheduler epoch与固定协议，2k/5k/10k阶段不重置学习率周期。

首段预算每臂10000，2k/5k为观察门槛，不因小幅负差或CI跨零淘汰；异常/严重失败仍需处理。后续按完整配对评测与长曲线安排20k/30k/50k，10k以后每2k评测。10k仅到cosine下降的早段（next倍率0.92714），不得把该结果视作完整衰减周期的上限。

关注完整PDMS、零分修复/新增零分、固定IL70高分队列贡献、NC/DAC/TTC安全转移、实际优势与被PPO截断的方向、实际LR和训练开销。NAVTEST不反馈场景级训练采样；历史DEV并非独立验证集。初段曲线用于判断机制，不用多种子重复。

## 相关实现提供了什么依据

[DPPO机器人RL官方furniture微调配置](https://github.com/irom-princeton/dppo/blob/main/cfg/furniture/finetune/one_leg_low/ft_ppo_diffusion_mlp.yaml) 使用actor warmup10、峰值1e−5、最低1e−6，但配置cycle10000/实际1000训练轮，计步单位与本项目不同。[DPPO can微调配置](https://github.com/irom-princeton/dppo/blob/main/cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp.yaml) 最低值与峰值均1e−4，不能说它证明所有机器人RL都需要衰减。本轮采用固定warmup+cosine是受这些实现启发、满足统一可比较性的待检验设计；1000/50000/.1并非论文证明的通用最优超参数。

## 实物与运行状态

注册和实时状态目录：`/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101`。`REGISTRATION.json`、`RUNS.json`、原生owner/8rank日志及完整评测收据是运行证据；本文不把登记视为实际启动。原HTML持续更新：`/mnt/project/qwen-world-rl-optimization-20261030/report.html`。ABCD四组RL保持用户终止状态，不恢复；B/D IL资源优先保护。
