# 路线二：物理反馈搜索与源FM吸收

本目录属于独立三路线研究，公共起点由主agent锁定为成熟CANONICAL50k完整合并模型；不是旧DDP策略提取实验的恢复。当前已完成CPU接口/算术资格，真实共同候选bank、评分门槛、GPU及场景预算由主agent统一调度。`qualification_cpu_v1.json`中的synthetic评分仅测试搜索逻辑，**不是驾驶研究结果**。

## 实际复用与边界

- 复用`safe_flow_improvement/candidate_search.py`的4维有界CEM draw、方差下限、Hermite路径和quintic平滑时钟/侧向扰动。新增最小heading residual适配，保证零控制对原生8×3请求完全一致；不独立抖动8个waypoint。
- F1/F2/F3/F4都固定围绕共同8条中的第0条ODE请求，不根据不同反馈换seed轨迹；后24条为三轮各8条。F2/F3共用精英复用、平滑更新与界限，只改变排序信息。
- F2只访问原始scalar PDMS；instrumented PoisonMetrics资格会在读取NC/DAC/TTC等任何额外字段时失败。
- F3采用先可行、再高真实分；未可行时使用真实NC/DAC/TTC缺失、全未来越界比例和实际模拟动态量的有界搜索cost。它是搜索启发式，不改变保存的官方PDMS，也不声称无偏policy gradient。
- 所有非法提议都消耗K32预算，保留原因与`raw_reward=null`；不是官方0分。有限且范围内的原生pose请求必须经过真实控制器评分，不能将未执行的Hermite加速度当物理非法。精确重复可复用评分但单列次数，不算新发现。

## 真实过程字段审计

当前canonical批量评分器已保存实际模拟`simulated_states[41,11]`、车身坐标`[41,5,2]`、逐物理时刻`non_drivable_body_by_time[41]`、首次责任碰撞时间、首次TTC规则违规时间及原始进度。第一状态t=0不进入未来行动cost。`official_first_ttc_infraction_seconds`是首次规则违规发生时刻，**不是最小TTC**。

当前没有完整责任碰撞/TTC逐时刻mask，因此它们的available mask为False，不制造持续时长。首次事件时刻只作诊断，不因事故更晚而给予更好排序。道路/对象连续margin首版未启用；几何重叠不冒充责任碰撞。F3若缺失其必须的真实越界mask或模拟状态会报错，不填0解释为安全。

F4只在合格可行候选内，使用真实进度、平均速度、制动起点和jerk的固定TRAIN尺度。每轮最多6条质量提议＋2条合格新颖度提议；没有可用合格父候选时关闭该支路。未读取未来图像、导航修改或语义行为标签。

## 源FM吸收

外部候选只进入非负、场景内归一化的监督目标分布，不进入普通GRPO ratio。标签是实际评分的原生请求编码，不是控制器执行状态。teacher温度0.1以原始0—1分数为单位；默认最多4条、至少0.9分且比共同第0条reference提高0.005，XY ADE去重阈值0.2m。没有合格目标不伪造成功；所有过滤阈值需随公共正式登记固定。

`absorption.native_source_fm_loss`与原`SourceFlowPolicy.forward`使用同一个源scheduler：shift5连续sigma、`x=(1-sigma)*target+sigma*noise`、velocity=`noise-target`及原training_weight，独立插值noise/time。CPU真实tiny ActionDiT的loss与全部参数梯度逐项一致，并执行一次optimizer更新。没有复用旧DDP的Beta时间/相反velocity实现。

## 已执行命令与当前证据

```bash
cd /mnt/project/VLA-Drive-qwen-flow-three-routes-20261024
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.feedback_validate \
  --output reports/qwen_flow_three_routes_20261024/feedback/qualification_cpu_v1.json
```

该资格覆盖零控制、4D平滑proposal、全部F0-F4的32次预算、F2信息隔离、未知反馈拒绝、事件时间语义、TRAIN-only teacher/尺度、非负FM权重、源FM loss/梯度对照与实际优化器步骤。

真实发现能力、学生吸收、部署泛化和后续GRPO目前尚未完成，不可由本CPU资格判为支持。下一步使用主agent统一的真实K32bank做同预算CPU搜索，随后给S1/S2/S3分配同一LoRA/FM更新窗口并重新ODE采样。

## 真实资格v1发现并隔离的预筛问题

首次GPU2真实profile完成8TRAIN/4DEV/G4的ODE前后采样、4次GT FM optimizer更新及480个Adam状态/冻结hash核对。但其中F1-F4的全部24个外部提议被旧请求曲线加速度代理预筛掉，未真正进入官方仿真，也未完成外部teacher FM。该scene原始pose请求的Hermite代理max acceleration约18.31m/s²，而真实官方执行的NC/DAC/TTC/Comfort均通过，说明它不是已执行动态量。不能用profile总体PASS掩盖该机制未被实际测试。[旁路审计](qualification_native_v1_audit.json)

v1全部证据保留。正式搜索前做最小修正：旧请求曲线极值只记录为诊断，有限/shape/参数范围合法请求必须进入锁定的真实控制器；F3仍基于实际模拟状态的动态量。未改变官方评分或安全门槛，未给异常填0。需要新v2资格ID重跑外部搜索与吸收。最新CPU v3补充了这个接口语义检查。

正式学生预算由主agent锁定为S1/S2/S3各256 updates、B16、相同512 TRAIN人口和GT曝光，AdamW5e-5、8步warmup后constant。S2/S3 teacher系数在固定TRAIN梯度上校准一次，目标teacher/GT梯度比例1.0、bounds[0.05,2]，保存cosine和bound命中；不使用DEV成绩调权重。由于共同起点已是RL50k，必须同时报告S3−S0和S3−S1，不能靠GT-only S1退化制造表面增益。

## 真实资格v2：外部请求确已评分与反传

不可变源码`5075dc5`上的v2真实GPU2资格已完成，进程exit0。F1-F4各24条外部请求全部进入canonical评分，无请求代理误拒；共224条真实评分、40批scorer，4次GT＋外部请求FM更新，480个LoRA Adam状态均到step4，冻结模型/reference hash不变。被选为编码/梯度probe的外部严格可行请求与base的ADE为0.18846m，明确`not_improvement_qualification=true`，没有因此放宽正式teacher阈值。[聚合证据](qualification_native_v2_summary.json)

该单个搜索场景完全安全，F2/F3/F4的K32安全oracle均约0.842633，F1约0.848656、F0约0.846046。它不支持物理反馈的额外价值判断，也不能拿搜索最高分当学生部署成绩。8TRAIN/4DEV的小型新root评测只用于接口与有限更新检查，未进入正式模型选择。v1误筛失败证据保留，所有资格权重均不用作正式学生初始化。

最新CPU v4另通过实际tiny原生LoRA/source-FM的READY保存与恢复：后续参数和Adam moments逐位一致。正式入口有预算锁；SIGUSR1/SIGTERM只请求完整optimizer后保存READY退出，评估在下一原子scene边界退出并可复用已完成记录，不把中断写成COMPLETE。
