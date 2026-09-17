# 采样、概率与损失约定

DDP 原 flow 为 `x(t)=(1-t)ε+t·action`，`v_t=action-ε`，t 从 0 增至 1。锁定 Flow-GRPO 源 sampler 使用递减 sigma；对应 `sigma=1-t`、`v_sigma=-v_t`、`d_sigma=-dt`。动作头 bucket 始终为 `int((step/K)*num_timestep_buckets)`，不把 sigma 当作 DDP bucket。

对 `t=0,...,(K-1)/K`、`dt=1/K`，令 `d=t`，首步取 `d=dt`，`g=η√((1-t)/d)`。转移为：

```
mean = x · (1 - g²/(2(1-t)) · dt)
       + v_t · (1 + g²t/(2(1-t))) · dt
std  = g · √dt
x_next = mean + std · z,  z ~ N(0,I)
```

mean/std/log-prob/ratio/KL 至少 FP32，网络可 BF16。采样与重算使用同一 std；最后一步仍为有密度的 Gaussian，不另换 deterministic endpoint。`η=0` 只用于原 ODE 推理，没有虚构 log-prob/KL。公式由固定上游文件作独立数值 oracle，并与 FP64 Torch Normal 对照。

rollout 保存 `[B,G,K+1,H,D]` 完整链、所有 old elementwise log-prob、mask、观测、seed、参数/奖励版本、物理轨迹、奖励及 advantage。更新固定同一 `x_t,x_next`，重新执行当前 VLM 和动作头；绝不重新采样下一状态或用 GT 替换候选。

默认 `flow_grpo_dimension_mean` 是 H×D 维度平均 log-prob 的官方 surrogate。`exp(mean(logp_cur-logp_old))` **不是联合密度的精确 importance ratio**。可配置的 `joint_sum` 使用联合 log-prob；三种策略和 KL 都必须使用同一归约。配置和 checkpoint 记录该区别，不能在恢复时混用。

reference 是从初始化 SFT 复制的完整动作依赖模型：独立 Qwen 视觉/语言、embedding、history/query、projection、动作头。它不共享 actor storage，不接收 current features，也不靠 disable_adapter 实现。与动作分布无关的辅助 decoder 不需要执行。

每个 scene 内按 G 个真实 reward 的 population std 计算 detached advantage；有效全等组保持零 policy advantage，但仍保留 reference 和 SFT loss。PPO clipped surrogate 默认 epsilon=0.02。conditional transition KL 为相同保存状态/观测下的 Gaussian `KL(current || SFT reference)`，使用实际 transition std；它不是最终轨迹边缘分布的 KL。

```
loss = mean_scenes(mean_valid_GK(clipped_GRPO))
     + 0.01 · mean_scenes(mean_valid_GK(conditional_KL))
     + 0.1  · mean_replay_scenes(original_SFT_loss)
```

原 SFT loss 每个 replay scene 只计算一次，保留原内部重复扩散样本及辅助权重，不乘 G×K。action-only 只保留源训练的动作监督。单卡/多卡以相同有效场景批次为基准：action-only 的 rollout seed 和 SFT replay 随机数绑定消费位置/scene key，避免 rank 数改变测试输入。精确 resume 仍要求原 world size。

reward 使用当前本地官方 NAVSIM v2 `pdm_score`，每条候选独立与同一官方参考轨迹比较；不把 G 条轨迹合并为改变分母的 proposal 池。CPU spawn worker、缓存和候选排序不改变逐条含义。零分是合法数据，评分异常/缺失 cache/timeout 则整批失败并保留失败 rollout。

训练 reward 版本明确为 `navsim_v2_one_stage_single_scene`：没有邻帧候选时 two-frame comfort 缺失，按官方 finalizer 的缺失值权重规则处理。部署验证使用原 one-stage aggregation 和固定 held-out navtrain 场景，存在邻接场景时调用官方邻帧 aggregator。二者语义区别在报告中保留，不将单场景 reward 宣称为全数据集邻帧 EPDMS。

future trajectory/image/depth 和周车未来信息只允许进入监督或 evaluator。policy observation 白名单仅 image/lang/state/token；不缓存可训练的旧条件用于 actor 更新。navtest 从训练、噪声校准和模型选择中完全排除。
