# Algorithms and scientific boundaries

所有独立算法由同一D44权重＋零LoRA增量＋新Adam开始，训练seed42。没有从资格或其他RL权重启动。

D0/D1/D2/D4/D5复用真实W3随机状态。每步Gaussian log-ratio对24坐标求和，再在W3平均损失；它是保存状态条件下的随机转移联合密度，不是最终物理轨迹边缘likelihood。创新不clamp，不继承source relative logprob clamp。当前、old、IL44 reference严格分离。

SafetyEP组内先硬安全feasible（NC=DAC=TTC=1），只给可行候选EP相对质量优势；全不可行组按三成本均值排序。raw优势在fresh全体有效scene×candidate上FP64共同std，cap5；全等十进制成本不造假优势。每个完整collection只更新一次自适应KL beta，M2内固定。

| 算法 | 与D0的区别 | 当前解释边界 |
|---|---|---|
| D0 | joint24 PPO clip0.1，native KL自适应 | 需用自己的B1 Navtest验证 |
| D0_CLIP020 | 只改clip0.2 | 不预设更大更好 |
| D0_CANONICAL_CREDIT | 改回canonical组内中心化信用 | 必要的SafetyEP对照，单独ID |
| D1 | 2epoch后可行质量中加入comfort和TRAIN GT endpoint | 不是GT作为策略输入 |
| D2 | 自己真实候选中严格安全且比固定B1高≥0.02，正clean-x和真实B1末步SmoothL1 | 离策略监督与PG ratio分开；G32追加只挖教师 |
| D3_FPO | MC4 CFM energy差的PPO ratio代理 | 明确不是精确likelihood |
| D3_AWM | 正优势权重matching＋参考保持 | 保守本地适配，不冒充完整官方signed AWM |
| D4 | 原IL R16+evaluation head监督on-policy因素/总分/排序/native事件 | 不要求post loss直接训练动作输出Linear；部署删R |
| D5 | r32文本＋last8 FiLM低秩＋动作I/O/position | 需单列表征漂移与遗忘 |
| D6 | 计划的分阶段组合 | 生命周期未完前拒绝调度，不用D0伪装 |

参考：[FPO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d43cc5692bf68944ee7cd31b97d0c11-Abstract-Conference.html)、[AWM](https://proceedings.mlr.press/v306/xue26c.html)、[AUGUstus](https://github.com/haha-yuki-haha/AUGUstus)。本轮不宣称完整复现AUGUstus Qwen3.5文本CoT-GRPO。直接复用的SimWAM vendor保留原许可与既有固定commit1433ee7e；新数学和适配清楚归属本项目。

D4事件标签取native at-fault collision/TTC时间与逐时刻non-drivable-body。首次TTC违规时刻不是最小TTC。缺标签应拒绝或mask，不填零冒充安全。事件全none的高准确率不能证明学会辨认稀有事故。

尚无本轮新RL完整Navtest结果，无法判断哪条路线提分。不存在“测试通过所以92完成”的结论。
