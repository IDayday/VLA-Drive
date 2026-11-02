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
| D6 | 已接入的分阶段组合 | 通过同源码真实阶段恢复资格前拒绝正式调度 |

参考：[FPO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d43cc5692bf68944ee7cd31b97d0c11-Abstract-Conference.html)、[AWM](https://proceedings.mlr.press/v306/xue26c.html)、[AUGUstus](https://github.com/haha-yuki-haha/AUGUstus)。本轮不宣称完整复现AUGUstus Qwen3.5文本CoT-GRPO。直接复用的SimWAM vendor保留原许可与既有固定commit1433ee7e；新数学和适配清楚归属本项目。

D4事件标签取native at-fault collision/TTC时间与逐时刻non-drivable-body。首次TTC违规时刻不是最小TTC。缺标签应拒绝或mask，不填零冒充安全。事件全none的高准确率不能证明学会辨认稀有事故。

尚无本轮新RL完整Navtest结果，无法判断哪条路线提分。不存在“测试通过所以92完成”的结论。

D6追加适配：FiLM last8增量从初始化预置为零，8epoch前梯度清空且不创建Adam状态；R readout在4epoch前同样不更新。60/25/15训练分层只读本run已提交的TRAIN nominal历史，池未具备时回退uniform并记录实际采样来源。历史教师最多8192个场景，各一条严格合格本run早期B1轨迹；不是其他RL权重/外部搜索/测试反馈。首见nominal是self-history基线，不冒称完整IL44普查。阶段III的时序成本明确只用native全窗口DAC采样mask作0.05有界tie-cost；NC/TTC首次事件用于R监督，不冒充连续最小TTC。


## 日志含义与已执行资格

D3-FPO真正优化的是`fpo_*`对应的CFM-energy ratio代理；未加前缀的Gaussian ratio/clip仅为同一策略原生转移诊断。D3-AWM的`pg`日志名沿用容器，但其值是正优势matching与clean-x参考保持，不是Gaussian PPO，也没有PPO clip。不能用AWM的native clip诊断推断matching更新被裁掉。真实B32/G16 FPO资格第一inner的energy-proxy ratio=1、log-ratio min/max=0；不是只检查另一条Gaussian重算。

D1/FPO/AWM已分别完成两卡、32个不同TRAIN场景/G16/M2的10次实际更新，随后才允许进入八卡长训队列；资源未准入不算已启动长训。D4/B32资格继续运行，历史八卡小batch post资格不冒充新的B32结果。

另用D1实际保存的32个TRAIN场景、512条候选与原始GT运行了阶段II标签/优势桥接：全部有限，组内raw和误差4.996e-16，有安全候选时违规候选正优势为0，未增加采样、评分或optimizer更新，见[D1真实标签桥接](D1_STAGE2_REAL_LABEL_CREDIT.json)。该样本没有全不可行组或低行程GT；这些边界由小型数学测试覆盖，不能声称真实32场景覆盖了它们。此证明不等于D1阶段II已经训练。

D2既有单批次加权auxiliary/PG梯度比约0.00168，说明这个batch的教师监督较弱；它不是全人口的校准结论，也不是已证明有效的吸收能力。正式0.5epoch后再按真实已保存batch与B1结果判断，不偷偷修改已登记lambda。

`scope.base_frozen=true`是继承的宽泛标记：真正冻结边界以完整`trainable_names`和optimizer角色为准。D5显式训练独立克隆的action I/O/position，D4显式训练克隆的readout；永久reference保留原IL参数。不能把这个标记解释为这些IL接口参数也被冻结。
