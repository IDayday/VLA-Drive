# 一次限定的KEEP/REPAIR学习修复对照

原六组seed42及三组seed43均已完成共同24epoch、9168更新。原损失、结果和检查点完整保留。本对照属于看到训练充分性诊断后的有限假设检验，不冒充初始预登记主表。

固定B训练曝光中，KEEP=190062，REPAIR=2969，合计193031有效曝光；REPAIR只占1.538%。固定训练诊断集上，MIN_FUT的REPAIR目标ADE从0.2536m降到0.2022m，KEEP修改量约0.00585m。共同终点的未来关系时间置换几乎未改变规划结果。这支持检验“稀少修复梯度被KEEP淹没”，尚不证明原因。

只新增MIN_NONE/MIN_CV/MIN_FUT seed42三个匹配小头，仍从相同初始化开始，B人口、24epoch、实际输入池、所有目标、模型、优化器、FP32和采样顺序相同。唯一变化是对原final_trajectory_loss的场景贡献乘固定训练统计权重：KEEP=193031/(2×190062)，REPAIR=193031/(2×2969)。保持每个optimizer batch的原有效场景分母，不做每microbatch类别归一化，不添加head、gate或部署条件。

原六组才是LOG×MIN的主比较。新三组只回答一次针对性修复是否改善稀有REPAIR学习及未来信息的增量，不能与未加权LOG作纯目标语义归因。额外上限为3×3600 GPU进程秒，CPU训练/评分预留8小时，总预算不增加。共同24终点做NATURAL与RECOVERY canonical；6/12/18只保留原生损失观察和检查点，不根据Navtest选取模型。

`--repair-class-balance`要求独立不可变训练曝光收据并写入checkpoint身份。缺失、错误fold/标签/计数、LOG/P使用均拒绝。默认不开启，历史路径不变。此为唯一额外学习修复；若仍缺乏未来价值证据，不继续调权重或扩大模型。

部署前后路径没有增加状态输入或启停门控。真实64场景连续48步与第7步中断后恢复的模型、optimizer、scheduler、RNG、数据位置和曝光逐位一致；不开启新权重时也与旧源码的48步训练状态逐位一致。见[evidence/BALANCED_RESUME_ACCEPTANCE_v14.json](evidence/BALANCED_RESUME_ACCEPTANCE_v14.json)。14项针对性测试通过。

三个共同24终点各做完整NATURAL/RECOVERY行为分析；MIN_FUT另做两种固定关系干预×两种分布共4项。此为同一稀少修复假设的结果解释，不新增训练组、不据干预结果选择部署开关。
