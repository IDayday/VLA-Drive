# 唯一尺度候选的证据接线：完整TRAIN主审计已到，逐update派生待到

状态：`PRIMARY_COMPLETE_DERIVED_PENDING`，尚未封最终decision，implementation/profile/formal training均未批准。本文件不修改已封存 `b1a8cf49` 的唯一候选及其准入条款，也不增加阈值、系数、variant或数据人口。

只读取已完成小摘要及327,728B的合并RESULT，SHA256 `0a6a84d3652122aeb69760b5237a56c71fec269c16b4066e78a0ddb05876d8cc` 已验证。选定事实与来源记录在 `PRIMARY_EVIDENCE.json`；原大反馈、每ledger紧凑结果、模型、DEV/Navtest均未读取。最终decision将另绑定Lead交付的单次派生RESULT。

| 已封提案§4条款 | 当前证据 | 此时判断 |
|---|---|---|
| 完整消费人口及独立核验 | 4096边界，131,064 scene呈现，524,256候选；80,933非零C1组精确对齐；原分析独立正确性/资源批准与最终COMPLETE匹配 | 已满足主审计人口条款；不是新variant实现/训练批准 |
| 每update N/std/scale、零组及固定分位 | 已有完整候选幅值分布；逐update派生尚未交付 | 待派生；不能用全程RMS或3584代替 |
| 每update舍入/epsilon/常数组异常 | 主审计重建最大误差1.11e−16、舍入残差最大1.15e−7；全程舍入协方差份额约3.62e−9 | 全程结果不能排除个别update舍入风险，等原预登记的逐update界 |
| 四窗口、EP-only与安全支持分层 | 四窗口均有C1和D/C/T区分；逐update事件/EP-only尺度对比待到 | 总体支持已覆盖四窗口，缩放不对称仍待定 |

人口是101,592唯一scene、1,176 log；72,120 scene一次、29,472 scene两次。四窗口非零C1呈现依次20,168/20,311/20,184/20,270；对应D/C/T区分呈现分别269/14/163、245/10/149、158/5/115、119/9/93。因此3584的无事件现象不能推广全TRAIN。

79,586/80,933=98.34%的非零组为EP-only，支持“非零组计数大多来自进度”。但优势值的带符号协方差代数分解为gated EP35.80%、gated TTC48.43%、gated Comfort15.78%，并非进度占全部能量。每一gated项包含NC×DAC乘法门控，因此TTC栏也不能解释成TTC自身的独立因果份额，更不是参数梯度份额。稀有大差值具有实际存在的量级风险，不能用数量占优替代它们。

全候选|A|中位数0.0002162457、P90 0.0016427040、最大0.75。非零组内4候选均保留，共323,732项，其中真正A非零为323,724项；其余8项是有效组内零A。所有候选仍属于未来每update normalizer人口，不能只保留非零候选。

D/C/T区分在同一C1候选上覆盖764/37/485唯一scene及395/31/294 log；这些是对应安全区分scene的支持。未来派生的“含C/D/T的update”层内unique scene/log会同时包括同batch其他scene，不能把它们误当新增安全事件样本。终局float32同分且安全事件非空的5,960对中有3,526对时序/严重性不同，覆盖963唯一scene；这是过程信息被终局折叠的证据，此处不自动转去过程训练或用物理时间指定Flow步。

独立设计审阅 `rollout_scale_design_correctness_v1/DESIGN_REVIEW.json` 已为 `PASS_DESIGN_MATH_EVIDENCE_PENDING`：8项合成CPU数学通过，没有实际分布式collective、模型或训练验证。其待补完整TRAIN/逐update证据与本接线一致。后续只能回答“已值得实现并交独立正确性及有限profile审阅”或“具体哪条既定证据仍缺”；不签训练批准、不把clip0.2或归一化描述成安全保证。
