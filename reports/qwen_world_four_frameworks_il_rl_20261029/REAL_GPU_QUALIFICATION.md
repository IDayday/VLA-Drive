# 真实八卡资格与恢复比较器修复

2026-10-29 UTC 状态。四组均使用原始完整模型、共同七项损失、global batch32、FP32存储/BF16前向，执行连续4次有效更新，以及从原生update2恢复至4。全部8个训练invocation退出0，所有rank optimizer skipped=0。资格权重不用于正式初始化，不能把这些4步记为IL100k。

|架构|连续更新3/4 wall秒|连续update4 rank0 allocated GiB|原生恢复训练|
|---|---:|---:|---|
|A|4.221 / 3.950|见原始metrics|2→4完成|
|B|4.477 / 3.933|见原始metrics|2→4完成|
|C|6.021 / 4.963|21.95|2→4完成|
|D|7.185 / 6.612|见原始metrics|2→4完成|

这些短窗口包含初始化、不同机器负载及保存影响，不作为正式长期ETA，也不是架构速度的严格匹配排名。完整DDP/ZeRO2 phase profile另行登记。

科学快照为 `fdaff809652370328e45c0d996ad706e1af2fc0c`，未热修改。A/B/C实际梯度审计中全部new_modules、readouts、预训练vision及language/merger张量均有非零梯度；D审计在原始资格目录中保留。2026-10-29 23:10 UTC读取八rank日志，对四组恢复update3/4的losses、gradient norm、LR、有效更新计数、cursor、scene IDs和后置scene选择逐字段比较，均与连续训练精确相同。

初次CPU checkpoint比较全部停在 `optimizer_state_dict.loss_scaler`。模型FP32权重及scheduler比较在此之前已经通过。该字段是DeepSpeed `LossScaler`对象，两个独立torch.load对象默认按对象身份比较，即使保存字段相同也不相等。实际A两端均为 `{'cur_scale': 1.0, 'dynamic': False}`。

修复仅在资格比较器：要求相同的具体LossScaler/DynamicLossScaler类，再递归严格比较全部`__dict__`字段，不跳过scaler、Adam或任何tensor。真实DeepSpeed独立序列化、scale/history/type不匹配拒绝等3项测试通过。原失败RESULT/日志保留，通过显式execution迁移收据只重试CPU比较，不重训或重采样。

新operations快照新增正式RL前真实资格门槛：从本组完整DEV选定的IL父模型生成一次真实rollout，执行4次inner，并从inner2恢复；实际sampling mean/std/raw/clamped logp、fresh ratio≤1e-4、原链/优势、adapter/Adam/scheduler/RNG和冻结父模型均必须通过，之后才启动RL50k。资格数据不计入正式50k。

科学训练、导出、评分和数据源码仍使用fda快照。新controller/比较器/profile统计通过显式双source receipt运行，不能重写旧qualification acceptance或忽略source_files差异。正式初始化永远重新加载同一通用Qwen并按共同seed创建新模块。

本次开发树完整针对性CPU测试实际执行：`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 <MODEL_PYTHON> -m pytest -q tests/world_compare`，**244 passed /31.56秒**。这不等于完整模型RL或Navtest资格。当前尚无本轮IL100k/RL50k或正式Navtest成绩。
