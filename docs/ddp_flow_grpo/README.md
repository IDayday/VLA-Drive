# DriveDreamer-Policy action-only Flow-GRPO

当前修复分支的代码、实测结果和命令见 [配对运行说明](paired.md) 与 [本轮验收报告](../../reports/ddp_flow_grpo_paired/validation.md)。工程状态 **NOT_READY**。F/U 均保留并冻结各自 SFT visual，其余原 SFT 可训练参数继续训练；没有新增辅助任务或 LoRA。

历史 `fc354f8` 验收、BF16 candidate chunk 失败、F50/U2短训等保存在 [历史报告](../../reports/ddp_flow_grpo/validation.md)，未覆盖。它们不作为本轮100→2000配对实验起点，也不替代新数值profile验收。

用户原有 `infer.py` / QwenPI 未提交变更已经作为独立来源提交 `4acd8e7` 纳入修复分支。干净 checkout 执行 `python scripts/flow_grpo/prepare_workspace.py`，无需也不得重复 apply `inherited_dirty.patch`。原工作区与旧 worktree 的未提交文件、训练作业、checkpoint 保持原样。

正式训练现在有机器可检查的放行门禁；没有匹配代码/配置/权重/资产/实测证据的记录时，仅允许最多8次更新的显式诊断。配对训练的100次短训也需通过两组生产profile验收。旧文档中的无放行长训命令已经撤下。
