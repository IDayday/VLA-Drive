# 迁移范围与兼容边界

模型仍使用原 PyTorch/Accelerate/DeepSpeed。QwenOFT 的条件编码、动作 query 提取、qwen_proj 和 flow velocity 被重构为 SFT、ODE 推理和 RL 共用内核；保留原推理格式及旧路径数值 oracle。没有接入新 RL 大框架，没有 FPO++、MoE、scorer、Pareto、新世界模型或 curriculum。

`starVLA/rl/flow_grpo` 负责合同、观测隔离、SDE、buffer、官方奖励、独立 reference、联合 loss、训练及 checkpoint。`scripts/flow_grpo` 提供本机 launcher、锁定源码/权重下载、配置生成和完整状态审计。`tests/flow_grpo/vendor` 只放必要的锁定 sampler/原 SFT 方法，不把上游图像或 LoRA trainer 接入生产。

原工作区审计基线为 `9fe1459b8f6ab69a15274450ec301d541209bedd`。继承的未提交 `infer.py` 和 `QwenPI_DrivoRSuprim.py` 变更及其 diff 单独保存，不能认领为本次修改。实际运行还需查阅每次启动的 `source_environment_*.json`：包括 HEAD、逐源文件 SHA、dirty diff SHA、包版本、CUDA/GPU 和 deterministic 选项。最终代码 commit 不等于较早回归实验的实际代码，报告明确区分。

官方完整 SFT checkpoint 使用 decoder-last/pre-norm 特征；用户提供的 action-only 源代码使用 normalized last_hidden_state。这一差异显式记在 checkpoint contract 中，不按目录名或当前构造默认值猜测。

ZeRO 1/2 及 torch DDP 路径使用包裹 actor 的 forward；不绕过 distributed wrapper 调用 trainable 子模型来更新。允许 microbatch=1、候选分块、non-reentrant checkpointing、累积、optimizer/reference CPU offload；拒绝未经审计的 transition chunk 配置。torch DDP 不等有效数的缩放用独立测试覆盖；生产训练对数据/reward 异常采用全 rank fail-fast，不静默丢弃部分候选造成不等计数。

FA2 backward 与 Wan attention 使用显式 deterministic 开关；默认 launcher 开启，并设置 cuBLAS workspace。旧未开启确定性的实验若 resume 逐位比较失败，保留失败证据；不能删除或改变容差来伪装通过。真实 checkpoint 的梯度对照和断点续训结果单独报告。

完整 checkpoint 保存 actor、optimizer、scheduler、Python/NumPy/Torch/CUDA及原辅助模型 RNG、消费位置、配置、processor、normalization 和 provenance。保存目录原子完成并写 COMPLETE。导出取实际 BF16 forward 权重，不把优化器 FP32 master 冒充推理权重。原 evaluator 严格检查允许的 runtime-only 模块差异；export 验证再比较完整 tensor 和相同输入/seed 的输出。
