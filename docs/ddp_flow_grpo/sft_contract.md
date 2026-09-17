# 源 SFT 与 RL 参数合同

用户指定的源码锁定为 `IDayday/VLA-Drive@f9449d55bea6895a7a0bd86d09d7ab85fd353f26`。根目录源码对应 visual-unfrozen 训练；冻结视觉版本使用发布分支中的 frozen-visual overlay。源 YAML/JSON、launcher、逐文件 SHA 及来源都保留，不运行 DLC 脚本。

| 初始化 | 源视觉状态 | GRPO 视觉状态 | 完整权重 SHA256 |
|---|---|---|---|
| frozen-visual step100000 | 冻结 | 冻结 | `9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4` |
| unfrozen-visual step120000 | 可训练 | 用户明确要求冻结 | `72e626e152a357ec9c478c4b253500599301b38b87758e0aebfa486ff3be21cc` |

两者均为 QwenOFT + GR00T FlowmatchingActionHead，minimal history/action prompt、Qwen normalized `last_hidden_state`、原 8×4 动作表示、10 步 Euler 推理。并非项目内的 QwenPI/多轨迹/层次规划器。动作输入为原来的当前三视角图像和 ego history/command，保持原始输入尺寸。

原 source frozen YAML 的 `trainer.freeze_modules` 是空字符串，但发布者说明和归档 launcher 明确记录冻结 visual 和 lm_head。解析配置按后两项证据恢复这两个冻结路径，同时在 `source_provenance.json` 保存原字段和解析结果。辅助任务开关采用实际合并配置中的关闭状态，不照搬归档 launcher 中未覆盖的默认值。

unfrozen 版本的源配置冻结 lm_head 和 `agent_dino_head`，视觉可训练并使用独立 2e-6 学习率。它比 frozen 文件多 6 个 inactive `agent_dino_head` tensor；严格保留并冻结这些参数，minimal 模式不调用该分支。两版本其余共有 tensor 形状相同。是否完整可加载最终以权重 SHA 和真实 strict load 为准。

每次启动先按源 SFT 构造并冻结模型、构造原 optimizer 参数组、导出 `source_sft_parameter_manifest.json`；然后应用用户明确授权的 visual freeze，导出 `actor_parameter_manifest.json`。除这一个显式例外，canonical name、别名/tied 参数、shape、dtype、trainability、group 必须逐项相同。`parameter_contract_exceptions.json` 记录视觉参数的精确 canonical 名单及参数数目。重复参数、漏收参数、零学习率、额外冻结/解冻直接失败。

源 SFT 中可训练的语言层、有效 embedding、history MLP、qwen_proj 和动作 DiT 继续训练。lm_head 的冻结按参数对象处理，若 checkpoint 存在 tied embedding，不能为了宣称 embedding 可训练而解除源冻结。是否实际取得策略梯度由独立 RL/reference/SFT/total backward 逐参数记录；`requires_grad=True` 仅是构造状态，不作为梯度证据。

`gradient_sources` 只记录实测非零梯度，`gradient_graph_sources` 另行记录得到梯度张量（包括真零）的分支。`gradient_coverage.json` 及 manifest 分类为 policy / auxiliary-only / SFT-only / zero-gradient-observed / dormant / frozen。辅助任务在最终 action-only 目标中不适用，不为满足旧完整 SFT 文本而重新引入。原官方完整 SFT 兼容配置仍保留所有原始辅助损失作为回归验证，相关报告与 action-only 报告分开。

原始 optimizer 模块分组和 AdamW 的 beta/eps/weight_decay 沿用。GRPO learning rate 默认是源 SFT 各组的 0.1 倍；常数 scheduler 使 stop boundary 不改变学习率轨迹。所有 trainable 参数必须被覆盖且只出现一次。
