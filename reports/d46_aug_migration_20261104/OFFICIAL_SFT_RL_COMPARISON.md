# 官方SFT直接进入RL的对照

2026-11-05：无需重新训练SFT。官方README明确Stage I以SFT best为父模型，Stage II才使用Stage I best。本地官方SFT完整权重已验证SHA，直接加载原模型、原配置、归一化统计与processor，初始化新Adam；D迁移各路线仍独立从D IL46开始，不交换权重或optimizer。

官方配置：B32/G8；LR1e-6；3步SDE训练、1步ODE部署；action clip0.2、action KL0.001；CoT KL0.01、split advantage；原Stage I为action+CoT联合训练。原配置不是仅动作JaT。

## 已完成的真实步骤

本地原模型已从官方SFT完成一次真实action-only JaT更新，未使用发布的Stage I权重初始化。G8候选均有真实PDM评分，组内奖励不相同，原生JaT loss反向及Adam执行成功。总诊断109.58秒（含8场景初始推理、加载、评分及更新），峰值显存38,326,660,096 bytes。单TRAIN场景PDMS从0.94230174变为0.94331006，NC/DAC/TTC均1；不能将此单样本变化当作整体提升。

前次显存失败是诊断脚本在eval模式执行反向，导致原模型条件式梯度检查点未启用。修正update模式、显式启用检查点且Adam foreach=False后完成更新。另测到BF16缓存采样和完整序列重计算的fresh ratio最大偏差0.01425248，所有候选均未触发原clip0.2；这项差异保留在结果中，不声称位级相等。先前本地额外1e-3判据不是官方约束，已取消该判据，原loss未改。

证据：[真实单次更新](OFFICIAL_SFT_ONE_UPDATE.json)。它不是完整Stage I复现：本次只1个TRAIN场景/G8/M1、action-only、单卡、FP32参数/BF16 autocast；原模型的8场景推理也不是NAVTEST评测。

## 数据和下一阶段

官方P3 TRAIN标注181,752,662 bytes已下载并核验SHA，含103288条记录及真实perception/prediction/planning标签。SFT配置实际cot_components=['prediction']，可用真实solution_prediction构造对应监督；不可将缺失的自由文本reasoning_trace伪造为已获取。官方通用cache脚本会按reasoning_trace缺失过滤一条样本，确切复现其103287人口还需对齐该场景身份；更不能悄悄把全103288称作原发布人口。

后续先按官方SFT输入、原生联合奖励和完整推理协议建立自己的SFT基线及Stage I学习曲线，再比较D迁移相对其自身IL46的增益。原始SFT→RL和D IL46→迁移RL是两个不同初始化体系；不能仅凭最终绝对分数归因于loss。继续登记场景曝光、评分次数、ratio/KL、梯度和零分修复/退化。达到完整NAVTEST人口之前不发布正式部署分数。

来源：[官方RL说明](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/README.md)、[Stage I配置](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/recipe/transfusion/config/stage1.yaml)。
