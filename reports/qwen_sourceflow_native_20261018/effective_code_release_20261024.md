# 当前有效Qwen-Flow：IL/RL代码交付（2026-10-24）

分支：`feature/qwen-sourceflow-native-vla-20261018`。当前有效版本是原生Qwen3-VL-2B＋完整SimWAM动作专家，不依赖旧DDP action query/history encoder/8×4编码。当前发布包含模型、完整IL、真实链Flow-GRPO、原生恢复、LoRA合并、单候选ODE部署、canonical评价和配对分析。

## 已有真实效果

固定12146场景/136 logs、canonical v1.1、seed2101、单候选10步shift5 ODE：自身IL53k为87.016025，RL_CANONICAL20k为90.284539（+3.268513），RL_SOURCE20k为90.233689（+3.217663）。尚未达到92；仍有新增失败，不能称逐场景安全保证。分项、区间和成本见[20k完整Navtest报告](RL20k_NAVTEST_20261024.md)。

## 实际训练源码身份

- 正式IL：`d4d0012769540af8378e013c072d6d8053d9f275`。
- 正式RL/本轮Navtest：`434cba8e47aef3e5ab8319f6b017ab5b0f860db1`。
- 两个运行commit都在本分支历史中；逐项核验run identity登记的源文件与对应git blob的SHA256全部一致，见[交付审计](effective_code_release_20261024.json)。
- 本分支当前模型、actor、sampler、loss、optimizer配置和原生恢复核心与正式RL快照逐文件无差异；后续增加的离线分析校验、运行报告也已保留。
- 旧run exact resume使用它登记的不可变源码及原生checkpoint。含新分析文件的HEAD会改变完整source-files身份，不绕过旧检查；跨源码恢复使用已有显式迁移收据入口。

## 代码入口

| 功能 | 已跟踪的实际入口 |
|---|---|
| Qwen-Flow模型 | [framework](../../starVLA/model/framework/qwen_sourceflow.py)、[完整模型](../../starVLA/model/modules/qwen_sourceflow/model.py) |
| 全token观测适配 | [condition provider](../../starVLA/model/modules/qwen_sourceflow/condition_provider.py)、[原生Qwen](../../starVLA/model/modules/qwen_sourceflow/qwen_encoder.py) |
| 源动作生成 | [MoT动作读取](../../starVLA/model/modules/qwen_sourceflow/cached_action_runner.py)、[FM/ODE](../../starVLA/model/modules/qwen_sourceflow/source_flow_policy.py)、[3维codec](../../starVLA/model/modules/qwen_sourceflow/action_codec.py) |
| 完整IL训练 | [train_il](../../scripts/analysis/qwen_sourceflow/train_il.py)、[IL配置](../../configs/qwen_sourceflow/il.yaml) |
| 原生RL入口 | [train_rl](../../scripts/analysis/qwen_sourceflow/train_rl.py)、[runtime](../../starVLA/rl/qwen_sourceflow/rl_runtime.py) |
| RL实际训练循环 | [共享source trainer](../../scripts/analysis/simwam_recipe/train.py) |
| actor/真实采样链 | [actor](../../starVLA/rl/qwen_sourceflow/actor.py)、[collection](../../starVLA/rl/qwen_sourceflow/collection.py)、[源SDE与概率](../../starVLA/rl/simwam_recipe/vendor/action_math.py) |
| PPO/参考锚定 | [源loss](../../starVLA/rl/simwam_recipe/vendor/trainer_math.py)、[共享适配](../../starVLA/rl/simwam_recipe/trainer_adapter.py) |
| 两组正式配方 | [CANONICAL](../../configs/qwen_sourceflow/rl_canonical.yaml)、[SOURCE](../../configs/qwen_sourceflow/rl_source.yaml) |
| 原生恢复 | [IL恢复](../../starVLA/rl/qwen_sourceflow/il_checkpoint.py)、[RL事务恢复](../../starVLA/rl/simwam_recipe/checkpoint.py)、[身份迁移](../../starVLA/rl/simwam_recipe/lineage.py) |
| 合并与部署 | [export_policy](../../scripts/analysis/qwen_sourceflow/export_policy.py)、[真实图像合并验证](../../scripts/analysis/qwen_sourceflow/validate_rl_deployment.py) |
| 正式评价与分析 | [evaluate_policy](../../scripts/analysis/qwen_sourceflow/evaluate_policy.py)、[完整汇总](../../scripts/analysis/qwen_sourceflow/summarize_results.py)、[自身IL配对](../../scripts/analysis/qwen_sourceflow/analyze_paired.py) |

RL不是占位脚本：入口调用已经实际运行的共享trainer；冻结Qwen/provider，仅训练240个attention投影的480个LoRA A/B张量；G8/K10、随机步骤7/8/9、四次真实AdamW更新、永久IL reference与source均值锚定均已接通。两种reward从同一官方仿真分项读出，正式评价始终canonical5:5:2。

独立源码、通用资产revision/许可证、数据划分、真实恢复/部署验收和执行命令均保留在本目录及configs。大权重、原始数据、缓存和私有场景图不随代码上传。

DINO当前帧辅助＋三视图三个时刻输入属于随后新开发的独立版本，不混入本次已有正收益的有效基线，也不热修改当前IL/RL或后台Navtest源码。
