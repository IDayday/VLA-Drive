# Flow 安全策略改进研究

当前状态是 **进行中**。本轮固定使用 F-SFT `frozen_visual step100000`，完整权重 SHA256 为 `9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4`。它提供了已验证的冻结上游缓存、完整动作头和原 ODE 部署入口，适合作为本轮隔离探索、目标、更新和采样迁移因素的共同起点。本轮没有重新进行 C/S/V/A/G 全家族排名，不能据此称它是所有模型中绝对最优的 base。G 系列继续原有工作。

代码起点是 `25f9148eb75ddf171313720e758a280b82c24f7f`，分支为 `research/flow-safe-improvement-diagnostics-20261013`。本次阶段提交保存数据合同、权重求解器、FM/NFT 核心目标、原模型适配和真实评分验证。D1/D2/D3、P0/P1/P2、三轮在线更新、NFT 匹配学习、第二训练 seed 和最终 Navtest **尚未完成**，不能将本页的接口测试作为强化学习效果或方法胜出的证据。

旧结果见[已完成的 DDP Flow-GRPO 报告](../ddp_flow_grpo_final_report_20261013/README.md)。最终 v1.1 PDMS 提高 0.1722 个百分点、v2 EPDMS 提高 0.2126 个百分点；v1 NC/TTC 下降，DAC 均值小幅提高，新增零分场景多于恢复场景。平均收益存在，但未证明稳定全面提升。36.204% 组等分不等于全部失败；至少 71.461% 有效优势组仅由 progress 区分。旧 reward 已逐条使用“官方 PDM 参考＋一条候选”，不归因于不存在的批量 EP 归一化错误。

复用内容包括 `action_head_policy.py`、原动作 DiT 和 `qwen_proj`、原 action-only 优化器分组、SDE 转移及 dimension-mean 定义、保存转移评估、冻结特征缓存、独立 SFT reference、原推理入口、数值配置和恢复工具。旧工程已做的完整动作头训练、359 张量梯度验证、global-batch 标准化、时间相关噪声、noisy-Euler、去噪 discount 和 GT replay 不作为本轮新方法。旧条件 KL 不是部署 ODE 的精确 KL，clip 0.02 也不是最终轨迹分布的 2% 变化上限。

本轮新增有限候选 FP64 权重求解：共同 `u=0.5 q_ref+0.5 empirical`、真实奖励下限和独立 NC/DAC/TTC 成本预算；零预算通过限制支持集处理，KL 始终相对原始 u。混合规则使用绝对收益和共同 `delta_useful=0.005`，不除以组内标准差。原始 SLSQP oracle 在小权重边界出现错误收敛，已保留失败记录并改用独立 [Clarabel 指数锥求解](https://clarabel.org/stable/examples/py/example_expcone/)，快速路径仍为低维 dual。它是有限候选上的受约束提取，不声称严格策略安全或精确 CVPO 复现。[CVPO 原文](https://proceedings.mlr.press/v162/liu22b.html)与[FSRL](https://github.com/liuzuxin/FSRL)仅提供相关目标分布背景。

FM 使用非负、按场景归一化的候选抽样，插值噪声与候选 root noise 分离。保留原 SFT 的时间分布，包括 `noise_s=0.999` 带来的很小负时间尾部，没有静默截断。NFT 核心取自[官方 DiffusionNFT](https://github.com/NVlabs/DiffusionNFT/tree/cbb14f84b8312b620390dfcbe2fab69c1383b104)，实际执行其训练代码中的小张量算术并比较梯度；图像方向到 DDP 的映射为 `s=1-t, v_DDP=-v_image`。官方非 LoRA reference 分支尚有 `assert False`，后续动作头实验使用原工程独立冻结 reference，并明确记录差别。当前只完成核心对齐，尚无 NFT 真实学习结论。

| 已执行验证 | 结果 | 证据 |
|---|---|---|
| 新增合同、去重、约束边界、FM 和官方 NFT 数值测试 | 33 通过 | [测试输出](evidence/contracts_tests.txt) |
| 两个真实训练场景，原导出 ODE 与新适配接口 | 最大绝对差 0 | [GPU 验证](evidence/action_adapter_qualification.json) |
| 独立旧源代码动作头 ODE | 最大绝对差 0 | 同上 |
| 旧 G16/K10/rho0.8 SDE 与新独立 RNG 接口的同样本核对 | chain 和 logprob 逐值一致 | 同上 |
| 完整动作头反传 | 359 个张量、819,503,620 参数；梯度有限且非零，上游无梯度 | 同上 |
| 外部请求轨迹编码往返 | 最大绝对差 4.77e-7 | 同上 |
| 真实 v1 缓存与逐候选正式评分 | 2 场景、12 次评分；顺序、切分和其他候选不影响同一候选 | [评分协议及结果](evidence/v1_reward_qualification.json) |
| 完整八卡 NCCL | 8 ranks 全部通过 | [资源记录](evidence/resources.json) |

这些是资格验证，不是学习结果。FM 计时约 0.226 秒的探针包含两次前向和一次反向；ODE 两场景前向约 0.390 秒；真实评分中位数约 0.031 秒/候选。样本很少，且未计入完整优化器、搜索和并行通信，不能用它们宣称总体加速倍数。完整成本登记将在正式训练前补齐。

数据复用既有合法完整 log 划分：RL_TRAIN 101,592 场景/1,176 logs，RL_DEV 1,696 场景/16 logs；不是重新凑出 1,696，也不是新抽出的 10%—15% logs。F-SFT 可能见过 RL_DEV，旧全量 RL 见过它；仅本轮 RL 更新未见。旧 Navtest 已反复观察。固定 reference seeds、独立评价 seeds、训练 seeds 和预登记子集见[人口及种子合同](evidence/population.json)。定向诊断和困难学习部分将在固定 SFT 参考反馈后、候选搜索前锁定，定向部分单独报告。

用户最新资源安排已落实：`training-vlawm-tsinghua-zt` 八卡供 RL 验证和训练独占，tsinghua 缓存集中到 `training-vlawm-tsinghua-zt2`，保留其 GPU0 上的 IQE 工作。分配时两台均没有活跃缓存生产进程，因此没有假称迁移了不存在的任务。只释放了已确认的压力填充进程，并保存其私有恢复记录；Base/G 未收到停止信号。资源路由与科学合同分开保存。

完整实验边界见[预登记配置](../../configs/flow_safe_improvement/registration_v1.yaml)，上游 commit 和代码 hash 见[来源锁](../../configs/flow_safe_improvement/upstream_lock.json)。P 组共同 4,096 场景、global scene batch 16、8 epochs；在线至少三轮，固定 SFT 安全预算；最终仍需新噪声下的单候选原 ODE 成绩。有限候选 best-of-N 将只作为机会诊断。

以下入口已实际执行。路径下的原始场景、请求、模拟状态、缓存、权重和私有恢复信息留在共享运行目录，不上传到 Git。`--output` 使用新的目录以保留既有证据；尚在构建中的缓存命令应等待或使用已记录恢复入口，不能并发重复启动同一任务。

```bash
cd /mnt/project/VLA-Drive-flow-safe-improvement-20261013
SFI_PY=/mnt/project/flow-safe-improvement-20261013/venv/bin/python
SFI_RUN=/mnt/project/flow-safe-improvement-20261013

$SFI_PY -m scripts.analysis.flow_safe_improvement.make_splits \
  --source /mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/split_manifest.json \
  --navtest-csv /mnt/project/DriveDreamer-Policy-full-navtrain/runs/full_navtrain_epoch1/evaluation/sft_navtest_seed42_pdms_v1/sft.csv \
  --output "$SFI_RUN/population"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OFFICIAL_NFT_SOURCE="$SFI_RUN/upstream/DiffusionNFT/scripts/train_nft_sd3.py" \
  $SFI_PY -m pytest -q tests/safe_flow_improvement

# 在已授权 RL 主机运行；当前实例的输出是 adapter_attempt3。
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1 \
  $SFI_PY -m scripts.analysis.flow_safe_improvement.qualify_action_adapter \
  --config configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml \
  --source-config /mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml \
  --population "$SFI_RUN/population/manifest.json" --output "$SFI_RUN/validation/adapter_NEW"

# 在已授权缓存主机运行；当前已完成的资格缓存为 2 场景。
CUDA_VISIBLE_DEVICES= $SFI_PY -m scripts.analysis.flow_safe_improvement.build_v1_cache \
  --population "$SFI_RUN/population/manifest.json" --cohort calibration_tokens --limit 2 \
  --workers 2 --cache-root "$SFI_RUN/metric_cache_v1" --output "$SFI_RUN/validation/v1_build_NEW"

CUDA_VISIBLE_DEVICES= $SFI_PY -m scripts.analysis.flow_safe_improvement.qualify_v1_reward \
  --requests "$SFI_RUN/validation/adapter_attempt3/requests.npz" \
  --population "$SFI_RUN/population/manifest.json" --cache-root "$SFI_RUN/metric_cache_v1" \
  --output "$SFI_RUN/validation/v1_score_NEW"
```

保留的失败包括 SLSQP oracle 边界不收敛、新 worktree 缺少相对资产路径、模型构造会原地补充 DiTConfig 导致缓存身份计算时机错误，以及一个恢复合同测试调用方式错误。已分别修正独立数值 oracle、显式只读源配置路径、构造前身份绑定和测试调用；没有放松原数值阈值、替换旧缓存或覆盖旧正式 run。
