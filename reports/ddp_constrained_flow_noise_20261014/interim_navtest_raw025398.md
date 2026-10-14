# 用户要求的中期完整 Navtest：相同 25,398 次原始曝光

状态：2026-10-14 19:13 UTC 启动，尚未发布完整分数。

用户明确要求“测一下测试集结果”。因此新增一次中期 Navtest 观察，
不将它称为正式预算终点，不用该结果调整 sigma、lambda、margin、训练预算或检查点选择。
两条 RL 线均取预先登记的首个 25,398 次原始场景曝光检查点，均为 update2030；
这是 203,184 次正式预算的 12.5%。SFT 同时使用匹配原始入口重新评价。

三组均为完整 12,146 场景 / 136 logs，训练 seed42、推理 seed2101，
原始观测、原 10 步 ODE、单候选、canonical v1.1。不使用 noise_head、候选筛选、
训练 condition cache、搜索或新 scorer。Navtest 已有历史使用，本次中期观察也须在最终结论中披露。

[权重与协议锁](interim_navtest_raw025398_lock.json) 在任何本次 Navtest 推理之前写入。
为原样复用冻结 sampler，锁中保留其历史 `LOCKED_BEFORE_FINAL_NAVTEST` 解析字段；
独立 schema、purpose、`formal_budget_complete=false` 和结果 context 明确这是中期观察。
不会进入 `final_confirmation_v1` 的正式终点/安全择优记录。

[执行登记](interim_navtest_raw025398_execution.json)：SFT 在 Tsinghua zt2，
C 在 Tsinghua zt4，N 在 Tsinghua zt；每模型八个进程，共 24 张不同物理 GPU。
保留已有训练和 census，未停止任何压力或未知进程。每组 16 个 CPU scorer，
在原子预测发布后立即评分。三组总候选评分预算为 36,438；保留失败分母。
已验证原推理单进程峰值为 7,950,648,320 bytes，本次调度仍采用更保守的
每卡空闲显存至少 35,000 MiB 门槛，并在启动时重新检查。

新增入口只连接原 locked sampler、canonical scorer 和配对分析器；完整基础 SFT
尚未完成时等待其完整 score bank，不将局部分数当基线。新增和既有选择测试共 12 项通过。
冻结的 83 个科学源码保持不变。

```bash
python -m scripts.analysis.constrained_flow_noise.interim_navtest lock \
  --run-root /mnt/project/constrained-flow-noise-20261014 \
  --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json \
  --c-run /mnt/project/constrained-flow-noise-20261014/formal_C_SDE_seed42_v1_probability_repair \
  --output NEW_OBSERVATION_LOCK

python -m scripts.analysis.constrained_flow_noise.interim_navtest evaluate \
  --lock OBSERVATION_LOCK --model SFT_OR_C_SDE_OR_N_FLOW_NOISE \
  --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json \
  --output MODEL_OUTPUT --baseline-scores SFT_OUTPUT/scores --workers 16
```

真实已执行参数位于执行登记。恢复同一 `evaluate` 命令，保持模型、锁、输出和源代码身份，
复用已验证的原子预测/评分记录；不覆盖旧锁或重选检查点。

同时修复最终确认控制器的 C 路径：验证原生恢复身份后跟随明确的 continuation 指针，
历史与续训曲线合并，重复采集的旧尝试保留成本记录。此修改只影响只读汇总和未来评价路由，
不改变两组训练参数、采样、梯度或终点选择规则。
