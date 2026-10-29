# 分阶段profile接口与已执行范围

本文件描述真实实现与CPU资格结果，不把待运行的完整GPU profile记为PASS。

## 实际入口

`starVLA/rl/qwen_world_compare/profiling.py` 的 `PhaseProfiler` 仅包装现有训练循环：

```python
with PhaseProfiler(output, rank=rank, wait=1, warmup=1, active=2,
                   metadata=common_identity) as profiler:
    # 现有 forward / backward / optimizer 计算不变
    profiler.step(update=completed_update, scenes=32,
                  wall_seconds=elapsed, data_wait_seconds=data_seconds)
```

更新编号必须连续且代表已验证的有效optimizer更新；`scenes=32`是全局逻辑曝光，不随八个rank累加成256。
`phase(name)` 使用 `record_function('compare/'+name)`。模型已有 current_Qwen、scene_C160、P_W_prefix、main_action、pre_readouts、clean_candidate_replay 标签；trainer标记backward与optimizer/communication。

输出每rank `PROFILE_rankNN.json` 和有限Chrome trace。wait1/warmup1/active2在4次更新内完整采集最后2次；CLI `--updates 4` 对应trainer `--profile-steps 2`，不错误地要求6步才能采完整。
CPU/device phase时间是**inclusive**，嵌套阶段不能相加。CUDA设备区间和共享worker wall不等于独占GPU busy。
记录峰值allocated/reserved、有效scene/s、真实窗口wall；未测得的GPU busy为null，不填0或拿CPU时间代替。

`aggregate_profiles()` 只在owned worker结束后聚合，拒绝缺rank、失败rank、不同更新序列/身份/曝光；吞吐用每更新最慢rank wall，global exposure只计一次。

## 真实CLI

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m tools.qwen_world_compare.profile \
  --config configs/qwen_world_compare/arch_a.yaml \
  --output <unique-profile-output> \
  --backend zero2 --micro-batch 2 --updates 4
```

此入口调用现有 `train_il.main()`，限定 `--qualification --joint --stop-after 4 --profile-no-save`，不另造优化循环。相同global32/精度/共同目标下可以用 `--backend ddp` 做容纳与吞吐对照。
必须由既有资源准入流程租赁8卡后调用；CLI不自行抢占或启动未授权GPU。
有限profile权重不会保存为正式初始化，不承接为正式IL。

真实执行过上述入口的 `--emit-command`，校验冻结recipe并输出实际trainer argv；它是无GPU命令资格，**不是一次GPU训练完成**。

## 部署profile

```bash
python -m tools.qwen_world_compare.profile \
  --export <sealed-qualified-export> \
  --data <sealed-TRAIN-root> \
  --output <unique-deployment-profile-output>
```

严格加载已封存的真实导出，使用TRAIN当前观测和scene绑定seed2101，测prefill、prefix、各denoiser、完整K10及端到端分段wall。只做单候选clean-x shift5，不调用teacher/scorer、不输出任何PDMS估计。
重复默认warmup1＋正式3次；额外真实forward次数单独登记。初始噪声固定，只是延迟重复，不是新增推理seed矩阵。

## 已执行测试与当前边界

`tests/world_compare/test_profile.py` 实际 **6 passed，1.78s**，覆盖CPU真实torch.profiler trace、两rank mock逻辑曝光聚合、缺失/失败/重复更新拒绝、故障证据、真实trainer命令合同、K10无scorer部署调用次数。
`py_compile`和真实CLI `--emit-command` 已执行。

尚未在本文件中登记任何八卡DDP/ZeRO2实测、真实Qwen分阶段时间或显存数值；必须由公共资格/调度器实际运行后追加结果，不能由参数量或CPU tiny结果推算PASS。
有限profile包含初始化附近和profiler自身开销，不能直接用来给100k训练生成精确ETA。后续正式无trace稳定窗口应另报。

