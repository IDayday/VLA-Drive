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

## fdaff80 真实资格与有界backend对照准备

科学快照为 `runtime/formal_fdaff80`（`fdaff809652370328e45c0d996ad706e1af2fc0c`），没有热修改。
A/B/C/D均已实际运行八卡joint4、未skip。下表是各rank0最后完整更新的实测及checkpoint声明的参数统计，**不是带trace的DDP/ZeRO2成对结果，也不是稳态ETA**。

| 架构 | 总optimizer参数 | rank0 allocated GiB | reserved GiB | 最后单次最慢rank秒 |
|---|---:|---:|---:|---:|
| A | 2,710,801,843 | 21.9053 | 26.9258 | 3.9512 |
| B | 2,661,686,323 | 21.5199 | 26.7695 | 3.9331 |
| C | 2,713,958,855 | 21.9474 | 26.6914 | 4.9633 |
| D | 2,220,489,267 | 22.2304 | 24.4590 | 6.6135 |

连续4与2+2恢复GPU进程均退出0；完整native比较器曾因LossScaler对象身份比较失败，主流程在独立操作源按严格类型/字段修复后重核。不能在CPU比较收据成功前把它描述成完整native恢复PASS。
现有资格owner覆盖真实七loss、Qwen语言/视觉梯度和native继续训练；完整部署原始观测/导出、RL首轮真实概率与mid-inner恢复等仍需各自资格证据，不由上述四步自动推导。

`PROFILE_PAIR_PLAN_fdaff80.json`已保存在真实运行root，状态明确为 `PLANNED_NOT_LAUNCHED`。八个有限任务共32次optimizer更新、1024次场景曝光；每架构两backend各4更新，保持同global32、相同七loss、同first32资格场景/目标/候选序列。当前agent没有派发任何GPU。
每臂输出独立 `profile_pairs_fdaff80/<A|B|C|D>/<zero2|ddp>`；A/B/C采用相同micro2、D采用相同micro1。完整argv均在计划收据中，供主owner在已有资格任务释放资源后租赁空闲节点执行。

注意 `--joint` 的有限资格从update0开始，辅助系数仍按登记warmup为1/1000至4/1000；它运行同一套七loss及全部joint参数，但不是训练1000步后完整系数的稳态窗口。两个backend完全匹配该条件，报告不隐瞒这一边界。

### DDP与存储边界

目前 `ProfileDDPEngine` 未实现`no_sync`，因此每个micro backward都会同步：micro2/accum2每update两次，D的micro1/accum4四次。数学目标可以相同，但通信成本不能视为已经优化；本轮没有擅自更改这个实现。
正式仍采用已验证ZeRO2。DDP容纳/吞吐profile通过不等于获得DDP native exact-resume资格，也不自动切换正式后端。

仅参数、梯度、两份Adam moment的FP32静态量（16字节/参数）已分别约40.39/39.66/40.44/33.09GiB；DDP bucket、激活、临时buffer和allocator保留尚未计入，不能根据ZeRO2约22GiB实测便宣称DDP必然同样占用。若实际OOM必须保存失败证据，不以删除任务/token改变数学后宣布原配方PASS。
A/B/C完整native checkpoint实测约40.395/39.663/40.442GiB，每个包含18项payload；profile使用no-save，避免把native写盘时间混入denoiser比较。native比较器应在内存足够的既有worker主机运行，双完整model约20GiB CPU载入，不在低内存前台并发重建。记录时共享磁盘可用约9.5TiB，不据此预设未来长期存储无约束。

### Profile完成门槛与只读pair CLI

开发树修复了一个纯报告边界：成功提前退出但不足wait/warmup/active窗口，以前仍写COMPLETE，现在为INCOMPLETE，分别记录计划与实际active更新。旧科学快照不热改；旧输出也必须通过下面的严格只读验收才能称完成。

```bash
python -m tools.qwen_world_compare.profile \
  --validate-pair <actual-zero2-output> <actual-ddp-output> \
  --architecture-id A --output <fresh-pair-summary.json>
```

验收要求八rank各4次实际optimizer更新且skip0、八个完整CUDA trace（有真实kernel事件）、全部必需阶段、global32、每全局update恰好8个post场景，以及相同源码/初始化/配置/数据/教师/目标/候选/evidence和实际scene顺序。
源码与确定性样本、query和已测试的全update归约构成分母一致的算术依据；当前fda没有单独持久化各步实际den值，收据明确 `independently_persisted_denominators=false`，不伪称独立测得逐值相等。不同micro时单独标出，不能归因纯backend加速。
新增验证器测试使用显式synthetic fixture，不冒充GPU运行证据。当前profile专用测试实际 **11 passed，1.64秒**，包含早停1/3步不能PASS、缺trace/CPU-only/未实际step拒绝、目标变更拒绝和micro差异标注；编译、真实CLI `--help`及diff检查通过。

共享未来targets_v2和1,454,240条native evidence已完成封存；本次只读核查COMPLETE/manifest/已有全量验证收据，不重跑数据producer。v1 targets仍拒绝用于正式训练。

