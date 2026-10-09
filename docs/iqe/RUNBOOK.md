# IQE 运行手册

工作区 `/mnt/project/DriveVLA-M0-iqe-20261009`；分支 `feature/s0-incremental-query-experts-20261009`。以下命令从该目录执行。训练环境为 `/root/miniconda3/envs/ddp/bin/python`，官方评分由配置内 `/root/miniconda3/envs/navsim/bin/python` 子进程执行。源码源头、真实数据、Qwen 通用初始化、地图路径已写入 main.yaml，未填造假权重路径。

**GPU 作业需要资源资格检查。** 用户后续明确授权 `training-vlawm-zt`、`training-vlawm-zt2`、`training-vlawm-zt3`，已在 `/mnt/project/server_dispatch_policy.json` 的 `task_authorizations.iqe_v1` 记录；原有 `training-vla-zt` / `training-vla-zt2` 仍可用。不能继承其他任务的更大主机权限。只允许停止已确认的压力脚本，保留全部科学训练。2026-10-09 实查：vlawm-zt 有正式 structured-world 训练；vlawm-zt2/3 的 GPU 占用来自当前 SSH 容器不可见的进程，尚未释放。CPU 真图像/真实官方评分 smoke 与 GPU/NCCL 验收分别报告。

```bash
cd /mnt/project/DriveVLA-M0-iqe-20261009
export IQE_PYTHON=/root/miniconda3/envs/ddp/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
$IQE_PYTHON -m iqe.cli preflight --config configs/iqe/main.yaml --repo .
$IQE_PYTHON -m iqe.cli build-splits --config configs/iqe/main.yaml --mode full --max-samples 101592 --device cpu
```

`--max-samples` 为每个角色的显式上限，实际角色不足该数时使用全部。全量导入会核验真实图像 hash，耗时和 I/O 需预算。main.yaml 的 Query 基础训练100000 updates/global batch32；新增专家和 Scorer 各3000/global batch128。smoke 独立输出目录、最多32步，不调用 navtest。

```bash
IQE_DEVICE=cuda IQE_SMOKE_STEPS=32 bash scripts/iqe/smoke_two_rounds.sh
```

训练效率先以独立 profile 配置执行8步。下例的30秒/update是启动前明确的资源预算，**不是实测值**。未通过时脚本退出，不启动正式训练；保留原日志进行 I/O、计算、通信诊断。调整 microbatch 必须保持有效 global batch，缓存仅用于已经冻结的模块，不能去掉辅助目标、降输入分辨率或替换动作目标。profile 与正式训练使用不同输出，不能把短 schedule 的 optimizer 状态续接成正式长 schedule。

```bash
bash scripts/iqe/qualified_base_training.sh configs/iqe/main.yaml configs/iqe/profile.yaml 101592 8 30
$IQE_PYTHON -m iqe.training.efficiency --directory outputs/iqe/main/query_base --warmup 3 --max-seconds-per-update 30 --watch-seconds 3600
```

上述脚本可作为独立作业管理器的前台命令。它不创建后台无限重试队列、不租云资源，也不覆盖已有进程。正式作业一旦开启，应保留 stdout/stderr 与 `status.json`、`steps.jsonl`、`EFFICIENCY.json`。基础训练 checkpoint 由结果 JSON 给出实际路径，不使用旧驾驶 checkpoint：

```bash
BASE=$($IQE_PYTHON -c 'from iqe.io import read_json; print(read_json("outputs/iqe/main/query_base/result.json")["checkpoint"])')
$IQE_PYTHON -m iqe.cli freeze-base --config configs/iqe/main.yaml --mode full --max-samples 101592 --checkpoint "$BASE" --device cpu
$IQE_PYTHON -m iqe.cli prepare-metric-contexts --config configs/iqe/main.yaml --mode full --max-samples 101592 --roles incremental_fit stage_val selector_cal --device cpu
$IQE_PYTHON -m iqe.cli verify-reference --config configs/iqe/main.yaml --mode full --max-samples 101592 --device cpu
$IQE_PYTHON -m iqe.cli cache-features --config configs/iqe/main.yaml --mode full --max-samples 101592 --roles incremental_fit stage_val selector_cal dev_report --device cuda
$IQE_PYTHON -m iqe.cli export-candidates --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 0 --device cuda
$IQE_PYTHON -m iqe.cli score-candidates --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 0 --backend reference --device cpu
$IQE_PYTHON -m iqe.cli build-round --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --device cuda
$IQE_PYTHON -m iqe.cli train-expert --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --device cuda
$IQE_PYTHON -m iqe.cli audit-frozen --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --device cuda
$IQE_PYTHON -m iqe.cli export-candidates --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --device cuda
$IQE_PYTHON -m iqe.cli score-candidates --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --backend reference --device cpu
$IQE_PYTHON -m iqe.cli evaluate-oracle --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --role stage_val --device cpu
$IQE_PYTHON -m iqe.cli train-scorer --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --device cuda
$IQE_PYTHON -m iqe.cli calibrate --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --role selector_cal --device cuda
$IQE_PYTHON -m iqe.cli evaluate --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --role dev_report --device cuda
$IQE_PYTHON -m iqe.cli export-bundle --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --require-gates --device cuda
bash scripts/iqe/run_round.sh configs/iqe/main.yaml 2 101592 --resume
```

`export-bundle` 默认只写不可变新包。满足 gate 后显式增加 `--activate` 才原子更新 active；smoke 被禁止激活。失败 gate 保留旧版。第二轮 run-round 在没有科学增益时拒绝昂贵自动扩容，仍可单独训练 Scorer，或 `run-round --selector-only` 复用已冻结池。Router 轮次由配置驱动独立训练、校准。

```bash
$IQE_PYTHON -m iqe.cli train-expert --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --resume auto --device cuda
$IQE_PYTHON -m iqe.cli train-scorer --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --resume auto --device cuda
$IQE_PYTHON -m iqe.cli run-round --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 2 --resume --device cuda
$IQE_PYTHON -m iqe.cli load-bundle --config configs/iqe/main.yaml --bundle outputs/iqe/main/rounds/round_001/bundle --mode full --max-samples 101592 --device cuda
$IQE_PYTHON -m iqe.cli rollback --config configs/iqe/main.yaml --mode full --max-samples 101592 --device cpu
bash scripts/iqe/benchmark.sh configs/iqe/main.yaml 2
```

恢复只保证 optimizer-step 边界。`--resume auto` 读取实际 `result.json` 的 checkpoint；若进程在写 result 之前中断，使用已完成 `step_*.pt.COMPLETE.json` 对应的真实 `.pt`。run-round 会自动寻找最后完整 receipt。改变 world size 不可 exact resume；专家/Scorer/Router 可显式 `--non-exact-finetune`，结果与正式 exact 分开。不能用修改 `--max-steps` 的方式延长已冻结 schedule。

多卡仅包装训练子命令，模块先建齐、冻结、再 optimizer/DDP。不能 torchrun 整条文件状态机。示例：

```bash
$IQE_PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 -m iqe.cli train-expert --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --device cuda
```

消融必须 fork 同一个已锁定基础、数据和初始候选银行，禁止每项实验重新初始化另一个 S0。fork 不复制 optimizer、不更改源权重。E3 必须提供严格验证的真实新数据 manifest；为空时实际实验仍叫 resampling_only。E12 尚无已绑定的可运行 LoRA-DiT 专家集合，不能仅运行其配置就声称完成对照。

```bash
PYTHONPATH=. $IQE_PYTHON scripts/iqe/fork_experiment.py --source-config configs/iqe/main.yaml --config configs/iqe/ablations/scene_router.yaml
$IQE_PYTHON -m iqe.cli run-round --config configs/iqe/ablations/scene_router.yaml --mode full --max-samples 101592 --round 1 --device cuda
$IQE_PYTHON -m iqe.cli train-router --config configs/iqe/ablations/scene_router.yaml --mode full --max-samples 101592 --round 1 --resume auto --device cuda
$IQE_PYTHON -m iqe.cli evaluate --config configs/iqe/ablations/scene_router.yaml --mode full --max-samples 101592 --round 1 --role dev_report --device cuda
PYTHONPATH=. $IQE_PYTHON scripts/iqe/fork_experiment.py --source-config configs/iqe/main.yaml --config configs/iqe/ablations/E1_policy_copy.yaml
$IQE_PYTHON -m iqe.cli build-round --config configs/iqe/ablations/E1_policy_copy.yaml --mode full --max-samples 101592 --round 1 --device cuda
$IQE_PYTHON -m iqe.cli train-policy-copy --config configs/iqe/ablations/E1_policy_copy.yaml --mode full --max-samples 101592 --round 1 --device cuda
$IQE_PYTHON -m iqe.cli evaluate-policy-copy --config configs/iqe/ablations/E1_policy_copy.yaml --mode full --max-samples 101592 --round 1 --role dev_report --device cuda
$IQE_PYTHON -m iqe.cli evaluate-baselines --config configs/iqe/main.yaml --mode full --max-samples 101592 --round 1 --role dev_report --device cpu
```

E4/E5/E6/E7/E8/E10/E11 同样 fork 对应配置后执行 run-round。E1 是在线图像完整策略副本 IL 微调，视觉塔仍按原框架冻结；不是把仅动作头更新伪称完整策略微调。其 checkpoint/评估独立保存，不替换 Base 或正式部署包。

final_test 只有独立 `final-test --bundle ... --final-manifest ... --mode full --max-samples ...` 入口，要求已接纳锁定包、独立组，记录访问，不写回训练/校准路径。本次没有可核实的 final_test manifest，因此不提供虚构绝对路径，也未运行。

实际已产生的 Query 基础诊断 checkpoint（真实图像、GT 与全部原辅助目标，CPU 1 update）为 `/mnt/project/iqe-runtime-audit-20261009/base_cpu_probe/query_base/step_000001.pt`，SHA256 为 `ff8f7905315d5d3b64e535c4799b872cd9466847a7f245d73c32ff8e8e1a266c`。这是 smoke 基础模型，禁止用 full 模式冻结或把其短 schedule 接成正式训练。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $IQE_PYTHON -m iqe.cli train-base --config /mnt/project/iqe-runtime-audit-20261009/base_cpu_probe.yaml --mode smoke --max-samples 32 --max-steps 1 --resume /mnt/project/iqe-runtime-audit-20261009/base_cpu_probe/query_base/step_000001.pt --device cpu
```

这条命令恢复已完成的1步作业，不会伪称又执行了训练。真实双轮另用隔离配置 `/mnt/project/iqe-runtime-audit-20261009/real_cpu_two_round.yaml`，fit32、stage_val4、selector_cal4、dev4，专家和Scorer各2步。完整状态见 `reports/iqe/REAL_SMOKE_REPORT.md`。

冻结特征可以确定性按场景分片；同一 cache key 的 encode/write 有文件锁。分片只改变离线生成调度，各场景仍使用完整原始图像、冻结链和 Decoder 条件。例如4个独立 CPU 作业分别用 `cache-features --shard-index 0 --num-shards 4` 至 `--shard-index 3 --num-shards 4`。已有完成标志/校验和正确的特征复用；不完整文件不会被当成缓存命中。

取得空闲 GPU 后，先运行真正的 NCCL 与真实框架探针，再测吞吐：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/iqe/gpu_qualification.sh /mnt/project/iqe-runtime-audit-20261009/real_cpu_two_round.yaml outputs/iqe/gpu_probe
```

该测试使用合成 tensor 验证 DDP 数学和 exact resume，随后独立使用真实 Qwen、图像和 GT 检验 GPU 接口；前者不替代后者。GPU 资源检查在 CUDA 上下文创建前由每台机器的 local rank0 执行，其他 ranks 通过 CPU rendezvous 同步结果，避免把本作业的上下文误认成已有任务。

真实私有专家恢复对照也已通过：`/mnt/project/iqe-runtime-audit-20261009/real_resume_probe_v2/interrupted/step_000001.pt` 是实际中断在第1步的 checkpoint。实验用同一个真实2步采样计划，与未中断运行比较下一批IDs、loss、实际gradient tensors、最终参数和RNG，全部完全一致。该诊断不改registry、不发布专家。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $IQE_PYTHON scripts/iqe/real_resume_probe.py --config /mnt/project/iqe-runtime-audit-20261009/real_cpu_two_round.yaml --round 1 --output outputs/iqe/real_resume_reproduction
```

E8的无校准value-only对照读取每个对应loss配置下 `evaluate-baselines` 输出中的 `value_only_scorer` 项；完整校准输出单列，不把没有component监督的head校准结果冒充纯value-only选择器。
