# 实际入口、身份与恢复

本页记录已经执行过的入口及对应恢复方式。运行中的正式IL不能再启动一份；新派发前读取 `/mnt/project/server_dispatch_policy.json` 并重新检查物理GPU及进程归属。所有命令使用真实存在的模块，不依赖旧DDP actor。

## 路径与快照

```bash
QSF_ROOT=/mnt/project/qwen-sourceflow-native-20261018
QSF_PY=/mnt/project/flow-safe-improvement-20261013/venv/bin/python
QSF_IL_SOURCE=$QSF_ROOT/snapshots/native_verified_startup
QSF_RELEASE=$QSF_ROOT/snapshots/native_release_qualification
```

正式IL源为 `d4d0012`，独立部署dtype修复为 `23e49ee`，真实恢复/LoRA部署资格入口为 `d6fda61`。部署补丁不进入正在运行的IL快照，不改变已训练的模型/optimizer。源码完整hash、依赖runtime hash及配置保存在各run identity内；不能仅核对Git分支名。

实际入口/CPU owner收据：

| 工作 | 私有运行记录 |
|---|---|
| 正式IL | `dispatch/formal_IL_v2/status.json`、`process.json`、`train.log` |
| IL模型、Adam、游标/RNG | `runs/IL_seed42_v2/checkpoints/`、`latest.json` |
| 全量DEV队列 | `milestone_queue_v2/owner.json`、`identity.json`、`status.json` |
| 真实GRPO前两inner | `dispatch/native_rl_v4/` |
| 同链恢复inner3/4 | `dispatch/native_rl_v4_resume/` |
| 独立连续4inner对照 | `dispatch/native_rl_v4_continuous/` |
| SOURCE奖励资格组 | `dispatch/native_rl_source_v4/` |
| standalone实际IL部署 | `dispatch/deployment_v4/` |
| 实际RL合并 | `dispatch/RL_merge_v1/` |
| 非零LoRA完整部署对照 | `dispatch/RL_merged_deploy_v1/` |

这些资格实验不计入正式训练预算，其物理计算与评分费用单列。控制器和worker通过owner lock、PID/start ticks、immutable spec及结果收据防止重复启动。当前没有自动恢复失败worker或在撤销主机重试的逻辑。

## 正式IL已执行命令

以下是已运行任务的命令记录，**不是要求再启动一份**：

```bash
cd "$QSF_IL_SOURCE"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$QSF_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.analysis.qwen_sourceflow.train_il \
  --config configs/qwen_sourceflow/il.yaml \
  --initialization "$QSF_ROOT/assets/action_generic.pt" \
  --output "$QSF_ROOT/runs/IL_seed42_v2"
```

需要暂停时只向已确认属于该run的8个原生Python worker发送SIGUSR1，保留至完成optimizer边界再原子保存。不能直接kill未知进程、删除lock文件后重复启动，或向torchrun误发原生暂停信号。

恢复使用同一不可变源码、同一run、同一8卡配置；先确认旧worker已退出，验证 `latest.json` 指向完整READY checkpoint，并以其真实路径替换下方参数：

```bash
cd "$QSF_IL_SOURCE"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$QSF_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.analysis.qwen_sourceflow.train_il \
  --config configs/qwen_sourceflow/il.yaml \
  --initialization "$QSF_ROOT/assets/action_generic.pt" \
  --output "$QSF_ROOT/runs/IL_seed42_v2" --resume "$QSF_READY_CHECKPOINT"
```

`QSF_READY_CHECKPOINT`必须来自已验证的该run边界；不是可任意换成merged模型的初始化接口。qualification曾从完整IL的update2恢复到4，模型/Adam/RNG逐值一致。恢复时不修改warmup、数据顺序或初始化权重。

## 已实际执行的真实GRPO与恢复

资格父模型是4步资格IL，不能进入正式RL。第一次在inner2暂停，再用相同源和output执行剩余两次inner：

```bash
cd "$QSF_IL_SOURCE"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$QSF_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.analysis.qwen_sourceflow.train_rl \
  --config configs/qwen_sourceflow/rl_canonical.yaml \
  --manifest /mnt/project/simwam-source-process-20261018/source_manifest.json \
  --il-checkpoint "$QSF_ROOT/qualification/IL_v3_continuous/checkpoints/update000004" \
  --output "$QSF_ROOT/qualification/RL_v4" --qualification \
  --stop-after-update 4 --save-every 2 \
  --resume "$QSF_ROOT/qualification/RL_v4/checkpoints/update000002_inner2"
```

真实8卡恢复比较也已执行：

```bash
cd "$QSF_RELEASE"
CUDA_VISIBLE_DEVICES='' "$QSF_PY" -m scripts.analysis.qwen_sourceflow.compare_rl_resume \
  --continuous "$QSF_ROOT/qualification/RL_v4_continuous/checkpoints/update000004_inner0" \
  --resumed "$QSF_ROOT/qualification/RL_v4/checkpoints/update000004_inner0" \
  --output "$QSF_ROOT/validation/RL_actual_resume_v4.json"
```

比较adapter、Adam、scheduler、所有rank RNG、逻辑计数、原chain、old probability、advantage和behavior identity，结果PASS。耗时本来会不同，因此不要求时间计数相等；不把时间排除解释为物理重算费用消失。

另一次两卡真实资格训练使用 `--fault before_finalize --fault-rank 1`，让16条真实轨迹完成评分后、global advantage提交前故意失败。训练进程非零退出是该测试的预期结果；随后实际执行：

```bash
cd /mnt/project/VLA-Drive-qwen-sourceflow-native-20261018
CUDA_VISIBLE_DEVICES='' "$QSF_PY" -m scripts.analysis.qwen_sourceflow.audit_collection_failure \
  --run "$QSF_ROOT/qualification/RL_failure_before_finalize" \
  --dispatch "$QSF_ROOT/dispatch/native_failure_before_finalize" \
  --output "$QSF_ROOT/validation/actual_native_collection_failure.json"
```

审计证明最新合法恢复点仍为READY/update0，物理评分16、已提交算法候选0，失败链没有advantage，也没有COMPLETE标记。此审计只读，不改写原失败目录。

正式RL入口同一模块，但不得带 `--qualification`：必须提供完整登记IL checkpoint和 `--parent-selection frozen_IL_parent.json`。runtime检查全部五个完整DEV及immutable父模型；因此现在不能用资格模型提前启动正式50k。

## 已执行的合并与独立部署

```bash
cd "$QSF_RELEASE"
CUDA_VISIBLE_DEVICES='' "$QSF_PY" -m scripts.analysis.qwen_sourceflow.export_policy \
  --checkpoint "$QSF_ROOT/qualification/RL_v4/checkpoints/update000004_inner0" \
  --output "$QSF_ROOT/qualification/RL_v4_merged"
```

导出保留完整Qwen/provider、原生动作专家及小型processor；LoRA B@A以FP32合并，输出包含hash和COMPLETE。已有output拒绝覆盖。部署bundle不等于可精确恢复的训练checkpoint，恢复必须用未合并LoRA＋Adam＋原链。

原始图像单候选10步ODE连通评价已经在真实TRAIN资格集执行：

```bash
cd "$QSF_ROOT/snapshots/native_runtime_v4"
CUDA_VISIBLE_DEVICES=1 "$QSF_PY" -m scripts.analysis.qwen_sourceflow.evaluate_policy \
  --checkpoint "$QSF_ROOT/qualification/IL_v3_continuous/checkpoints/update000004" \
  --output "$QSF_ROOT/qualification/evaluation_TRAIN2_v3" \
  --cohort TRAIN --data "$QSF_ROOT/data/TRAIN_v3" --qualification --limit 2
```

完整评价禁用limit；数据、噪声seed、协议、人口hash及模型身份必须匹配。每个candidate独立对官方参考评分，同一仿真分项重组source和canonical读出，不把两者当相同指标。

## 旧里程碑队列（v2历史记录）

以下v2命令为原执行记录。2026-10-20核查发现20k的55个worker全部成功，但控制器在完成receipt/liveness收尾竞争中退出；原目录与failure保留。当前实际owner为下文v3，不再启动v2。

```bash
cd "$QSF_IL_SOURCE"
CUDA_VISIBLE_DEVICES='' "$QSF_PY" -m scripts.analysis.qwen_sourceflow.milestone_queue \
  --source "$QSF_IL_SOURCE" --il-run "$QSF_ROOT/runs/IL_seed42_v2" \
  --control-dir "$QSF_ROOT/milestone_queue_v2"
```

同一入口 `--once` 已实际执行，之后单一owner持续等待。不同shard使用scene绑定的同一seed2101；分配前重新读取live授权，检查两次显存/真实owner和物理UUID，未完成的scene必须补齐才可汇总。每次invocation分开记录new/reused、forward、评分等待、加载和worker wall；worker wall不是独占GPU compute。

队列完成全部五个DEV后实际调用 `freeze_il`，锁定父模型；输出 `READY_FOR_RL.json`。这只表示IL选择前置完成，不表示两组RL或Navtest已经执行。正式RL成本profile、两组50k和最终Navtest仍需后续任务收据证明。

## 当前里程碑队列v3：已执行恢复

实际CPU调度源码为不可变`990e570`，推理源码仍为`d4d0012`，摘要/选择源码为`b784e82`。原1696份20k结果仅补汇总，没有重新推理。实际执行`--once`后，单一owner后台运行：

```bash
cd /mnt/project/qwen-sourceflow-native-20261018/snapshots/milestone_receipt_fix_20261020
CUDA_VISIBLE_DEVICES='' "$QSF_PY" -m scripts.analysis.qwen_sourceflow.milestone_queue \
  --source "$QSF_IL_SOURCE" \
  --analysis-source "$QSF_ROOT/snapshots/campaign_v2" \
  --il-run "$QSF_ROOT/runs/IL_seed42_v2" \
  --control-dir "$QSF_ROOT/milestone_queue_v3"
```

该命令记录当前已运行任务，不要重复启动。`milestone_queue_v3/{identity,status,owner,controller_launch,SOURCE_MIGRATION}.json`保存实际身份；20k已识别COMPLETE，等待登记40k，依旧不自动重试失败worker。CPU竞争测试实际执行：

```bash
CUDA_VISIBLE_DEVICES='' "$QSF_PY" -m scripts.analysis.qwen_sourceflow.validate_milestone_receipts \
  --output "$QSF_ROOT/validation/milestone_receipt_race_20261020.json"
```

结果7项PASS。下一轮重测须用新的输出文件名保留原证据。

## 旧后续owner v2（历史记录）

2026-10-20的用户中期Navtest发现原评价入口只配置TRAIN/DEV缓存。下面v2 owner命令为历史记录；旧目录保留，已在尚无子任务时显式迁移为文末v3。正式训练源码不变。

当前 `campaign_v2` 的实际源码是不可变 `b784e82`，训练与评价源码仍固定为 `d4d0012`。CLI `--once` 已实际执行，随后单一CPU owner后台运行；重复启动被owner lock拒绝。当前等待完整IL和五个DEV，没有任何提前派发的正式RL子任务。

```bash
QSF_CAMPAIGN_SOURCE=$QSF_ROOT/snapshots/campaign_v2
cd "$QSF_CAMPAIGN_SOURCE"
CUDA_VISIBLE_DEVICES='' "$QSF_PY" -m scripts.analysis.qwen_sourceflow.campaign \
  --training-source "$QSF_IL_SOURCE" --analysis-source "$QSF_CAMPAIGN_SOURCE" \
  --control-dir "$QSF_ROOT/campaign_v2"
```

上述v2命令是历史owner记录，不要再启动。旧只读状态为 `campaign_v2/status.json`、`identity.json`、`controller_launch.json`。源变更收据在 `SOURCE_MIGRATION.json`：早期仅等待的owner在没有子任务时被替换，原目录与不可变源码保留；正式IL没有中断。

它复用 `train_rl`、`export_policy`、`validate_rl_deployment`、`evaluate_policy`、`summarize_results` 和 `analyze_paired`。正式50k必须核对八rank完整边界、真实Adam/scheduler=50000、12500份fresh rollout/100000场景曝光/800000候选，以及两组实际场景流hash。每个RL导出需先完成raw-image 10步ODE的未合并/合并数值验收，才能进入DEV或Navtest。CPU导出一次一个，GPU加载前即预留物理UUID。

12项CPU调度测试已执行：

```bash
cd "$QSF_CAMPAIGN_SOURCE"
CUDA_VISIBLE_DEVICES='' "$QSF_PY" -m scripts.analysis.qwen_sourceflow.validate_campaign \
  --output "$QSF_ROOT/validation/campaign_v4.json"
```

正式RL中断时，owner会停止新派发并保留失败/恢复边界，不自动生成新链或重置Adam。实际恢复仍使用前面已验收的同run `train_rl --resume` 入口；需要先核对COMPLETE/阶段、未合并LoRA、旧链/优势、source identity和原worker退出状态。

## 当前后续owner v3与中期Navtest

2026-10-20按用户要求的20k中期全量Navtest已完成，结果和实际48-rank命令见[报告](interim_NAVTEST_IL020000_20261020.md)。它不伪造冻结IL选择文件，不加入qualification标签，也不影响已登记的五个DEV父模型选择。

确认旧CPU owner仍在等待且没有子任务后，仅向经PID/start身份核对的旧CPU owner发送终止信号。实际执行新的`--once`并后台运行：

```bash
cd "$QSF_ROOT/snapshots/post_IL_evaluation_fix_20261020"
CUDA_VISIBLE_DEVICES='' "$QSF_PY" -m scripts.analysis.qwen_sourceflow.campaign \
  --training-source "$QSF_IL_SOURCE" \
  --analysis-source "$QSF_ROOT/snapshots/campaign_v2" \
  --evaluation-source "$QSF_ROOT/snapshots/interim_navtest_cachefix_20261020" \
  --control-dir "$QSF_ROOT/campaign_v3"
```

这是已运行owner的记录，不要重复启动。controller为`f23c1ce`、training仍为`d4d0012`、evaluation为已实际完成全Navtest的`777a9bd`、analysis仍为`b784e82`。四种身份分别记录；修复没有进入正式trainer。`campaign_v3/{owner,status,identity,controller_launch,SOURCE_MIGRATION}.json`保存真实状态与祖先，目前仍等待完整IL和五个DEV。15项CPU调度测试PASS，IL和`milestone_queue_v3`没有中断。
