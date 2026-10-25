# 执行、恢复与产物核验

当前工作区是 `/mnt/project/VLA-Drive-qwen-flow-world-postcot-500m-20261025`。
正式运行使用封存的集成源码；`SOURCE_BINDING.json` 记录实际 commit，工程验收中的
`source_snapshot`、`source_commit` 和 `training_source_files` 必须与它一致。
不要修改正在运行的 snapshot，也不要将资格测试权重用于正式训练。

```bash
export PY=/mnt/project/flow-safe-improvement-20261013/venv/bin/python
export ART=/mnt/project/qwen-world-postcot-20261025
export SOURCE="$ART/snapshots/formal_main_v2"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
```

下文是可执行的阶段入口，不是一组需要全部重新运行的命令。已有全量候选库和
DEV 教师缓存应直接复用；TRAIN 教师生产由现有 worker/watcher 继续完成。
先检查 `campaign/STATE.json`、`runtime/*/CURRENT_WORKERS.json` 和实际 job 的
`RUNNING.json` / `RESULT.json`，并核实 PID 启动时间。不要根据旧 RUNNING 字样
重复派发。只有确认原 worker/子进程退出，才恢复同一训练或生产分片。

## 推荐：一个后台协调器推进完整实验

在永久授权主机上，封存源码存在后执行一次。若协调器已经运行，不再启动第二个；
原协调器已核实退出时，同一命令从已有 STATE 继续，不创建新的训练 lineage。

```bash
test -f "$SOURCE/SOURCE_BINDING.json"
cd "$SOURCE"
mkdir -p "$ART/campaign"
nohup "$PY" -u -m tools.qwen_world_postcot.campaign \
  --source "$SOURCE" --execute-ready-stages --loop --interval 60 \
  >> "$ART/campaign/controller.log" 2>&1 &
```

协调器等待真实 `audits/ENGINEERING_ACCEPTANCE.json` 和完整监督，不跳过验收。
随后执行：100k IL（含 1k 接口 warm start）→每 5k 完整 DEV→冻结最高 DEV 父本
→该父本的短程真实 PDM RL 工程检查→独立正式 50k RL→固定 RL 端点完整 DEV
→冻结 IL/RL 模型→完整 NAVTEST。RL DEV 端点为 5k/10k/20k/35k/50k。
资格 RL 不作为正式 RL 的初始化权重。主采样 seed2101 单列；冻结模型后额外的
2102/2103 是推理随机性分析，不是三个训练 seed，也不用于重选模型。

```bash
cd "$SOURCE"
"$PY" -m tools.qwen_world_postcot.update_status --output "$ART/campaign/RUN_STATUS.json"
cat "$ART/campaign/STATE.json"
cat "$ART/runs/IL_full_seed42/MONITOR.json"
cat "$ART/runs/RL_full_CANONICAL_seed42/MONITOR.json"
```

未启动的 run 可以没有 MONITOR 文件。完整训练不是 smoke：IL global32、micro4、
FM repeat1、100k 原时间轴；OOM 时仅采用已验收的 micro2×accum2。IL 阶段准确称为
“轨迹模仿学习＋表征辅助监督＋行动评价辅助监督”，`uses_metric_supervision=true`。

## 固定数据和教师缓存

已经封存的 TRAIN 是 90,890 场景；DEV 为 1,696 场景。TRAIN 排除了 DEV/NAVTEST
的整条物理日志。不要用旧 101,592 场景 manifest 替代，也不要重写现有 records。
正式目标仅使用 `TRAIN_resolved_v3` / `DEV_resolved_v3`；旧镜像不完整的 TRAIN、
canonical_v2 永不作为正式完整缓存。

以下是实际 TRAIN producer 的 rank0 命令。它属于现有 **21 个跨主机分片**，不是
在一台服务器再开 21 个进程。GPU 由 job_worker 的 CUDA_VISIBLE_DEVICES 映射。
只在旧 writer 已退出且调度器明确分配该 rank 后恢复；全局已提交 token 会跳过。

```bash
cd "$ART/snapshots/teacher_resolved_v3_final_20261025T160330Z"
"$PY" -m tools.qwen_world_postcot.teacher_cache build \
  --population "$ART/data/TRAIN/manifest.json" \
  --root "$ART/teachers/TRAIN_resolved_v3" \
  --dino-root /mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm \
  --wa-source "$ART/assets/WA-JEPA-source" \
  --wa-checkpoint "$ART/assets/wa-jepa-model_state_dict.pt" \
  --device cuda:0 --rank 0 --world-size 21 --max-scenes 4500 \
  --max-seconds 22000 --shard-records 256 --image-lru 64 --cpu-threads 2 \
  --priority-scenes "$ART/data/TRAIN_PROFILE_PRIORITY1024.json"
```

DEV 已完成，通常只需要核验。下面保留它的完整生产命令用于重建或恢复，不要重复
启动现有已完成任务。更改 world-size 前须让所有旧 writer 在原生边界退出。

```bash
"$PY" -m tools.qwen_world_postcot.teacher_cache build \
  --population "$ART/data/DEV/manifest.json" \
  --root "$ART/teachers/DEV_resolved_v3" \
  --dino-root /mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm \
  --wa-source "$ART/assets/WA-JEPA-source" \
  --wa-checkpoint "$ART/assets/wa-jepa-model_state_dict.pt" \
  --device cuda:0 --rank 0 --world-size 1 --max-scenes 1696 \
  --max-seconds 7000 --shard-records 128 --image-lru 64 --cpu-threads 2
"$PY" -m tools.qwen_world_postcot.teacher_cache verify \
  --population "$ART/data/TRAIN/manifest.json" \
  --root "$ART/teachers/TRAIN_resolved_v3" --sample-records 64 --finalize
"$PY" -m tools.qwen_world_postcot.teacher_cache verify \
  --population "$ART/data/DEV/manifest.json" \
  --root "$ART/teachers/DEV_resolved_v3" --sample-records 64 --full-shards --finalize
```

不齐时 finalize 非零退出；目录存在不算 COMPLETE。TRAIN 的完整 shard checksum
扫描约数百 GB，仅在最终签收或明确完整性诊断时加 `--full-shards`，不要每步重复。
当前 producer/watcher 的实际 argv 与资源归属见 `runtime/teacher_TRAIN_resolved_v3*`。

## 完整候选库：复用、恢复、验证

`candidates/full_train_initial_v1` 已 COMPLETE：90,890×16=1,454,240 条，hash
`356d24a0e8e9e6095a9dd1f8d2d96fe39394ca51408124f6d51b62cd99f95b7f`。
已有 own-native 索引 `candidates/own_native_complete.sqlite` 不需重新扫描 200k 行为文件。
真实 GT 也经过评分；坏候选仅供行动评价，不能进入主 imitation loss。

这是生产时的16个固定分片的完整恢复命令，使用实际封存源码、原 FULL_BUILD_ORDER、
原16候选预算和输出。示例仅用于 `training-vla-zt` 上原CPU112–127配额，先核实所有
旧writer已退出；不在另一主机照搬CPU亲和。已经COMPLETE的相同分片会复用。
新机器应先按授权CPU资源重新调度，不改变rank/world或消费清单。

```bash
cd "$ART/snapshots/post_production_v3"
pids=()
for rank in $(seq 0 15); do
  cpu=$((112 + rank))
  budget=$(( ((90890 + 15 - rank) / 16) * 16 ))
  nice -n 5 taskset -c "$cpu" "$PY" -m tools.qwen_world_postcot.build_candidate_bank \
    --data-root "$ART/data/TRAIN" \
    --output "$ART/candidates/full_train_initial_v1" \
    --scene-manifest "$ART/candidates/full_production_control/FULL_BUILD_ORDER.json" \
    --all-scenes --target-candidates 16 --rank "$rank" --world 16 \
    --max-simulation-budget "$budget" \
    --extra-candidates-index "$ART/candidates/own_native_complete.sqlite" \
    --reuse-bank "$ART/candidates/pilot32" \
    --reuse-bank "$ART/candidates/performance1024" --resume &
  pids+=("$!")
done
bank_status=0
for pid in "${pids[@]}"; do
  wait "$pid" || bank_status=1
done
if [ "$bank_status" -eq 0 ]; then
  "$PY" -m tools.qwen_world_postcot.build_candidate_bank \
    --output "$ART/candidates/full_train_initial_v1" --world 16 --finalize
fi
```

最终合并只在所有声明的分片 COMPLETE 后执行。需要重新核验时，使用正式集成源码
中的最新 coverage 工具，给报告新的输出名；下面不会构建大缓存或重跑训练。
coverage 全量检查 offsets、记录 checksum、FP32 坐标与时间来源修正；audit 的
264 次真实官方模拟另算成本。

```bash
cd "$SOURCE"
export AUDIT_ID=$(date -u +%Y%m%dT%H%M%SZ)
"$PY" -m tools.qwen_world_postcot.audit_candidate_labels coverage \
  --bank "$ART/candidates/full_train_initial_v1" --data-root "$ART/data/TRAIN" \
  --output "$ART/audits/candidate_coverage_$AUDIT_ID.json" \
  --workers 8 --max-records 1454240
"$PY" -m tools.qwen_world_postcot.audit_candidate_labels audit \
  --bank "$ART/candidates/full_train_initial_v1" \
  --output "$ART/audits/candidate_labels_$AUDIT_ID.json" \
  --max-scenes 8 --max-candidates 16 --max-simulation-budget 264
```

已完成的对应证据是 `candidates/full_train_initial_v1_LABEL_AUDIT.json` 和
`candidates/full_train_initial_v1_COVERAGE_AND_TIMING_PROVENANCE_v2.json`，无需为了
启动训练再做一遍。625 条错时刻 logged poses 的 provenance overlay 不能误称
合格固定 dt imitation GT；其真实候选坐标与官方评分仍有效。

## 周期候选刷新：10k 生产、5k 固定版本

协调器异步为 IL10k/20k/…/90k 各生产一次，最多九次。每 scene 用当前 snapshot
的一个 Qwen 前向生成两条独立噪声 K10 轨迹，保留十四条已核验历史候选，最多十六条。
旧分数不因新候选加入而重归一化。失败时保留原银行与重试收据，不阻塞 IL。

手动恢复示例仅用于 controller 未管理该分片、原 producer 已核实退出后；正式 source
checkpoint 必须记录自己实际消费的 `builder.bank_root/hash`。单卡 world1 的完整
预算如下；若已有 world>1 的部分产物，必须保持其原 world 和 rank，不能改成此例。

```bash
cd "$SOURCE"
"$PY" -m tools.qwen_world_postcot.refresh_candidates \
  --source-checkpoint "$ART/runs/IL_full_seed42/checkpoints/update010000" \
  --export "$ART/exports/IL_update010000" --data "$ART/data/TRAIN" \
  --output "$ART/candidate_refresh/update010000/bank" \
  --rank 0 --world 1 --max-new-candidates 181780 \
  --max-simulation-budget 545340 --resume
"$PY" -m tools.qwen_world_postcot.refresh_candidates --phase finalize \
  --data "$ART/data/TRAIN" --output "$ART/candidate_refresh/update010000/bank" \
  --world 1 --registry "$ART/candidate_refresh/REGISTRY.json"
```

只有全量覆盖才能原子发布 registry。训练在每 5k optimizer 边界锁定一个版本决定，
checkpoint 保存 actual bank path/hash、两种候选采样状态、三个 RNG 和 bank_binding；
恢复读取保存版本，不重新追随 latest registry，也不重选已经锁定的同一边界。
资格 `audits/refresh32/bank_v2` 只验证 32 场景/64 新评分的工程链路，显式禁止进入
正式 registry，不能用它替代 10k 正式 snapshot。

## 手动 IL 与原生恢复

优先交给协调器。以下仅用于**永久授权主机**上，确认 controller 不再派发、原训练
各 rank 已退出后的一次手动接管；不要与后台 controller 同时启动同一 run。

```bash
cd "$SOURCE"
"$PY" -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m tools.qwen_world_postcot.train_il --config configs/qwen_world_postcot/il.yaml \
  --output "$ART/runs/IL_full_seed42" \
  --teacher-cache "$ART/teachers/TRAIN_resolved_v3" \
  --candidate-bank "$ART/candidates/full_train_initial_v1"
```

已有进度时使用恢复命令，不再次运行上面的无 resume 入口：

```bash
export IL_CP=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' \
  "$ART/runs/IL_full_seed42/LATEST.json")
"$PY" -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m tools.qwen_world_postcot.train_il --config configs/qwen_world_postcot/il.yaml \
  --output "$ART/runs/IL_full_seed42" \
  --teacher-cache "$ART/teachers/TRAIN_resolved_v3" \
  --candidate-bank "$ART/candidates/full_train_initial_v1" --resume "$IL_CP"
```

恢复保持原模型、Adam/master、阶段、scheduler、100k 时间轴、实际 bank、RNG 与消费
cursor；不恢复预取线程读到但未消费的位置。请求边界保存时使用对应 job 的
STOP_REQUESTED，或只对已核验 PID/starttime 的本任务 trainer rank 发 SIGUSR1；
不要向 torchrun launcher 或旧实验发信号。临时节点23:14:18 UTC 开始请求退出，
23:34:18 UTC 到期；只有旧进程退出且原生 checkpoint 封存才迁到永久资源。
强制到期退出明确标 FORCED_EXIT_REVIEW_REQUIRED，不宣称未保存工作无损。

## 完整 DEV、严格导出、正式 RL

协调器会执行所有固定端点。下面是 5k IL 的实际格式；目标已存在时先读取收据，
不要重复 export 或改变其身份。评测 `--rank 0 --world 1` 在单张已分配 GPU 上执行，
完整 DEV，不是小样本资格前缀；多 worker 要共用固定 world 和独立 rank。

```bash
cd "$SOURCE"
"$PY" -m tools.qwen_world_postcot.export \
  --checkpoint "$ART/runs/IL_full_seed42/checkpoints/update005000" \
  --output "$ART/exports/IL_update005000"
"$PY" -m tools.qwen_world_postcot.evaluate --export "$ART/exports/IL_update005000" \
  --cohort DEV --data "$ART/data/DEV" --output "$ART/evaluations/DEV_IL_update005000" \
  --rank 0 --world 1 --seed 2101
"$PY" -m tools.qwen_world_postcot.summarize \
  --evaluation "$ART/evaluations/DEV_IL_update005000" \
  --output "$ART/evaluations/DEV_IL_update005000/SUMMARY.json"
```

正式 RL 只能在完整100k IL及其20个完整DEV结果后，读取冻结选择；不要求先超过某个
PDMS门槛。下面的变量直接读取真实 selection，不使用路径占位符：

```bash
export IL_SELECTED=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' \
  "$ART/campaign/IL_SELECTION.json")
"$PY" -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m tools.qwen_world_postcot.train_rl --config configs/qwen_world_postcot/rl.yaml \
  --manifest "$ART/data/RL_HARD.json" --data "$ART/data/TRAIN" \
  --il-checkpoint "$IL_SELECTED" --parent-selection "$ART/campaign/IL_SELECTION.json" \
  --output "$ART/runs/RL_full_CANONICAL_seed42" --save-every 1000
```

RL 恢复使用其自身 progress 的最新完整点，不回到短程资格模型，也不重新加载一个
更晚的 IL 父本：

```bash
export RL_CP=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' \
  "$ART/runs/RL_full_CANONICAL_seed42/progress.json")
"$PY" -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m tools.qwen_world_postcot.train_rl --config configs/qwen_world_postcot/rl.yaml \
  --manifest "$ART/data/RL_HARD.json" --data "$ART/data/TRAIN" \
  --il-checkpoint "$IL_SELECTED" --parent-selection "$ART/campaign/IL_SELECTION.json" \
  --output "$ART/runs/RL_full_CANONICAL_seed42" --save-every 1000 --resume "$RL_CP"
```

选中 RL checkpoint 的导出（IL_SELECTED 必须仍为冻结父本）：

```bash
export RL_SELECTED=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' \
  "$ART/campaign/RL_SELECTION.json")
"$PY" -m tools.qwen_world_postcot.export --checkpoint "$IL_SELECTED" \
  --rl-checkpoint "$RL_SELECTED" --output "$ART/exports/RL_selected_manual"
```

此独立 manual 输出避免覆盖 controller 已创建的正式 export。部署保留 LoRA wrappers，
strict load，不将旧 velocity 权重当 clean-x，不默认 merge 后声称 BF16 bitwise 等价。

## DEV 锁定后的完整 NAVTEST 与采样波动

使用 controller selection 中已经完整 DEV 评测过的同一 export。不能用另一个模型的
DEV 文件充数，也不能在 NAVTEST 后重选模型。NAVTEST数据复用已封存的合法输入，
不加入训练、难例挖掘或候选库。

```bash
export IL_EXPORT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["export"])' "$ART/campaign/IL_SELECTION.json")
export IL_DEV=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["evaluation"])' "$ART/campaign/IL_SELECTION.json")
export RL_EXPORT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["export"])' "$ART/campaign/RL_SELECTION.json")
export RL_DEV=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["evaluation"])' "$ART/campaign/RL_SELECTION.json")
"$PY" -m tools.qwen_world_postcot.evaluate --export "$IL_EXPORT" \
  --cohort NAVTEST --data /mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST \
  --output "$ART/evaluations/NAVTEST_final_IL" --rank 0 --world 1 --seed 2101 \
  --parent-selection "$ART/campaign/IL_SELECTION.json" --dev-evaluation "$IL_DEV"
"$PY" -m tools.qwen_world_postcot.evaluate --export "$RL_EXPORT" \
  --cohort NAVTEST --data /mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST \
  --output "$ART/evaluations/NAVTEST_final_RL" --rank 0 --world 1 --seed 2101 \
  --parent-selection "$ART/campaign/IL_SELECTION.json" --dev-evaluation "$RL_DEV"
"$PY" -m tools.qwen_world_postcot.summarize \
  --evaluation "$ART/evaluations/NAVTEST_final_RL" \
  --baseline "$ART/evaluations/NAVTEST_final_IL" \
  --legacy-baseline-binding "$ART/campaign/BASELINE_BINDING.json" \
  --output "$ART/campaign/MANUAL_PRIMARY_COMPARISON.json"
```

原生 IL77k/canonical-RL50k 已在新结果前按最大完整 step 固定，不能按分数另选；其
父本、训练人口、架构和精度差异单列。主结果仍是 K1/K10/shift5/seed2101。
附加采样 seed2102 的 IL 入口如下；2103 同样由 controller 按预登记计划执行：

```bash
"$PY" -m tools.qwen_world_postcot.evaluate --export "$IL_EXPORT" \
  --cohort NAVTEST --data /mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST \
  --output "$ART/evaluations/NAVTEST_seed2102_IL" --rank 0 --world 1 --seed 2102 \
  --post-selection-seed-variation \
  --parent-selection "$ART/campaign/IL_SELECTION.json" --dev-evaluation "$IL_DEV"
```

所有正式结果必须完整12146场景；不能用 `--qualification --limit` 冒充主表。
每个旧行复用仍验证模型、协议、当前请求和 context；异常不得写有效0分。

## RGB decoder与真实学生特征诊断

RGB decoder已完成原定2000步，固定640场景/32000次scene呈现，原生恢复1473→2000；
完成权重是`$ART/rgb_decoder/train_main_seed42/step_002000.pt`，固定DEV16重建L1
0.07258202。不要再次启动该已完成训练。最新离线helper增加输出锁、禁止覆盖、
拒绝落后恢复点和终态无更新返回；它不修改实际运行中的formal_main_v2源码。

CNN自身的旧teacher identity723a保持，因此其合法原依赖仍是`teachers/TRAIN`。
这不允许把旧root用于规划IL；规划只使用resolved_v3。以下是终态恢复的**零更新**
命令，须从最新交付工作区的helper执行；`--device cuda:0`保持原身份，终态在模型
构造前返回，不占GPU。CPU-only实测已确认checkpoint/receipt/config/status字节和mtime
不变；不要从旧1473点重跑覆盖已有2000点。

```bash
export OFFLINE_TOOLS=/mnt/project/VLA-Drive-qwen-flow-world-postcot-500m-20261025
cd "$OFFLINE_TOOLS"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.qwen_world_postcot.train_rgb_decoder   --population "$ART/data/TRAIN/manifest.json" --teacher-cache "$ART/teachers/TRAIN"   --output "$ART/rgb_decoder/train_main_seed42" --device cuda:0   --updates 2000 --batch-scenes 16 --seed 42 --save-every 100   --max-seconds 1 --allow-building-cache   --resume "$ART/rgb_decoder/train_main_seed42/step_002000.pt"
```

实际正式1000步checkpoint已硬链接归档为`assets/IL001000_INTERFACE_BOUNDARY`，
不因rolling清理丢失最初表征诊断。以下命令先核实空闲GPU、分配CUDA设备后执行，
已有产物不要覆盖。只提取当前观测和训练好的DINO readout，不运行actor、教师、GT
或未来监督。该token在看重建误差之前已按固定hash选定；不是Navtest。

```bash
cd "$SOURCE"
"$PY" -m tools.qwen_world_postcot.extract_student_features   --checkpoint "$ART/assets/IL001000_INTERFACE_BOUNDARY"   --data "$ART/data/DEV" --cohort DEV --teacher-cache "$ART/teachers/DEV_resolved_v3"   --token 2518c058a2765f66 --output "$ART/offline_student_visualization/manual_IL001000.pt"   --device cuda:0
"$PY" -m tools.qwen_world_postcot.visualize_features   --population "$ART/data/DEV/manifest.json" --teacher-cache "$ART/teachers/DEV_resolved_v3"   --decoder "$ART/rgb_decoder/train_main_seed42/step_001473.pt"   --student-features "$ART/offline_student_visualization/manual_IL001000.pt"   --token 2518c058a2765f66 --output "$ART/offline_student_visualization/manual_IL001000.png"   --device cpu --shared-pca
```

已经执行的原图/teacher/student/误差与共同PCA图位于
`$ART/offline_student_visualization/IL001000_DEV_fixed1/`，来源封条为
`VISUALIZATION_RECEIPT.json`和`FINAL_VALIDATION.json`。实际使用预先锁定1473步
冻结decoder；该单场景teacher/student RGB L1=0.060794/0.145036。不是换成2000步
decoder后仍沿用旧数字，也不是规划PDMS或checkpoint选择依据。原图不进Git。
