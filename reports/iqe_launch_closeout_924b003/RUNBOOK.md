# IQE closeout launch and recovery runbook

Repository/worktree: `/mnt/project/DriveVLA-M0-iqe-closeout-924b003`, branch `fix/iqe-launch-closeout-924b003`, pinned start `924b003ad84ffc2d51e1cb4f5e140663f3280152`. All interfaces below were checked against this source's CLI parser/help. No command in this document was implicitly scheduled. Full training, full banking, new navtest and ACTIVE updates were not started.

Use an allowed, currently free worker. The existing Base remains on `training-vla-zt2`; do not launch another job in its output directory. Read `/mnt/project/server_dispatch_policy.json` at dispatch. GPU CLI gates reject occupied selected devices. `training-vlawm-zt` was free for the actual eight-GPU profile; it later acquired another research workload. Do not assume it is still available.

## Fixed identities and budget

```bash
cd /mnt/project/DriveVLA-M0-iqe-closeout-924b003
IQE_PY=/root/miniconda3/envs/ddp/bin/python
IQE_CONFIG=configs/iqe/formal/round1_locked.yaml
IQE_BASE=/mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_022000.pt
IQE_EVAL=/mnt/project/iqe-launch-runtime-924b003/qualification/base_evaluation/9b086ca1d178ed4d278926093320dd46354e2624f934d277031e5db0f72cc1ad/EVALUATION.json
IQE_ROOT=/mnt/project/DriveVLA-M0-iqe-closeout-924b003/outputs/iqe/formal_round1_22k
```

Base SHA256 `42320ea3e6fb09c6cd2b955dde7398a4033338a98a68bc4bac111fe8e5e01e93`. Qualification `175b61961441ccd3005a6b794ed5443198e470ff7f7cb88b612774255772ddb6`. Actual formal LOCKED_S0 exists under IQE_ROOT. A new Base must use another experiment root, not overwrite this lock.

Full data source is `/mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/scenes.json`: 90,892 fit, 5,563 stage_val, 5,137 selector_cal, 1,696 dev. `--max-samples 101592` is a **per-role ceiling**, not a request to duplicate observations; actual total is 103,288. There are no real new observations: main experiment is **resampling_only**. The training-root caches need official context preparation before score-bank generation.

Formal expert: AdamW, lr3e-4, wd0.01, global128, micro128 on one GPU, max3000 optimizer steps, warmup100, clip1, validation200, patience5. Scorer: lr1e-4, otherwise corresponding global128/3000/200 defaults. Seeds: init42, training42, sampler42, bootstrap42; stable per-module/round derivation is recorded. CPU scoring/cache workers2, torch threads8. No formal recipe was reduced to the diagnostic two-step budget. All commands resolve config into their output root.

## Base evaluation and lock (already executed)

```bash
"$IQE_PY" -m iqe.cli preflight --config "$IQE_CONFIG" --repo . --mode full --max-samples 101592 --device cpu
"$IQE_PY" -m iqe.cli evaluate-base --config configs/iqe/formal/qualification_22k.yaml --checkpoint "$IQE_BASE" --role stage_val --mode profile --max-samples 32 --device cuda
"$IQE_PY" -m iqe.cli qualify-base --config configs/iqe/formal/qualification_22k.yaml --checkpoint "$IQE_BASE" --evaluation "$IQE_EVAL" --select --mode profile --max-samples 32 --device cpu
"$IQE_PY" -m iqe.cli freeze-base --config "$IQE_CONFIG" --checkpoint "$IQE_BASE" --mode full --max-samples 101592 --device cpu
```

Preflight reports the immutable historical framework scope, whose original status remains FRAMEWORK_BOUND_QUERY_UNTRAINED; it now preserves the explicit22k checkpoint in the resolved launch config. The actual trained experiment identity is separately checked in LOCKED_S0 by freeze/use-locked.

The evaluate-base result is complete and revalidated/reused on CPU with the same command and `--device cpu`; no GPU is needed for valid cached completion. If a score failed transiently, rerun the same evaluation identity: only failed labels retry, within `base_evaluation.scoring_attempts_per_scene_per_call`. Do not delete its historical attempts. Qualification is engineering eligibility to investigate this actual trained Base; it does not establish convergence or permit serving release. No new navtest command is included.

## Formal input preparation (full jobs not executed)

Acquire a per-experiment lock in the launch shell and keep it held for the selected sequence. Check actual process cmdlines/output roots before launching; this lock prevents duplicate cooperating launchers, not already running unguarded jobs.

```bash
exec 9>"$IQE_ROOT/round1.launch.lock"
flock -n 9 || exit 2
"$IQE_PY" -m iqe.cli build-splits --config "$IQE_CONFIG" --mode full --max-samples 101592 --device cpu
"$IQE_PY" -m iqe.cli prepare-metric-contexts --config "$IQE_CONFIG" --roles incremental_fit stage_val selector_cal --mode full --max-samples 101592 --device cpu
"$IQE_PY" -m iqe.cli verify-reference --config "$IQE_CONFIG" --mode full --max-samples 101592 --device cpu
"$IQE_PY" -m iqe.cli cache-features --config "$IQE_CONFIG" --roles incremental_fit stage_val selector_cal dev_report --mode full --max-samples 101592 --device cuda
"$IQE_PY" -m iqe.cli export-candidates --config "$IQE_CONFIG" --round 0 --mode full --max-samples 101592 --device cuda
"$IQE_PY" -m iqe.cli score-candidates --config "$IQE_CONFIG" --round 0 --backend reference --mode full --max-samples 101592 --device cpu
"$IQE_PY" -m iqe.cli build-round --config "$IQE_CONFIG" --round 1 --mode full --max-samples 101592 --max-steps 3000 --device cpu
```

Dependencies, in order: strict lock and source-group audit → real official contexts → reference equivalence/invariance → full FeatureBundle cache → Base actual trajectories → official labels → remaining coverage/GT audits, fixed targets, retrieval/anchors and sampling plan. No eligible hard examples is a scientific data outcome; do not lower thresholds. Only 32 official stage_val contexts and bounded real banks were built in closeout. Full inputs remain WAITING_INPUTS.

Context preparation resumes completed logs; features and bank labels resume checksum-valid per-scene artifacts. Rerun the same command with identical config/budget. `cache-features` supports explicit `--num-shards N --shard-index I`; each shard requires separately free resources. Do not change the manifest under completed artifacts.

## First expert, Scorer and locked evaluation

After the preceding inputs pass, the resampling_only round-one experiment is engineering-qualified to proceed. No E3/E12 or demonstrated scientific gain is a prerequisite. Each GPU command below uses world_size1; only private heads train. The formal optimizer recipe remains unchanged.

```bash
"$IQE_PY" -m iqe.cli train-expert --config "$IQE_CONFIG" --round 1 --mode full --max-samples 101592 --max-steps 3000 --device cuda
"$IQE_PY" -m iqe.cli audit-frozen --config "$IQE_CONFIG" --round 1 --mode full --max-samples 101592 --device cuda
"$IQE_PY" -m iqe.cli export-candidates --config "$IQE_CONFIG" --round 1 --mode full --max-samples 101592 --device cuda
"$IQE_PY" -m iqe.cli score-candidates --config "$IQE_CONFIG" --round 1 --backend reference --mode full --max-samples 101592 --device cpu
"$IQE_PY" -m iqe.cli evaluate-oracle --config "$IQE_CONFIG" --round 1 --role stage_val --mode full --max-samples 101592 --device cpu
"$IQE_PY" -m iqe.cli train-scorer --config "$IQE_CONFIG" --round 1 --mode full --max-samples 101592 --max-steps 3000 --device cuda
"$IQE_PY" -m iqe.cli calibrate --config "$IQE_CONFIG" --round 1 --role selector_cal --mode full --max-samples 101592 --device cuda
"$IQE_PY" -m iqe.cli evaluate --config "$IQE_CONFIG" --round 1 --role dev_report --mode full --max-samples 101592 --device cuda
"$IQE_PY" -m iqe.cli evaluate --config "$IQE_CONFIG" --round 1 --role incremental_fit --mode full --max-samples 101592 --device cuda
"$IQE_PY" -m iqe.cli export-bundle --config "$IQE_CONFIG" --round 1 --require-gates --mode full --max-samples 101592 --device cuda
```

Scorer requires the entire accepted old/new candidate bank. Oracle failures do not make the Scorer code incomplete; scientific acceptance determines whether the new expert enters the accepted pool. Stage_val chooses network checkpoints; selector_cal alone calibrates; fixed dev reports locked settings. Base and previous serving bundle are both compared. Release-gate failure retains the previous serving version. The export command does **not** activate it. No `--activate` is authorized here.

## Next workflow and ordinary resume (not launched)

```bash
"$IQE_PY" -m iqe.cli train-expert --config "$IQE_CONFIG" --round 1 --mode full --max-samples 101592 --max-steps 3000 --device cuda --resume auto
"$IQE_PY" -m iqe.cli train-scorer --config "$IQE_CONFIG" --round 1 --mode full --max-samples 101592 --max-steps 3000 --device cuda --resume auto
"$IQE_PY" -m iqe.cli run-round --config "$IQE_CONFIG" --round 2 --mode full --max-samples 101592 --max-steps 3000 --device cuda --resume
```

`run-round` is one coordinator, world_size1. It preserves scientific expansion gates, remine/replay and previous serving identity. A Scorer-only workflow uses the last command plus `--selector-only`; it consumes no expansion attempt. Do not execute both variants under the same round identity. Neither command automatically activates a bundle.

Auto resume enumerates receipts, checks checksums/config/data and chooses the greatest compatible optimizer step. Stale result.json cannot supersede it. A newer corrupt/partial file produces an explicit fallback-required error; do not silently restart. Explicitly restoring an older checkpoint requires a fork output root when higher progress exists. Terminated early-stop/max-step checkpoints complete missing result/receipt with **zero** additional updates; operator-paused and ordinary checkpoints continue.

## Actual diagnostic recovery paths

These files were produced with the actual formal 22k start, real images/GT/official cache, and two optimizer updates per trainer. They are engineering diagnostics, never formal deployment models.

```bash
IQE_DIAG=configs/iqe/formal/incremental_qualification_22k_gpu.yaml
IQE_DIAG_ROOT=/mnt/project/iqe-launch-runtime-924b003/incremental_qualification_gpu
"$IQE_PY" -m iqe.cli train-expert --config "$IQE_DIAG" --round 1 --mode smoke --max-samples 32 --max-steps 2 --device cuda --resume "$IQE_DIAG_ROOT/rounds/round_001/expert/step_000002.pt"
"$IQE_PY" -m iqe.cli train-scorer --config "$IQE_DIAG" --round 1 --mode smoke --max-samples 32 --max-steps 2 --device cuda --resume "$IQE_DIAG_ROOT/rounds/round_001/scorer/step_000002.pt"
"$IQE_PY" -m iqe.cli load-bundle --config "$IQE_DIAG" --bundle "$IQE_DIAG_ROOT/rounds/round_001/bundle" --mode smoke --max-samples 32 --device cpu
```

Use a currently free GPU for CUDA commands. Existing completed diagnostics can be checksum-validated without doing further optimization. Ordinary pause recovery was actually executed from step_000001.pt before step_000002.pt existed; do not point the current higher-progress directory back at step1. `REAL_RESUME.json` records continuous-vs-resumed equality and terminal zero-update recovery.

## Actual Base execution profile and limits

The actual eight-GPU CLI profile was executed on a free `training-vlawm-zt`:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$IQE_PY" -u -m torch.distributed.run --standalone --nproc_per_node=8 -m iqe.cli train-base --config configs/iqe/performance/qualified_prefetch.yaml --mode profile --max-samples 101592 --stop-after 21032 --resume /mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_021000.pt --device cuda
```

Actual result: 32 updates, checkpoint `/mnt/project/iqe-launch-runtime-924b003/base_runtime_profile/query_base/step_021032.pt`, STOPPED_AT_BOUNDARY. To resume this diagnostic only, use the same command with `--resume /mnt/project/iqe-launch-runtime-924b003/base_runtime_profile/query_base/step_021032.pt --stop-after 21064` after resources are free. That is another bounded profile, not a formal Base. Preserve the original byte-identical fit manifest at this diagnostic output root; changing it is rejected.

Stable actual trainer p50 wall2.7666s, p953.1418s; compute2.6733s; same global32/FP32 parameters/FP32 vision/BF16 language autocast/all auxiliary losses. Controlled eight-rank A/B 4.108→2.753s and exact losses/gradients/parameters/Adam/RNG. Explicit deterministic-backward migration prevents nondeterminism from masquerading as a prefetch difference; it is not a bitwise guarantee against the old nondeterministic training backend. No precision reduction or functional pruning was used. The S-series1.735s used a different BF16/ZeRO2 execution path; current optimization does not claim that speed.

No optimized full Base was started, and the old Base was not stopped. Prefer checkpoint-preserving continuation when a fully allocated server and an explicit migration/monitoring plan are ready; restarting from zero discards useful progress. Existing navtest monitoring still targets its unchanged old run path and must not be silently redirected or duplicated.

## Tests and optional publication

```bash
"$IQE_PY" -m pytest tests/iqe -q
# Only on currently free GPUs, respecting the dispatch policy:
IQE_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0,1 "$IQE_PY" -m pytest tests/iqe/test_training_resume_ddp.py -k nccl -q
IQE_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0,1 "$IQE_PY" -m pytest tests/iqe/test_closeout_terminal.py -k nccl -q
# Not executed; publish only this new branch if explicitly requested:
git push -u origin fix/iqe-launch-closeout-924b003
```
