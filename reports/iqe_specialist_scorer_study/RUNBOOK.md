# Fixed specialist / scratch Scorer campaign

This campaign is already active. The commands below are stage/recovery entry
points, not permission to duplicate live processes. Check the corresponding
runtime STATUS/LAUNCH and process identity before running a stage.

Repository: `/mnt/project/DriveVLA-M0-iqe-specialist-scorer-study`.
Branch: `research/iqe-specialist-scorer-transfer-study`.
Root: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
Config: `configs/iqe/specialist_scorer/study.yaml`.
Base: `/mnt/project/iqe-remote-rolling-cache-20261012/continuation/step_070000.pt`,
SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`.
Base binding is immutable and resolved once. No Base training or locked22k
identity was changed. No campaign Navtest/ACTIVE/push is authorized or performed.

Current observation, 2026-10-14 15:30 UTC: all six experts completed5000
updates. The frozen K4 pool contains Base, E11 3000EMA, interaction3000raw,
and safe_progress1000raw. The latter is raw, not EMA. Stage_val1024/53logs
Base is91.39146%, pool Oracle96.52152%; these are not test or Scorer results.
The core paired set is125 scenes from full D_E74311; matched core H_val is
only1 scene, insufficient for a specialty-specific generalization claim.
Extensions use116/393 training scenes and9/11 hard validation scenes.

The sole active completion controller is now rl-zt2 PID757988, GPUs4–7.
Its immutable launch receipt is
`runtime/pool_recovery_20261014T1459Z/LAUNCH.json`. The old controller failed
when it attempted to regenerate already-scored validation bytes; the fixed
pool reads strictly verified original checkpoint/trajectory/official receipts
instead. See `COMPLETED_VALIDATION_REUSE_FIX.md`. No score, target, tolerance,
checkpoint or endpoint selection rule was rewritten. Old waiting role PIDs
were checked idle before replacement; no expert was retrained.

All20,325 spatial contexts are COMPLETE. Deployment stage_val, selector_cal
and dev banks are COMPLETE; D_S16581 is still building. Scorers are waiting
for that full bank, not already trained. The controller schedules rich bank
and S00/S10, followed by S01/S11 when rich labels are complete, all on the
permanent rl-zt2 node. Actual phase/PID state is in
`runtime/completion_driver/STATUS.json` and `runtime/bank_parallel/STATUS.json`.
Physical capacity and unknown ownership are checked before each GPU launch.
Do not restart the old local/zt5/zt6 controllers or completed cache commands.

Temporary acceleration uses16 GPU workers and8 CPU helpers under the exact
15:07:18–23:07:18 UTC lease. It only precomputes the already-budgeted six
raw intermediate columns and assists existing official score jobs; no new
expert, Scorer, candidate recipe or optimizer update is introduced. See
`TEMPORARY_RESOURCE_EXECUTION.md` for receipts and absolute drain deadlines.

## Setup and immutable inputs

On an allocated host with the verified environment:

```bash
cd /mnt/project/DriveVLA-M0-iqe-specialist-scorer-study
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli --help
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli research-preflight --config configs/iqe/specialist_scorer/study.yaml
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli bind-base --config configs/iqe/specialist_scorer/study.yaml
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli audit-historical-subsets --config configs/iqe/specialist_scorer/study.yaml
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli audit-target-opportunities --config configs/iqe/specialist_scorer/study.yaml
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli select-pilot --config configs/iqe/specialist_scorer/study.yaml
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli build-paired-expert-data --config configs/iqe/specialist_scorer/study.yaml
```

Roles, scenes, observation hashes, targets and scoring protocol are checked at
their actual consumers. D_E and D_S are log-disjoint; D_S16581 is new-expert
heldout, but Base has seen it. Stage_val1024 selects endpoints, selector_cal1024
only optional margin, fixed dev1696/16logs only final reporting. Final_test is
not accessed. Historical completed Navtest is descriptive only.

## Spatial context recovery

The frozen source has W144×2048, A8×2048 and three-view final merged V. The same
20,325-scene topology remains `rlzt2_six_v1` even after moving workers to zt5.
Its name is an immutable plan ID, not a host requirement. Every worker holds an
exclusive lock; don't launch a second process waiting on the same live worker.
Example recovery for worker4 on its now-assigned zt5 GPU3:

```bash
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.context_parallel worker --config configs/iqe/specialist_scorer/study.yaml --topology rlzt2_six_v1 --worker 4 --device cuda:0
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.context_parallel verify --config configs/iqe/specialist_scorer/study.yaml --topology rlzt2_six_v1
```

Worker0/2 completed on rl-zt2; worker1/5 completed after exact original-batch
replay on zt5; worker3/4 also completed. Whole-cache COMPLETE verified all six
receipts and exact full-scene coverage. A changed batch shape caused six
language-context mismatches. The verified original batch plan resolves them to
exact raw/W/A equality; tolerance1e-5, capture source and old cache are unchanged.
See `evidence/ORIGINAL_BATCH_REPLAY.json`. No fallback changes precision or labels.

## Expert stages

Six attempts maximum, each5000 updates/global128/micro16/warmup100/cosine-to0.1,
AdamW1e-4/weight_decay0/clip1/EMA0.999FP32. The four pilot cells share the same
125 scenes and sample order. Raw/EMA at1000/3000/5000 are snapshots, not extra
experts. Commands for each existing cell (only after confirming no matching live run):

```bash
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell E00 --device cuda:0 --resume auto
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell E01 --device cuda:0 --resume auto
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell E10 --device cuda:0 --resume auto
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell E11 --device cuda:0 --resume auto
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli evaluate-expert-matrix --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell EXT_interaction --device cuda:0 --resume auto
CUDA_VISIBLE_DEVICES=6 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell EXT_safe_progress --device cuda:0 --resume auto
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli freeze-candidate-pool --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
```

For a genuinely fresh new cell omit `--resume auto`; for a completed cell resume
must add zero updates. Current four completed paths include
`experts/E00/step_005000.pt` under the root. Recipe/pool selection only reads
stage_val; unused or redundant specialists are not retained to force a K.
No external action teacher or Scorer weights are loaded.

## Candidate banks and four Scorers

Only after the pool is locked:

```bash
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli build-deployment-bank --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=6 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli build-rich-bank --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-scorer-cell --config configs/iqe/specialist_scorer/study.yaml --cell S00 --device cuda:0
CUDA_VISIBLE_DEVICES=6 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-scorer-cell --config configs/iqe/specialist_scorer/study.yaml --cell S01 --device cuda:0
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-scorer-cell --config configs/iqe/specialist_scorer/study.yaml --cell S10 --device cuda:0
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-scorer-cell --config configs/iqe/specialist_scorer/study.yaml --cell S11 --device cuda:0
```

S00/S10 can start with the deployment bank; S01/S11 require the rich bank. Spatial
S10/S11 require the full spatial receipt. Banks never use GT as a deployment
candidate. Rich GT-identical optimized trajectories are excluded; legal low-score
negative actions remain. All official scores use independent fresh candidate
contexts, score scale[0,1], retries and strict checksum joins, up to8 native CPU workers per job (bank runtime route caps each role at4).

Each Scorer is random initialized,6000 updates/global128/micro16,AdamW1e-4,
weight_decay0.01,warmup100 then constantlr. First3000 natural128 pointwise;
last3000 natural64+comparison64. q∝gap/P_x; sampled ranking is not reweighted.
S01/S11 use rich supervision only in phase1, all four use the same final pool.
Same-cell interrupted resume adds `--resume auto` to the exact command. It
preserves both sampler cursors, RNG, optimizer/scheduler and q/bank identity.

## Fixed final evaluation, research export and checks

```bash
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli evaluate-scorer-matrix --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli calibrate-selection --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli evaluate-final-system --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli benchmark-system --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.reporting --config configs/iqe/specialist_scorer/study.yaml
```

Final evaluation writes raw argmax and independent-calibration margin separately,
all fixed policies, Oracle, regret/risk and rich/spatial diagnostics. It exports
`research_bundle/bundle.pt` without activating it. Benchmark includes real images,
one visual/language forward, all private branches, semantic/spatial timing and
an independent-process strict reload. Full Scorer results are still pending;
expert Oracle numbers are not reported as actual selection gains.

Already completed real checks: private-reader4-update uninterrupted/resume and
terminal behavior; official two-scene Base/GT reference invariance;256 real-image
spatial pilot;16-record lossless reader test. Required later Scorer two-stream
resume and bundle roundtrip run on actual final banks/weights. These qualification
weights do not become deployment candidates.

## Controller recovery and another Base

Only when the existing zt6 controller and its children are confirmed inactive:

```bash
/root/miniconda3/envs/ddp/bin/python scripts/iqe/specialist_study_completion.py --config configs/iqe/specialist_scorer/study.yaml --gpus 5 6 1 4
```

Its RUN.lock prevents concurrent controllers; it reuses complete stages and
resumes incomplete cells. Resource routing is separate from scientific identity.
See runtime/completion_driver/STATUS.json and runtime/tsinghua56 launch receipts.
Never blindly kill process groups or restart old queue controllers.

A different Base requires a new campaign_id/output_root/reuse_root/YAML. Choose an
explicit checkpoint in the intended Base run or base_checkpoint:latest_complete.
The `prepare-context-source --action bind/cache/score` entry creates that source's
immutable binding and context/official labels; `bind-base` then imports it once.
Use `--max-samples256` only as the actual CLI's separate arguments
`--max-samples 256` for the bounded cache pilot. Do not edit this active campaign
or reuse its W/A, candidate labels, experts or Scorers with another Base.
