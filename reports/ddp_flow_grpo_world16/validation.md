# World16 paired training qualification and resource allocation

This report records the resource expansion requested on2026-09-17/18. It does
not claim an improvement in RL quality. The historical BF16 candidate-chunk1/2
failure remains FAIL; only fixed chunk1 is a supported candidate for release.
Both native semantic releases are now **READY_FOR_THIS_PROFILE** for the
explicit world16/chunk1 profile below. Formal training is running through the
paired controller; live optimizer progress is recorded separately below.

## Allocation and preserved work

| Experiment | Servers | GPU allocation | CPU allocation |
|---|---|---|---|
| F→RL | local `training-vla-zt` + `training-rl-zt4` |16 A800-SXM4-80GB,8 per node|256 logical /128 physical cores available across2 nodes|
| U→RL | `training-vla-zt2` + `training-vla-zt3` |16 A800-SXM4-80GB,8 per node|256 logical /128 physical cores available across2 nodes|

All training ranks use OMP/BLAS/MKL threads1 and one reward worker. Training
has16 reward workers per experiment,32 total; data workers remain0. Evaluation
assigns8 physical cores per GPU worker. Each node has about1TiB RAM. This is the
allocation, not a claim that every CPU core is continuously busy.

Only the16 explicitly identified `/mnt/project/gpu_stress.py` process groups
on `training-vla-zt2/3` were stopped. Their argv/PID records are preserved in
`runs/resource_reallocation_v1`. Original world4 training was stopped after both
complete update100 checkpoints and all1696-scene dev evaluations, retained at
`runs/paired_full_navtrain_v3`. The expanded runs start from their own original
SFT checkpoints at update0; this is not cross-world exact optimizer resume.

At01:47UTC, the initial F secondary node `training-rl-zt2` acquired other
container workloads on GPUs0–4. Their host PIDs were not visible in this
container; none was stopped. `training-rl-zt4`, previously occupied, was now
fully idle and substituted. Its existing9GB dependency environment was copied
from the tested local installation, with no package upgrades. The allocation
plan requires an additional actual two-update relocation run and exact full
boundary comparison before a release can be used. A changed plan selects a
new, initially unreleased source-bound namespace.

NCCL uses Socket over eth0; these containers have no InfiniBand device. The
replacement F pair completed the real16-rank reduction probe in13.01seconds.
The268MB all-reduce averaged.15316seconds (~1.75GB/s algorithmic bandwidth).
This is a communication check, not numerical actor acceptance.

## Unchanged model and experiment contract

F uses frozen_visual step100000 SHA256
`9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4`.
U uses unfrozen_visual step120000 SHA256
`72e626e152a357ec9c478c4b253500599301b38b87758e0aebfa486ff3be21cc`.
The exact source checkpoint code is
`f9449d55bea6895a7a0bd86d09d7ab85fd353f26`.

Both retain672 trainable tensors /2,233,120,260 scalar parameters: language309,
history4, projector4 and action355 tensors. Names, shapes, aliases and optimizer
groups are compared individually. Each own visual stays equal to its own SFT
initialization and excluded from the optimizer. No LoRA, additional language
freezing, auxiliary task or candidate replacement is introduced. Inactive DINO
remains inactive. Both nonvisual optimizer recipes use the same three groups
at absolute LR1e-6, with inherited matching Adam parameters and weight decay.
The frozen tied embedding/lm_head contract remains unchanged.

The actual target is `bf16_zero2_fp32_partition_v2`: BF16 actor, FlashAttention2,
ZeRO2, observed FP32 gradient accumulation/reduction/partitions, FP32 master
weights and Adam states, activation checkpointing ON, no optimizer/reference
offload, TF32 OFF. Candidate chunk1 and transition chunk1 only. All16 ranks'
observed dtype inventories are attached to each semantic evidence bundle.
The declaration alone is not treated as dtype evidence.

Fixed settings: global scene batch16; microbatch1; world16; accumulation1;
G8/K10; inner_epochs2; training seed42; noise.1; dimension-mean surrogate;
PPO clip.02; reference KL.01; original action SFT replay.1. Both reference models
remain their own complete original SFT policies. Current policy conditions are
recomputed; the saved behavior chain/old probabilities/advantages are fixed.
The600second process-group deadline replaces an inadequate120second startup
deadline; numerical tolerances are unchanged.

Each group has2000 optimizer updates, save every100, full dev every200, plus
step0/100 short-run checks. The budget means16000 fresh scene rollouts,
128000 generated candidates and32000 weighted scene replay exposures, with each
behavior batch reused twice. `inner_epochs=2` is not two dataset epochs; fresh
rollouts cover about.1575 passes of the101592-token RL train list.

Assets use the complete published v2 manifest
`04a0e93c1ea44902a4f73bf6973b236442ec5c8815ce6155c186a4d98254f420`
at `runs/paired_full_assets_v1/published_v2`, with516446 files (~112.7GB),
101592 training scenes and1696 dev scenes grouped in16 held-out logs. Navtest
has12146 scenes and never enters training/reward/replay/selection. These dev
scenes may have appeared in source SFT. NAVSIM v2 one-scene training reward has
no available two-frame comfort; official one-stage evaluation includes adjacent
scenes when available. No reward/protocol/denominator changes were made.

## Actual completed qualification

Raw runs are under `runs/resource_reallocation_v2`; compact observed values are
in `measured_qualification.json`. For each variant, full_cont, repeat, resume,
scale, rl_only, sft_only and ref_only completed with both node exit codes0.
These are real CUDA/DeepSpeed runs, each limited to at most2 optimizer updates.

| Check | F | U |
|---|---|---|
| Parameter object/alias optimizer contract |PASS|PASS|
| RL-only, SFT-only, reference-only actual optimizer gradient coverage,16 ranks ×672 tensors |PASS|PASS|
| Official reward groups with nonzero advantages |9/16; all groups retained|9/16; all groups retained|
| Own visual and complete reference unchanged, every rank/update |PASS|PASS|
| Independent CUDA Adam moments/master maximum absolute error |0 /0 /0|0 /0 /0|
| Actual forward weight elements changed after first update |34,897,003|34,947,431|
| Separate two-update repeat: all stored states exact |PASS|PASS|
| Inner-epoch boundary save/resume: all stored states exact |PASS|PASS|
| Original-interface export and fixed-noise ODE max error |0|0|
| World1 accumulation16 vs world16 accumulation1 moments, inherited1% module relative-L2 gate |PASS|PASS|
| Historical BF16 candidate chunk1/2 |FAIL, preserved|FAIL, preserved|

The largest world1/16 module moment relative-L2 discrepancy is.00367263
(.3673%), below the unchanged.01 bound. This is approximate cross-world scaling,
not an exact tensor equality claim. Exact repeat/resume checks are same-world.
All gradient comparisons include actual FP32 optimizer inputs, not only a
post-hoc `.float()` cast. The CUDA Adam comparator covers all672 parameters.

Both first pre-update ratios are exactly[1,1]. F's first post-update ratio
range is[.9561009407,1.0368417501]; U's is[.9652500153,1.0154968500]. Their second
pre-update probes equal those ranges on the same saved chain. Both inner epochs
retain identical behavior hashes on all16 ranks. Reward errors were0.

Six unchanged-component gates are explicitly reused from the previous complete
v3 evidence: high-precision chunk mathematics, CUDA source ODE/SFT oracle,
distributed control faults, global metrics, evaluation protocol and clean
checkout. Their original exit/profile/content evidence is checked against the
unchanged native executable SHA. They are not relabeled as newly executed
world16 GPU tests. Existing CUDA checkpointing ON/OFF evidence is reused with
the new actual world1/16 accumulation comparison.

## Control flow and evaluation acceleration

The native actor/NAVSIM/existing-tests digest remains
`854dbc2ecb67fb28a235ccc7238729f7560a7ccafead08d9f5307a626152153c`.
New cluster source, tests and allocation JSON are independently content-addressed.
The real publisher still checks17 semantic gates and freshly reads locked asset
bytes. Direct native commands without the current cluster environment point to
UNRELEASED. The real launchers check source-bound CPU receipts and relocation
evidence before every segment; editing code/tests/plan cannot reuse the old
cluster release. No manual READY record is created.

New tests exercise actual subprocess exit/timeout cleanup, status-write failure,
CPU affinity and GPU allocation,16-shard numeric ordering, exact boundary
observer semantics, whole-log sharding, evaluation transactions/reuse/corruption,
fixed five-seed scheduling, complete paired controller restart and native latest
checkpoint planning, process-level controller locking, occupied420MiB CUDA
contexts, changed code/test/plan invalidation, placement evidence and direct SSH
startup without PYTHONPATH. The24 new CPU tests passed, exit0. The final clean
checkout receipt passed all52 tests (24 new +28 affected native), exit0, with
zero failures/errors/skips. Its28 affected native tests come from evaluation
transactions, orchestration resume, paired evaluation and real adjacent NAVSIM
aggregation. The unaffected full157-test native suite was not repeated in this
resource-only change; its earlier v3 run remains historical evidence.

Actual F-SFT and U-step100 whole-log parallel evaluation comparisons passed for
all1696 token trajectories, all score components and the complete adjacency map.
The U measured8-GPU run took565.03seconds versus1455.82seconds before (~2.58×).
F's reused merge duration is not presented as an inference speedup. The final
five dev seeds use four disjoint four-GPU groups to avoid idle cards behind the
largest556-scene whole log; navtest uses all16 GPUs per request. Stable
seed+token noise preserves predictions independent of worker assignment.
Best==last and repeated SFT/dev/seed42 requests reuse verified complete results.

The retained original world4 dev seed42 scores are F-SFT.9362947686,
F-update100.9325597734, U-SFT.9418673185 and U-update100.9334005709. Both short
runs declined on that seed. They are separate historical results, not new-world
training initializations, not full navtest, and not evidence that RL improves.
The two SFT training steps differ; no causal claim about visual unfreezing is made.

## Failures retained

All original candidate-chunk failures and v1 startup failures are preserved.
The original120second timeout failed for F initial/repeat and U scale before
updates. The successful600second recipe was rerun completely in v2. An initial
parallel evaluation parent numerical-identity mismatch and lossy CSV parsing
failure remain recorded; corrected exact comparisons use round-trip parsing.
The first replacement-node probe failed because a clean SSH shell did not set
PYTHONPATH; no actor update ran. The fixed supervisor bootstraps its own root,
and the separate retry completed all16 ranks. A real subprocess regression
covers this failure, not a mocked import.

The initial publication attempts bound to the old allocation are obsolete after
the node/code change. Their outputs remain in the previous release namespace;
they cannot authorize the new plan. Fresh publication requires the new CPU
receipt and the additional relocation boundary evidence.

## Commands

Use the installed interpreter; do not upgrade the environment:

```bash
cd /mnt/project/DriveDreamer-Policy-paired
export PYTHONPATH="$PWD/navsim:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/root/miniconda3/envs/ddp/bin/python

# Actual source-bound CPU receipt; this does not certify GPU behavior.
$PY scripts/cluster_flow_grpo/test_release.py

# Read-only collation and native semantic publication after measured checks.
$PY scripts/cluster_flow_grpo/release.py --variant f --root runs/resource_reallocation_v2
$PY scripts/cluster_flow_grpo/release.py --variant u --root runs/resource_reallocation_v2
$PY scripts/cluster_flow_grpo/release.py --print-directory

# Start only once on a new output directory; rejects overlapping controllers.
$PY scripts/cluster_flow_grpo/paired.py --plan configs/cluster_flow_grpo/paired_world16.json
# Resume the SAME world16 run from its latest verified complete checkpoint.
$PY scripts/cluster_flow_grpo/paired.py --plan configs/cluster_flow_grpo/paired_world16.json --resume

# Export an existing completed checkpoint via the original implementation.
$PY -m starVLA.rl.flow_grpo.cli export \
  --checkpoint runs/paired_full_navtrain_world16/frozen_visual/checkpoints/update_000100 \
  --output-dir runs/paired_full_navtrain_world16/frozen_visual/export_update100
```

Already complete output directories are protected. The paired controller invokes
the original evaluator for dev and full navtest with seeds42–46; no final score
is declared before all requested tokens/seeds complete.

## Released identity and actual startup

Both native publishers exited0 and issued all17 gates as PASS under namespace
`d62de6bf255797c2945beff5dc93fea782be4ede40db38f9c4f4d7d887bc2770`.
The publication files are `releases/<namespace>/release_frozen_visual.json` and
`release_unfrozen_visual.json`. Full current assets were read and checked during
publication; each trainer also performs its own startup/resume checks.

The final orchestration code commit is
`c44829240b2185f685ab05f4031beea658a42426`, pushed normally to the existing repair
branch. The original measured world16 model runs recorded source Git SHA
`a700c959b2748991c9c298f1faca6cea8a717881`; native executable/source identities
are unchanged. Environment and processor/config hashes are exported separately
in `f_executed_environment.json` and `u_executed_environment.json`. Actual
versions are PyTorch2.5.1+cu124, Accelerate1.5.2, DeepSpeed0.16.9,
Transformers4.57.0 and flash-attn2.7.4. No dependency was upgraded.

The replacement-node two-update run passed, exit codes[0,0], followed by an
exact50-file/28,731-value checkpoint comparison, exit0. These include model,
optimizer/master, scheduler, RNG and pending chain tensors. Its spec, completed
control and exhaustive comparison hashes are bound to the release; a missing,
changed or non-PASS result rejects launch. This check is in addition to the
original world16 qualification; it does not replace it with a CPU-only test.

At02:44:55UTC on2026-09-18, the dependency pipeline executed the actual paired
controller as PID1310510. The command is the `paired.py --plan ...` entry above,
not a detached proposal or a GPU-waiting script. Its log is
`runs/resource_reallocation_v2/formal_pipeline.log`; outputs are
`runs/paired_full_navtrain_world16`. Both native releases were checked before
this launch. Startup asset validation and initial model loading precede the
first optimizer update; launch alone is not reported as a successful update.

At02:44UTC the shared volume had about1.4TiB available, close to the~1.4TiB
planned full checkpoint/export budget. Only completed diagnostic state binaries
created under this task's `resource_reallocation_v1/v2` are being moved to local
NVMe `/root/ddp-flow-grpo-diagnostic-archive/20260918` on
`training-vla-zt-worker-0`. There are749 files/993,806,519,232bytes. Each copy is
stream-hashed, fsynced and independently reread for SHA256 verification before
an atomic replacement of the original file by a symlink. Original diagnostic
paths remain readable on this host. For historical state inspection from other
hosts, restore/copy from this named archive first; these local archive links
are not claimed to be shared storage. Logs, JSON evidence, behavior buffers,
SFT sources, the prior world4 update100 and new formal checkpoints stay on the
shared volume. Nothing is discarded or relabeled PASS. The copy process is
limited to2 threads/256MiB per second total to reduce interference. Per-file
hash receipts and a completion marker record actual progress.

Engineering conclusion: **READY_FOR_THIS_PROFILE** (two identical A800 nodes,
world16/global16/accum1, fixed chunk1 and the released numerical profile).
Quality conclusion: **insufficient evidence** until the new paired budget and
complete five-seed official evaluations finish. Historical chunk>=2 remains
unsupported. Historical failures are unchanged; no seed/sample selection or
reward change was used to obtain release.

The score-corrected sampler reference remains the locked Flow-GRPO commit
`879042cf5707f8b90daa98d147d7deac2317c5da` in `reference_lock.json`.
Full asset publication stays `ASSETS_READY_ONLY`; it is not the GPU release.

The following are existing native audit/evaluation entries. Choose a new output
directory, or allow the evaluator to validate/reuse an identical completed
transaction. Do not start these GPU commands on occupied training devices.

```bash
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
WORLD_SIZE=16 $PY -m starVLA.rl.flow_grpo.cli preflight \
  --config configs/flow_grpo/paired_world16_frozen_visual.yaml \
  --max-updates 2 --output-dir reports/manual_world16_preflight \
  --set runtime.run_mode=diagnostic

# Explicit original-protocol dev inputs, here using the F SFT baseline.
$PY -m starVLA.rl.flow_grpo.cli evaluate \
  --config configs/flow_grpo/paired_world16_frozen_visual.yaml \
  --checkpoint /mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-checkpoints-v1/frozen_visual \
  --split rl_dev --tokens runs/paired_full_assets_v1/dev_tokens.json \
  --data-root runs/paired_full_assets_v1/dataset \
  --metric-cache runs/paired_full_assets_v1/metric_cache_navtrain_v2 \
  --seed 42 --metric-protocol navsim_v2_official_one_stage \
  --output-dir runs/manual_f_sft_dev_seed42

# Same protocol on complete navtest; this never enables navtest training reward.
$PY -m starVLA.rl.flow_grpo.cli evaluate \
  --config configs/flow_grpo/paired_world16_frozen_visual.yaml \
  --checkpoint /mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-checkpoints-v1/frozen_visual \
  --split navtest --tokens /mnt/project/DriveDreamer-Policy/test_meta.json \
  --data-root runs/paired_full_assets_v1/dataset \
  --metric-cache runs/metric_cache_navtest_v2 \
  --seed 42 --metric-protocol navsim_v2_official_one_stage \
  --output-dir runs/manual_f_sft_navtest_seed42
```

The actual paired controller already schedules all five seeds42–46 and both
SFT/last/best checkpoints; these manual single-seed examples do not replace it.

## Observed live training (snapshot, not completed budget)

Snapshot UTC: `2026-09-18T03:16:47.107005+00:00`. Both variants performed real optimizer updates;
the controller remains active.

| Variant | Completed updates | Mean warm update | Mean fresh rollout | Amortized seconds/update | Reward errors |
|---|---:|---:|---:|---:|---:|
|frozen_visual|11|25.742s|8.381s|29.932s|0|
|unfrozen_visual|5|27.796s|9.306s|32.449s|0|

The actual first two global losses, rewards, ratios, reference KL and pre/post
clip norms exactly match the corresponding accepted full-world16 diagnostic
runs. All16 ranks reuse the same behavior hash across the two inner epochs.
This is observed production startup consistency, not a final performance claim.
Full small summaries and hashes are in `live_training_snapshot.json`.

The early throughput gives roughly18–20 hours for the2000-update training
budget in parallel, and an initial estimate of24–30 hours including checkpoint
I/O, scheduled restarts, dev and full five-seed navtest evaluations. This is a
projection from initial real updates, not a completion guarantee or a measured
full-run duration. Compared with the old~86seconds/update, the amortized compute
rate improved by roughly2.6–2.9×. Initialization/checkpoint/evaluation overhead
is excluded from that speed ratio and included only in the broader ETA range.

The32 GPUs are actively assigned to this experiment. Training outputs remain
on shared storage; completed diagnostic state is copied to the named local
archive with byte verification. `diagnostic_archive.jsonl` is a stable link to
the live per-file receipt under the run directory, so background copy progress
does not mutate tracked report content. The live copy is separate from training
and does not issue acceptance or alter any model tensor.
