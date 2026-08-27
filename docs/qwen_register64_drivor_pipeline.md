# Qwen Register64 + DrivoR staged planning pipeline

> The PDMS-maximization route that combines exact v1.1 labels, pseudo-expert
> coverage, alternating refinement and calibrated structured/direct selection
> is documented in [register64_pdms_first_closed_loop.md](register64_pdms_first_closed_loop.md).
> This document remains the contract for the original independent G/B/S/SD/SH
> baseline and ablations.

## Scope and relation to the historical baseline

The historical `QwenPI_DrivoRSuprim` path remains intact and is still the
reference experiment:

```text
Qwen
  -> baseline action-token hidden states
  -> GlobalSceneQFormer
  -> Flow matching loss
  -> K=64 initial noises x NFE=10 Euler sampling
  -> DynamicMetricSupervisor
  -> DrivoR
  -> optional DriveSuprim
```

Register64 is a new, independent baseline. It does not replace or delete
`QwenPI_DrivoRSuprim`, `FlowmatchingActionHead`, its OFF/ON configs, or the
optimized NAVSIM metric supervisor. It removes Flow only from proposal
generation in the new path because K trajectories no longer need K independent
noise paths and repeated DiT evaluations.

## Architecture

```text
Qwen language model (baseline trainability contract)
  -> full last_hidden [B,L,2048]
  -> GlobalSceneQFormer, 16 queries
  -> global scene tokens [B,16,256]
  -> 64 learned trajectory registers [64,256]
  -> add encoded ego state [B,1,256]
  -> 4 decoder blocks
       register self-attention (1 head)
       register-to-scene cross-attention (1 head)
       256 -> 1024 -> 256 FFN
  -> one final donor MLP (256 -> 1024 -> 1024 -> 24, with LayerNorm)
     applied once to every final-layer register
  -> 64 trajectories [B,64,8,3]
```

One register represents one complete trajectory, not one pose. In the
production `stage_loss_mode: final_only` topology (equivalent to donor
`prev_weight=0.0`), the four decoder layers still return intermediate tokens
for diagnostics, but only the final-layer tokens are decoded. Consequently
there is exactly one proposal head and one proposal tensor in `proposal_list`;
there are no disconnected trainable heads. The `all_layers` ablation alone
instantiates the initial-register head plus four decoder-layer heads and
returns five proposal stages. The same implementation supports
`proposal_num: 1` for R1.

At the production dimensions, the final-only `RegisterTrajectoryGenerator`
has 5,841,176 parameters. The optional all-layers topology has 11,207,032. The
16-query Q-Former has 4,747,008 and the history-state input MLP has 4,206,592,
for 14,794,776 non-Qwen Stage-G parameters in the production topology.

There is no random proposal noise, Flow time, Euler integration,
`num_inference_timesteps`, candidate chunking, source embedding, or scorer
feedback in `RegisterTrajectoryGenerator`.

## Staged training

### Stage G: generator

Stage G trains only:

- baseline-trainable Qwen language parameters;
- the history-state input MLP;
- `GlobalSceneQFormer`;
- `RegisterTrajectoryGenerator`.

The Qwen visual tower and tied LM head/input embedding follow the existing
baseline freeze policy. The Q-Former input is not detached, so winner-take-all
trajectory loss reaches the permitted Qwen language layers. Stage G never
constructs Flow DiT, DrivoR, DriveSuprim, a static score store, or NAVSIM metric
evaluation. It has no `NAVSIM_METRIC_CACHE_ROOT` dependency.

The objective is final-stage winner-take-all three-dimensional L1:

```text
candidate_error[b,k] = mean_t ||proposal[b,k,t] - GT[b,t]||_1
loss = mean_b min_k candidate_error[b,k]
```

Diversity statistics are monitoring-only. The production diversity weight is
zero. Training is epoch based, up to 25 epochs, with a global batch of 64 and
early stopping on validation minADE@64. Before the first optimizer update, a
gradient-hook gate fails the run if any trainable parameter was absent from
the loss graph on any rank. Hooks are used because DeepSpeed ZeRO may clear or
partition `.grad` before the trainer can inspect it directly.

Every epoch runs only the low-cost geometric validation on the fixed
1,024-scene subset. Stage G selects and exports
`best_minade_generator.pt`; NAVSIM cache loading and PDM scoring start only
after that checkpoint has been frozen for Stage B.

Before a full 25-epoch run, use two explicit gates:

1. **G0 small overfit:** train on 64--256 scenes for a few hundred optimizer
   steps and confirm the loss falls and the runtime gradient gate passes.
2. **G1 matched five-epoch pilot:** compare R1 and R64 with identical data and
   optimization. Inspect minADE/minFDE together with pairwise ADE/FDE,
   active-register ratio, normalized usage entropy, and the winner histogram.
   PDM Oracle is computed after freezing the pilot checkpoint, never inside
   Stage G. These are decision gates for register collapse, not extra loss
   terms.

The independent visual-unfrozen OFF ablation keeps the same prompt, Qwen
language layers, Q-Former, Register64 generator, candidate-bank builder, and
DrivoR recipe. Its only model-boundary change is that the Qwen visual tower is
trainable while the unused tied LM head/input embedding remains frozen. Visual
parameters use LR `2e-6`; the remaining Qwen parameters keep LR `1e-5`; the
new scene/generator modules keep LR `2e-4`. Vision-block gradient checkpointing
is enabled and Stage-G global batch is reduced from 64 to 32 (16 devices x 2)
to match the established visual-tuning memory profile. Raw images are
mandatory: the launcher forcibly disables Qwen feature caches, and the model
fails fast if a cached visual payload reaches a trainable visual tower.

No hard collapse threshold or diversity regularizer is assumed before this
pilot supplies an empirical scale.

### Stage B: candidate bank

The selected generator component checkpoint is frozen and evaluated once per
scene. The complete 64-proposal tensor is submitted to
`DynamicMetricSupervisor` as one PDM pool. It is never split into independent
PDM sub-pools. CPU scoring is asynchronous, allowing the next Qwen/Register
forward to overlap with the previous NAVSIM scoring job.

The 16-rank production bank profile uses three dataloader workers and four
metric-scoring processes per rank. Including the 16 host rank processes, this
maps `16 * (1 + 3 + 4) = 128` CPU slots onto the 128-core DLC container.
`NAVSIM_NUM_WORKERS` and `NAVSIM_METRIC_WORKERS` remain explicit overrides for
a different topology.

Each distributed rank owns one resumable LMDB. The bank manifest binds every
record to the exact generator checkpoint SHA256, generator config hash,
repository commit, coordinate convention, metric schema, and optional dense
memory contract. A separate immutable build identity also binds the split,
distributed world size, datalist, metric-cache root, storage dtype, and scene
shape before the first sample is written. `--resume` fails if any of those
change; `--overwrite` removes only recognized artifacts below the exact
`.../train` or `.../val` root and refuses directories containing unrelated
files. This prevents stale LMDB keys or a changed rank topology from being
silently mixed into a new bank. The bank dataloader disables distributed
even-batch padding and the frozen model is not DDP-wrapped, so a non-divisible
final batch never creates duplicate scene tokens. The default bank stores
global scene tokens but not dense Qwen memory.

Raw tensor payload per default record is about 13.6 KiB before LMDB and
`torch.save` overhead. A practical estimate is roughly 16--22 KiB per scene;
100,000 scenes therefore need approximately 1.6--2.2 GiB. The build report
records the measured bank size.

### Stage S: DrivoR

Stage S imports only `CandidateBankDataset`, the donor-register DrivoR scorer,
and its loss. It does not instantiate or execute Qwen, Q-Former, Register64,
Flow, DriveSuprim, NAVSIM, raw images, or static vocabulary data.

Proposal geometry is explicitly detached inside `DrivoRDynamicScorer`. The
scorer uses `24 -> 1024 -> 256` trajectory embedding, `4 -> 1024 -> 256` ego
encoding, four single-head donor-register decoder layers, and the six existing
DrivoR metric heads. The existing `DrivoRMetricLoss` and
`aggregate_drivor_score` are unchanged. The main checkpoint is selected by
lowest validation regret; best selected PDMS and last checkpoints are also
written. For `N` train-bank scenes, the default run has
`5 * ceil(N / 256)` optimizer steps and the configured cap is
`10 * ceil(N / 256)`.

### Stage SD: optional dynamic DriveSuprim

```text
candidate bank -> frozen DrivoR -> predicted Top-32
               -> 3-layer DriveSuprim fine selector -> final dynamic trajectory
```

Only the fine decoder and its shared metric/imitation heads train. It uses
global scene tokens by default, so the normal candidate bank is sufficient.
The default schedule is three epochs, at most five, global batch 256.
That is `3 * ceil(N / 256)` default optimizer steps.

### Stage SH: optional static + dynamic DriveSuprim

```text
candidate bank -> frozen DrivoR -> Top-32 dynamic -> upsample 8 to 40
8192 static trajectories + 32 dynamic trajectories
  -> one shared coarse scorer over 8224 candidates
  -> global Top-256
  -> 3-layer fine selector
  -> final trajectory
```

The candidate embedding, metric heads, and decoder are shared by static and
dynamic candidates; integer provenance is used only for indexing. No source
embedding exists. Hybrid startup is gated by
`tools/validate_static_dynamic_metric_parity.py`, including vocabulary hashes,
trajectory ordering, discrete metric agreement, continuous/aggregate MAE, and
Spearman correlation. Dense fine memory is optional and fails immediately if
the bank manifest does not contain it.
The hybrid schedule uses `6 * ceil(N / 64)` optimizer steps by default and is
capped at `10 * ceil(N / 64)`.

### Independent scorer/refiner training recipes

Stage S, SD, and SH do not inherit the Stage-G optimizer or epoch schedule.
Each YAML contains a mandatory `training_profile` with the donor paper,
official repository revision, reference recipe, and the explicit bank-only
adaptation. The entrypoints reject a missing/wrong profile and reject any
bank-only config that contains a Qwen `framework` section. The paper title,
repository URL, audited donor revision, and recorded reference-recipe fields
are also validated exactly at startup, so edited or stale provenance cannot
silently launch.

| Stage | Official donor anchor | Register64 bank-only recipe | Why it differs |
|---|---|---|---|
| S / DrivoR | NAVSIM-v2: AdamW, LR `2e-4`, global batch 64, 10 epochs; 4 decoder layers, one head; selector weights NOC/DAC/DDC/TTC/EP/C=`10/13/6/14/15/2` | AdamW, LR `2e-4`, global batch 256, standalone default 5 epochs (cap 10), cosine/5% warmup; complete DLC job requests 10 | Qwen, image backbone, and generator are absent; the larger bank batch improves scorer throughput, while regret-based early stopping bounds overfit |
| SD / dynamic DriveSuprim | DriveSuprim: LR `7.5e-5`, 3-layer refinement; the published route trains the full static-vocabulary model | AdamW, LR `7.5e-5`, global batch 256, 3 epochs (cap 5), cosine/5% warmup | Top-32 fine-only is a new, smaller ablation with frozen DrivoR and no 8192-way coarse stage |
| SH / hybrid DriveSuprim | 8 GPUs x batch 8, LR `7.5e-5`, 6 epochs for ViT or 10 for CNN, one 3-layer 8192-to-256 stage | AdamW, LR `7.5e-5`, global batch 64, 6 epochs (cap 10), cosine/5% warmup | Candidate labels and scene tokens are immutable bank inputs; the project optimizer/scheduler contract replaces full-backbone Adam training |

The architecture/loss settings come from the official
[DrivoR repository](https://github.com/valeoai/DrivoR) at
`f02665403df799c1b4ddd8b0d34e073f0555c13a`, the matching action decoder in
[DriveVLA-M0](https://github.com/ZebinX/DriveVLA-M0) at
`7fabe160fc9bb41f9278845b36d457bf871f697a`, and the official
[DriveSuprim repository](https://github.com/William-Yao-2000/DriveSuprim) at
`80fe792d7654a596d92e20d030d1650f6f605c02`. The table deliberately separates
donor settings from project adaptations instead of presenting the offline
selector stages as exact reproductions of full end-to-end donor training.

## Checkpoint dependency graph

```text
Qwen base
  + Stage-G generator component checkpoint
      -> train/val candidate-bank manifests
          -> Stage-S DrivoR checkpoint
              -> optional Stage-SD checkpoint
              -> optional Stage-SH checkpoint + fixed static vocabulary
```

The Stage-G component contains only trainable Qwen weights, the history-state
MLP, Q-Former, and Register generator. Its metadata also binds the proposal
head mode and count, preventing a legacy five-head component from being loaded
into the final-only topology. Full Accelerate checkpoints separately contain
optimizer, scheduler, RNG, epoch, step, and best sparse-Oracle state for
resume. Every downstream checkpoint validates its upstream SHA/hash and
dimensions before loading. Each train stage also emits a hashed
`training_complete.json`; orchestration never mistakes an intermediate best
checkpoint from an interrupted job for a completed stage.

## Experiments

| Experiment | Generator | Selector | Config |
|---|---|---|---|
| R1 | Register1 | none | `qwen_register1_generator.yaml`, `register1_inference.yaml` |
| R64 | Register64 | proposal 0/random/geometric oracle | `qwen_register64_generator.yaml` |
| R64-O | frozen Register64 bank | true PDM oracle | `register64_candidate_bank.yaml` |
| R64-D | frozen Register64 | DrivoR | `register64_drivor_scorer.yaml`, `register64_inference.yaml` |
| R64-D-SD | frozen Register64 | DrivoR Top-32 + dynamic DriveSuprim | `register64_drivor_suprim_dynamic.yaml` |
| R64-D-SH | frozen Register64 | DrivoR Top-32 + 8192 static + DriveSuprim | `register64_drivor_suprim_hybrid.yaml` |

## Environment and commands

### Complete paired DLC jobs

The production wrappers execute the entire dependency graph and return trained
components plus strict official-score artifacts:

```text
deterministic navtrain train/val holdout
  -> Stage G Register64 (best minADE@64 checkpoint; no metric cache)
  -> build/resume NAVSIM-v2 navtrain metric cache
  -> Stage B train and val candidate banks
  -> Stage S DrivoR
  -> optional Stage SD DriveSuprim dynamic Top-32
  -> one full navtest prediction export
  -> NAVSIM v1.1 official PDMS (summary row: average)
  -> NAVSIM-v2-devkit one-stage navtest EPDMS (summary: average_all_frames)
  -> summary/summary.json, summary/summary.csv, and summary/summary.md
```

The ON arm is the clean DriveSuprim ablation: it uses the same 64 dynamic
proposals and adds only the dynamic Top-32 fine selector. It intentionally does
not introduce the 8192 static vocabulary, because that would change both the
candidate set and the module switch at once.

From a non-interactive 16-device DLC container whose checkout is under
`/mnt/zhangt_workspace/project/VLA-Drive-DDP-DRS`:

```bash
cd /mnt/zhangt_workspace/project/VLA-Drive-DDP-DRS

# OFF: Register64 + DrivoR
bash ./run_register64_drivor_off_dlc.sh

# ON: Register64 + DrivoR + DriveSuprim dynamic Top-32
bash ./run_register64_drivor_suprim_on_dlc.sh

# OFF visual-unfrozen ablation: Register64 + DrivoR
bash ./run_register64_drivor_off_visual_unfrozen_dlc.sh
```

Use a stable run id and `--resume` after a container restart:

```bash
REGISTER64_RUN_ID=register64-drivor-off-formal \
  bash ./run_register64_drivor_off_dlc.sh --resume

REGISTER64_RUN_ID=register64-drivor-suprim-on-formal \
  bash ./run_register64_drivor_suprim_on_dlc.sh --resume

REGISTER64_RUN_ID=register64-drivor-off-visual-unfrozen-formal \
  bash ./run_register64_drivor_off_visual_unfrozen_dlc.sh --resume
```

For the visual-unfrozen job, run the non-mutating contract check and hardware
preflight before formal training:

```bash
REGISTER64_RUN_ID=register64-drivor-off-visual-unfrozen-formal \
  bash ./run_register64_drivor_off_visual_unfrozen_dlc.sh --dry-run

REGISTER64_RUN_ID=register64-drivor-off-visual-unfrozen-formal \
  bash ./run_register64_drivor_off_visual_unfrozen_dlc.sh --preflight-only
```

`--dry-run` prints every stage without importing Python or writing files;
`--preflight-only` checks the branch, configs, Qwen/data paths, image-path
relocation, device count, and a BF16 accelerator operation without training.
Missing navtrain-v2, navtest-v1.1, and navtest-v2 metric caches are built with
96 CPU workers by default. The navtrain-v2 cache phase begins only after Stage
G has completed, so cache preparation cannot delay the first training log or
occupy the 16-device training allocation before the generator. Set the paths
from `env.local.example.sh` to reuse strictly versioned immutable caches, or set
`REGISTER64_BUILD_CACHES=0` to require that all three already exist. A training
feature cache is used only when `REGISTER64_TRAIN_FEATURE_CACHE_ROOT` is
explicit; navtest never aliases that cache. The production pipeline requests
the full 10-epoch DrivoR v2 schedule by default; set
`REGISTER64_DRIVOR_EPOCHS=5` only for the shorter bank ablation.

The requested second score is specifically the complete **navtest one-stage**
EPDMS produced by the repository's NAVSIM-v2 devkit. It is not the separate
two-stage `navhard_two_stage` challenge protocol, which requires synthetic
follow-up scenes not present in a navtest-only run.

### Why there is one scorer, not separate PDMS/EPDMS scorers

The default candidate bank and selector labels are generated once with the
NAVSIM-v2 metric schema. DrivoR predicts its six interpretable safety,
compliance, progress, and comfort terms; DriveSuprim predicts the expanded v2
terms. PDMS and EPDMS are then two frozen-model evaluation protocols, not two
heads trained on the navtest set. Both official evaluations therefore consume
the exact same exported `[8,3]` trajectory files and checkpoint.

Training separate scorers is meaningful only as an explicit label-domain
ablation: construct one candidate bank with the v1.1 cache and a second bank
with the v2 cache, train independent checkpoints, and never compare them as a
pure DriveSuprim ON/OFF switch. The official DrivoR project likewise publishes
separately trained NAVSIM-v1 and NAVSIM-v2 model recipes rather than selecting
a scorer on both test protocols. The present main baseline is v2-trained and
reports both generalization scores.

### Manual stage commands

All DLC paths use `/mnt/zhangt_workspace`. Set the data/model artifacts first:

```bash
export QWEN_VLM_PATH=/path/to/Qwen3-VL
export DATA_ROOT=/path/to/processed/navsim
export NAVSIM_DATALIST_PATH=/path/to/train.json
export NAVSIM_VAL_DATALIST_PATH=/path/to/val.json
export VLA_OUTPUT_ROOT=/mnt/zhangt_workspace/results/Checkpoints
export REGISTER64_BANK_ROOT=/mnt/zhangt_workspace/register64_candidate_bank
```

Use a checkpoint/run-specific `REGISTER64_BANK_ROOT` when comparing multiple
generators. Reusing one root with a different checkpoint intentionally fails
the resume identity check; use a new root or an explicit `--overwrite` rebuild.

Single-process commands match the script interfaces in the configs. On the
16-GPU DLC job, prepend the same command with the site's `accelerate launch`
or `torchrun` configuration so `Accelerator` sees all ranks.

Generator R64:

```bash
python starVLA/training/train_register_generator.py \
  --config starVLA/config/training/qwen_register64_generator.yaml
```

Matched R1 uses the identical stage and head implementation:

```bash
python starVLA/training/train_register_generator.py \
  --config starVLA/config/training/qwen_register1_generator.yaml
```

After geometry-based generator selection, bind the Stage-B metric cache and
checkpoint:

```bash
export NAVSIM_METRIC_CACHE_ROOT=/path/to/navsim/metric_cache
export REGISTER64_GENERATOR_CHECKPOINT=/mnt/zhangt_workspace/results/Checkpoints/qwen_register64_generator/best_minade_generator.pt

python starVLA/training/build_register_candidate_bank.py \
  --config starVLA/config/training/register64_candidate_bank.yaml \
  --split train --resume --backend process --workers-per-rank 4

python starVLA/training/build_register_candidate_bank.py \
  --config starVLA/config/training/register64_candidate_bank.yaml \
  --split val --resume --backend process --workers-per-rank 4
```

DrivoR:

```bash
python starVLA/training/train_register_drivor.py \
  --config starVLA/config/training/register64_drivor_scorer.yaml

export REGISTER64_DRIVOR_CHECKPOINT=/mnt/zhangt_workspace/results/Checkpoints/register64_drivor_scorer/best_regret.pt
```

Dynamic DriveSuprim:

```bash
python starVLA/training/train_register_suprim.py \
  --config starVLA/config/training/register64_drivor_suprim_dynamic.yaml
```

Hybrid parity and training:

```bash
export SUPRIM_VOCAB_PATH=/path/to/static_vocab.npy
export SUPRIM_STATIC_SCORE_CACHE=/path/to/static_scores
export SUPRIM_SCORE_CACHE_VOCAB_SHA256=<sha256>
export SUPRIM_PARITY_REPORT=/mnt/zhangt_workspace/register64_parity_report.json

python tools/validate_static_dynamic_metric_parity.py \
  --config starVLA/config/training/register64_drivor_suprim_hybrid.yaml

python starVLA/training/train_register_suprim.py \
  --config starVLA/config/training/register64_drivor_suprim_hybrid.yaml
```

Integrated R64-D evaluation:

```bash
python starVLA/training/evaluate_register_planner.py \
  --config starVLA/config/training/register64_inference.yaml
```

R1 and optional selector inference use their independent configs:

```bash
python starVLA/training/evaluate_register_planner.py \
  --config starVLA/config/training/register1_inference.yaml

export REGISTER64_SUPRIM_DYNAMIC_CHECKPOINT=/path/to/suprim_dynamic/best_regret.pt
python starVLA/training/evaluate_register_planner.py \
  --config starVLA/config/training/register64_suprim_dynamic_inference.yaml

export REGISTER64_SUPRIM_HYBRID_CHECKPOINT=/path/to/suprim_hybrid/best_regret.pt
python starVLA/training/evaluate_register_planner.py \
  --config starVLA/config/training/register64_suprim_hybrid_inference.yaml
```

Inference constructs Qwen, Q-Former, generator, and learned selectors only. It
does not read the candidate bank, ground truth, NAVSIM metric cache, or static
score cache. Hybrid inference reads only the fixed static trajectory vocabulary.

## Metrics and performance

Stage G logs minADE/minFDE at 1 and 64, pairwise ADE/FDE, active-register ratio,
usage entropy, throughput, seconds per step, epoch time, samples, and peak
memory. It does not load or validate a NAVSIM metric cache. Stage B reports
Oracle@64 PDMS, proposal-0 PDMS, oracle gain, feasible rate, register usage,
diversity, finite-label rates, storage size, and wall time. Stage S/SD/SH
report true selected score, oracle, regret, pairwise ranking accuracy,
Recall@1/5/10/32, and refinement gain where applicable.

Run the hardware comparison with:

```bash
python tools/benchmark_register_generator.py \
  --flow-config starVLA/config/training/qwenpi_drivor_suprim.yaml \
  --register-config starVLA/config/training/qwen_register64_generator.yaml \
  --batch-size 1 --candidate-chunk-size 32 \
  --output /mnt/zhangt_workspace/register_generator_benchmark.json
```

The benchmark uses the production Flow K64/NFE10 and Register64 dimensions,
the same synthetic Qwen hidden/scene tensors, device, and dtype. It reports
generator latency, peak memory, proposals per second, post-Qwen end-to-end
latency, Stage-G backward time, projected 25-epoch time, and Stage-S backward
time. No speedup value should be quoted until this command has run on the
target hardware.

### Local structural benchmark (rerun 2026-08-26)

The required production-dimension benchmark was run on one local PPU-ZW810E
(96 GiB), BF16, batch 1, candidate chunk 32, with two warmups and ten measured
inference iterations. Both proposal generators used initialized weights; the
result therefore measures architecture/runtime cost, not model quality. Common
Qwen backbone execution is excluded. Register64 used the production
`final_only` topology with one proposal head and 5,841,176 parameters.

| Measurement | Flow K64/NFE10 | Register64 | Ratio |
|---|---:|---:|---:|
| Generator forward | 245.32 ms | 2.18 ms | 112.72x faster |
| Post-Qwen hidden + Q-Former + generator | 240.25 ms | 4.68 ms | 51.37x faster |
| Proposals/s | 260.88 | 29,406.55 | 112.72x |
| Incremental peak allocation | 39.37 MB | 1.48 MB | 26.69x lower |

Total peak allocated memory (with both models resident in the benchmark
process) was 1.691 GB for the Flow call and 1.662 GB for Register64; incremental
allocation is the more informative generator comparison. A five-iteration
backward probe measured 11.37 ms/step for the standalone Register generator and
11.61 ms/step for standalone DrivoR at batch 1. Epoch projections in the raw
artifact use the benchmark's synthetic 1,000 steps/epoch and are not a dataset
runtime forecast. The existing artifact path was retained when rerunning after
the donor proposal-head correction; raw values are in
`docs/register64_benchmark_local_20260824.json`.

## Historical 100k joint training

The earlier 100k-step experiment remains a valid historical comparison but is
not the training recipe for Register64. Register64 trains the generator for at
most 25 epochs, materializes candidates once, trains DrivoR for 5--10 epochs,
and trains DriveSuprim only when that optional ablation is requested. This
prevents repeated Qwen/Flow/NAVSIM work during scorer optimization and makes
each component independently measurable and replaceable.
