# Qwen Register64 + DrivoR staged planning pipeline

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
  -> one donor MLP per stage (256 -> 1024 -> 1024 -> 24, with LayerNorm)
     applied to every register
  -> 64 trajectories [B,64,8,3]
```

One register represents one complete trajectory, not one pose. The initial
register state and all four decoder layers have separate stage heads (five
heads total); each stage head is shared across registers and maps each register
token to its own complete 24-D trajectory. A forward therefore returns five
proposal stages. Production training uses only the final stage
(`stage_loss_mode: final_only`, equivalent to donor `prev_weight=0.0`). The
same implementation supports `proposal_num: 1` for R1.

At the production dimensions, `RegisterTrajectoryGenerator` has 11,207,032
parameters. The 16-query Q-Former has 4,747,008 and the history-state input MLP
has 4,206,592, for 20,160,632 non-Qwen Stage-G parameters in total.

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
evaluation during an optimizer step.

The objective is final-stage winner-take-all three-dimensional L1:

```text
candidate_error[b,k] = mean_t ||proposal[b,k,t] - GT[b,t]||_1
loss = mean_b min_k candidate_error[b,k]
```

Diversity statistics are monitoring-only. The production diversity weight is
zero. Training is epoch based, up to 25 epochs, with a global batch of 64 and
early stopping on validation minADE@64.

### Stage B: candidate bank

The selected generator component checkpoint is frozen and evaluated once per
scene. The complete 64-proposal tensor is submitted to
`DynamicMetricSupervisor` as one PDM pool. It is never split into independent
PDM sub-pools. CPU scoring is asynchronous, allowing the next Qwen/Register
forward to overlap with the previous NAVSIM scoring job.

The 16-rank production bank profile uses four dataloader workers and four
metric-scoring processes per rank: 128 CPU workers in total on the 128-core DLC
container. `NAVSIM_NUM_WORKERS` and `NAVSIM_METRIC_WORKERS` remain explicit
overrides for a different topology.

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
| S / DrivoR | NAVSIM-v2: AdamW, LR `2e-4`, global batch 64, 10 epochs; 4 decoder layers, one head | AdamW, LR `2e-4`, global batch 256, 5 epochs (cap 10), cosine/5% warmup | Qwen, image backbone, and generator are absent; the larger bank batch improves scorer throughput, while regret-based early stopping bounds overfit |
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
MLP, Q-Former, and Register generator. Full Accelerate checkpoints separately
contain optimizer, scheduler, RNG, epoch, and step state for resume. Every
downstream checkpoint validates its upstream SHA/hash and dimensions before
loading.

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

All DLC paths use `/mnt/zhangt_workspace`. Set the data/model artifacts first:

```bash
export QWEN_VLM_PATH=/path/to/Qwen3-VL
export DATA_ROOT=/path/to/processed/navsim
export NAVSIM_DATALIST_PATH=/path/to/train.pkl
export NAVSIM_VAL_DATALIST_PATH=/path/to/val.pkl
export NAVSIM_METRIC_CACHE_ROOT=/path/to/navsim/metric_cache
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

After selecting the component checkpoint:

```bash
export REGISTER64_GENERATOR_CHECKPOINT=/mnt/zhangt_workspace/results/Checkpoints/qwen_register64_generator/best_generator.pt

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
memory. Stage B reports Oracle@64 PDMS, proposal-0 PDMS, oracle gain, feasible
rate, register usage, diversity, finite-label rates, storage size, and wall
time. Stage S/SD/SH report true selected score, oracle, regret, pairwise ranking
accuracy, Recall@1/5/10/32, and refinement gain where applicable.

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

### Local structural benchmark (rerun 2026-08-25)

The required production-dimension benchmark was run on one local PPU-ZW810E
(96 GiB), BF16, batch 1, candidate chunk 32, with two warmups and ten measured
inference iterations. Both proposal heads used initialized weights; the result
therefore measures architecture/runtime cost, not model quality. Common Qwen
backbone execution is excluded.

| Measurement | Flow K64/NFE10 | Register64 | Ratio |
|---|---:|---:|---:|
| Generator forward | 223.32 ms | 2.69 ms | 83.16x faster |
| Post-Qwen hidden + Q-Former + generator | 221.96 ms | 4.99 ms | 44.51x faster |
| Proposals/s | 286.58 | 23,831.13 | 83.16x |
| Incremental peak allocation | 39.37 MB | 1.49 MB | 26.47x lower |

Total peak allocated memory (with both models resident in the benchmark
process) was 1.702 GB for the Flow call and 1.673 GB for Register64; incremental
allocation is the more informative generator comparison. A five-iteration
backward probe measured 11.85 ms/step for the standalone Register generator and
12.95 ms/step for standalone DrivoR at batch 1. Epoch projections in the raw
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
