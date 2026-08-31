# DriveVLA-M0 Stage-2 local reproduction

This setup follows the paper's two-stage boundary:

- Stage 1: restore only the 1,005 `backbone` tensors from the released Base checkpoint. The released Base VLM is valid here because Stage 2 freezes the VLM.
- Stage 2: leave every `action_head` tensor randomly initialized, including Compress/Q-Former, the 64-trajectory decoder, and the scorer; freeze the complete VLM and train only `action_head`.

The public ReCogDrive VQA checkpoint is already available locally at
`/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B`.  The dense VLM
weights in the DriveVLA-M0 checkpoint match all 685 tensors in that public
checkpoint exactly, while DriveVLA-M0 additionally stores 160 trained LoRA
A/B pairs.
Restoring all 1,005 released `backbone` tensors therefore preserves the exact
post-VQA Stage-1 state, including those adapters, without restoring any Stage-2
action tensor.

The paper's similarly named scores refer to different systems.  The ablation
table's memory-free `Base Model` is the VLM + 64-proposal action decoder +
scorer baseline and reports 91.0 PDMS.  `DriveVLA-M0-Base` (92.3) already adds
roughly 4K memory cases plus retrieval and TTT; `DriveVLA-M0-Scale` (94.1)
expands that memory to roughly 10K cases.  The released memory-free checkpoint
scores 90.9594 in the local full 12,146-scene Navtest evaluation, matching the
paper's rounded 91.0 baseline.

For a stronger open-source reference, the official ChainFlow-VLA Stage-2
submission was also rescored locally against the same metric cache and all
12,146 scenes. It obtains 0.9484683792 PDMS (94.8468), matching its reported
94.85. The per-scene result is stored at
`/mnt/project/DriveVLA-M0-runs/chainflow_submission_navtest/2026.08.27.11.27.12.csv`.

The released checkpoint name fixes the published checkpoint target, although
the paper does not state the run's final stopping epoch. NAVSIM `navtrain`
contains 85,109 train and 18,179 validation samples, or 103,288 after the
repository's `run_training_full.py` concatenation. Conditional on the private
run using that same set once per epoch and standard Lightning/DDP step
semantics, effective global batch 16 yields 6,456 steps per epoch. Training
through epoch index 26 (27 completed epochs) therefore produces exactly
174,312 steps, matching `best-epoch_26-step_174312.server_merged.ckpt`. The
checkpoint does not directly store global batch, per-device batch, or gradient
accumulation, so this is a strong reconstruction rather than a direct fact.
For comparison, effective batch 32 gives only 87,156 optimizer steps over 27
single-pass epochs; it matches 174,312 only if each reported epoch processes
roughly twice the reconstructed dataset. See `audit_stage2_batch_layout.py`
and `batch_layout_inference.json` for the explicit assumptions and alternative
factorizations.

The deleted historical checkpoint shards also preserve two facts that the
deployment YAML does not: every shard records PyTorch Lightning 2.2.1, and the
original run directory is named
`training_episode_Nav1_traj_long_25epochs_visionlora`. The `traj_long` label is
supported by a released but disabled training branch:
`long_trajectory_additional_poses=2` reads ten logged-future poses, cubic-spline
resamples them to eight poses ending at five seconds, and adds a second
min-over-64 L1 term beside the normal four-second GT target. It does not create
a counterfactual future; it is a second supervised horizon from the same logged
future.

The `25epochs` fragment is less conclusive than `traj_long`. Raw Lightning loop
state proves 174,312 updates arranged as 27 blocks of 6,456 and shows that the
scheduler was stepped on every update, but the stripped checkpoint contains
neither `max_epochs` nor the scheduler's configured horizon. If the released
SequentialLR was configured for 25 epochs and then allowed to run for 27, its
CosineAnnealingLR reaches zero at step 161,400 and rises again to a multiplier
of 0.01937 by step 174,312; a 27-epoch horizon instead decays to zero at the
last step. The 25-horizon trace has 7.31% less integrated LR and 3.77% less
square-root LR-squared budget. See `audit_stage2_schedule_horizon.py` and
`scheduler_horizon_counterfactual.json`. This is a real late-training control,
but the directory label alone is not sufficient evidence to start another
full run before the current epoch-9 milestone.

A read-only long-target cache was built for all 103,288 samples. In a strict
Lightning-2.2.1/Transformers-4.48.3/eager/seed-2/16x1 1,000-step A/B, enabling
only this extra target raised paired 128-log best-of-64 PDMS from 0.943566 to
0.977166 (difference +0.033601, log-bootstrap 95% CI
[+0.006762,+0.065910]). It also increased mean endpoint pairwise distance from
4.444 m to 6.009 m. This is currently the primary diagnosed cause of the old
proposal-bank collapse; selected PDMS is not expected to converge in only
1,000 steps.

The completed local run's validation deficit is localized to trajectory
proposal generation, not proposal ranking. On the same full navtrain
validation set, the public/local selected PDMS values are 0.951474/0.938030,
while their best-of-64 ceilings are 0.988530/0.974710. The scorer regrets are
therefore nearly identical (0.037056/0.036681), and the entire selected-score
gap is already present before scorer selection. The local checkpoint also has
lower selected-trajectory L2 despite its worse planning score. Reproduction
controls must consequently preserve proposal diversity and planning coverage;
training loss or L2 alone is not a sufficient success criterion.

The paper explicitly reports 16 H20 GPUs for Base Model training, AdamW, and a
`1e-4` learning rate, but it does not report per-device batch size, global batch
size, epoch count, precision, scheduler, warmup, or wall-clock time. The public
checkpoint name and released data path provide a strong reconstruction of the
missing run shape:

- released run (strong conditional inference): effective global batch 16
- released rank layout (weaker inference): with 16 reported H20 GPUs, 16 ranks
  x batch 1 x accumulation 1 is the most natural configuration, but alternatives
  whose effective product is 16 cannot be excluded without the private launcher
- local: 8 GPUs x batch 2
- global batch: 16
- optimizer: AdamW; the paper reports `1e-4`, but does not say whether this is
  the pre-scaling `base_lr` field or the actual optimizer-group LR
- epochs: 27
- proposals: 64

The released checkpoint also carries an exact initialization fingerprint.
Because `prev_weight=0` leaves trajectory heads 0--3 untrained, all 5,365,856
of their values can be compared with fresh `ActionDecoder` initializations. The
released checkpoint matches seed 2 bit-for-bit; the previous local full run
matches seed 0 bit-for-bit. `train_stage2_reproduction.sh` therefore defaults to
seed 2. This is a necessary configuration correction, but a 1,000-step A/B did
not show a seed-2 quality advantage, so it is not treated as the sole cause.

`initialize_from_config=true` is not a remaining mismatch.  Release `b9a4f27`,
the current YAML, and the active reproduction all select it; the true branch's
AST is unchanged.  Both release and current constructors instantiate
`ActionDecoder` before the VLM, so the large config-only VLM initialization
cannot perturb the action-head seed fingerprint.  Re-run
`audit_stage2_initialization_path.py` for the source/order/live-process audit.

The repository's generic `default_training.yaml` values (`batch_size=64`,
`max_epochs=20`, and `devices=1`) do not reproduce the released checkpoint's
epoch/step pair and are not treated as the original run recipe. The first local
full run used BF16 DDP, FlashAttention 2, a permanently eval-mode frozen VLM,
and dataset padding before distributed shuffling. Those choices reproduce the
step count but not the released configuration or the standard PyTorch sampling
semantics. In particular, the released config has `use_flash_attn: false`, while
the paper does not state precision, frozen-module mode, scheduler, or software
versions.

Use `train_stage2_reproduction.sh` for the controlled reproduction path. It
defaults to seed 2, eager attention, the released all-parameter AdamW decay
behavior, a reference sampler that shuffles the original 103,288 samples
before appending eight samples, the recovered long-2 target cache, and the
currently best-supported source warmup-cosine schedule. It fails closed unless
Lightning 2.2.1 and Transformers 4.48.3 are active. This still cannot be bitwise identical to an
inferred 16-rank x batch-1 job when run as 8 ranks x batch 2: the global batch
members match, but action-head dropout masks and floating-point reduction order
remain rank-layout dependent.

When both the local host and `training-vla-zt2` are available, use
`launch_stage2_multinode_reproduction.sh`. It locks both nodes to the same
packed Python/Torch/Lightning environment and launches 16 ranks x batch 1 with
global batch 16, testing the layout best supported by the conditional step-count
reconstruction and paper hardware count. It does not claim the private layout is
directly known. The launcher checks both runtimes and both GPU sets before mutating the
run directory, uses a shared rendezvous, records both node PIDs, and attaches
the normal completion/Navtest watcher to global rank zero. A short multi-node
smoke run must pass before a multi-day run is started.

The generic `train_stage2_full.sh` keeps its compatibility defaults for old
experiments. The two reproduction entrypoints do not: on this audited server
they now default to the long-2 cache, `long_trajectory_additional_poses=2`,
Lightning 2.2.1, Transformers 4.48.3, and source warmup-cosine. The historical
checkpoint directly proves the Lightning version and long-target branch; the
scheduler remains the highest-priority private-launcher hypothesis and is still
overridable. The corrected multi-node run can therefore be launched directly:

```bash
./local_stage2/launch_stage2_multinode_reproduction.sh
```

The equivalent fully explicit invocation is:

```bash
DRIVEVLA_NAVTRAIN_FEATURE_CACHE=/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_long2 \
STAGE2_LONG_TRAJECTORY_ADDITIONAL_POSES=2 \
STAGE2_LIGHTNING_OVERLAY=/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/lightning_2_2_1 \
STAGE2_REQUIRE_LIGHTNING_VERSION=2.2.1 \
STAGE2_TRANSFORMERS_OVERLAY=/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/transformers_4_48_3 \
STAGE2_REQUIRE_TRANSFORMERS_VERSION=4.48.3 \
STAGE2_SCHEDULER=source_cosine \
STAGE2_BASE_LR=1e-4 \
STAGE2_BASE_BATCH_SIZE=16 \
  ./local_stage2/launch_stage2_multinode_reproduction.sh
```

Both version requirements are checked on node 0 and node 1 before either
launcher allocates GPUs. The two runtime JSON records must be byte-for-byte
identical. This matters even when the Python executable path is shared because
host-local package paths can otherwise resolve different dependencies.

The ReCogDrive Stage-1 artifact records Transformers 4.37.2, while the raw
InternVL3 model directory records 4.48.3. These are materially different Qwen2
implementations, so both must be treated as runtime hypotheses. A fixed public
checkpoint subset can be evaluated under either isolated stack with:

```bash
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/nuplan-devkit:${TRANSFORMERS_OVERLAY}:${LIGHTNING_OVERLAY}:${EXTRA_SITE}" \
  ${PORTABLE_PYTHON} local_stage2/audit_stage2_public_runtime.py \
  --name RUNTIME_NAME --samples 128 --batch-size 2 \
  --output /mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/numerics/RUNTIME_NAME.pt
```

The tool round-robins across validation logs and saves every proposal,
selection, offline factor, and token rather than comparing only aggregate
PDMS. `watch_rl_zt3_and_launch_tf437_control.sh` waits for the explicitly
authorized rl-zt3 GPUs 3,5,6,7 and then runs a bounded 4-rank x batch-1 x
accumulation-4 control. It refuses GPUs used by non-stress jobs and locks
Transformers 4.37.2, tokenizers 0.15.1, PEFT 0.10.0, and Lightning 2.5.1 before
launching.

The public repository describes itself as a deployment package and does not
publish the private Stage-2 launcher. Its generic training YAML still points at
the final Base checkpoint, contains single-device defaults that contradict the
paper's 16-H20 run, and the released training entrypoint auto-selects the newest
checkpoint from sibling output directories. These files are valuable code and
checkpoint compatibility evidence, but together they are not a self-contained
from-scratch reproduction recipe.

The paper reports `1e-4`, while the released YAML contains `base_lr=5e-4` and
`base_batch_size=64`. The released optimizer scales those fields by
`sqrt(effective_global_batch/base_batch)`. This leaves four materially
different public interpretations: paper value as actual LR (`1e-4`), paper
value as a pre-scaling field (`5e-5` at global batch 16), untouched YAML fields
combined with the YAML's declared agent batch metadata (`8.84e-5`), or YAML
fields combined with the inferred global batch (`2.5e-4`). The private launcher
that would resolve this ambiguity was not released. The reproduction defaults
to the conventional reading that the paper reports the actual optimizer LR,
and every run records the optimizer-group LR rather than inferring it from a
field name. Constant `5e-5` is a required full-run control, not merely a small
learning-rate sensitivity test.

The released YAML explicitly uses `scheduler_args: null`, but that same file
also contains single-device and batch defaults that contradict the paper and
checkpoint step count, so it is not the private training recipe. The released
implementation contains a complete 10% linear-warmup/cosine branch. That
branch is line-for-line inherited from DrivoR, whose training YAML enables it;
ReCogDrive, the VLM source used by DriveVLA-M0, also uses warmup/cosine for its
imitation stage. The released action-head checkpoint has only about 0.53 times
the RMS displacement of the completed local constant-`1e-4` run (module-wise
ratios are 0.48--0.66). Both constant `5e-5` and peak-`1e-4` cosine have
approximately half the integrated LR of constant `1e-4`, so displacement alone
cannot distinguish them. Source lineage plus the paper's reported `1e-4` make
peak-`1e-4` source cosine the primary full-run hypothesis, with constant
`5e-5` retained as the competing control:

```bash
STAGE2_SCHEDULER=source_cosine \
STAGE2_BASE_LR=1e-4 \
STAGE2_BASE_BATCH_SIZE=16 \
  ./local_stage2/launch_stage2_multinode_reproduction.sh
```

`STAGE2_BASE_LR` and `STAGE2_BASE_BATCH_SIZE` are explicit because the optimizer
scales the configured base LR by the square root of effective/base batch size.
The launcher logs the actual optimizer-group LR; do not infer it from YAML
names alone.

For a running experiment, verify the values Lightning actually applied (not
only the Hydra configuration or checkpoint counters) with:

```bash
python local_stage2/audit_active_stage2_lr_trace.py \
  --event-dir /path/to/lightning_logs/version_0 \
  --output reports/stage2_reproduction_diagnosis/active_run_lr_trace.json
```

The audit compares every persisted TensorBoard LR point to the source
warmup/cosine formula and fails if the absolute error exceeds its tolerance.

The shared path `/mnt/project/DriveVLA-M0-env/bin/python` is a symlink into a
host-local conda environment. It can therefore resolve to different Python and
Lightning versions on different servers even though the path text is the
same. Every launch now prints a `STAGE2_RUNTIME` JSON line. Set both
`STAGE2_REQUIRE_LIGHTNING_VERSION` and
`STAGE2_REQUIRE_TRANSFORMERS_VERSION` to fail before GPU allocation when the
runtime does not match the intended experiment.

The log files come from `/mnt/project/DriveDreamer-Policy/navsim_raw`, while
sensor images default to the complete duplicate at
`/mnt/project/onevl_navsim_data/sensor_blobs`. The former sensor copy contains
at least one truncated log (36 rather than 168 frames per camera); shared files
were checksum-compared before selecting the complete copy. Override
`DRIVEVLA_SENSOR_ROOT` if the data is moved.

Run in order:

```bash
./local_stage2/cache_full_navtrain.sh
python local_stage2/build_stage2_long_target_cache.py \
  --source-cache /mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full \
  --output-cache /mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_long2 \
  --additional-poses 2
./local_stage2/smoke_stage2.sh
./local_stage2/launch_stage2_reproduction.sh
./local_stage2/launch_stage2_multinode_reproduction.sh
./local_stage2/evaluate_checkpoint.sh CHECKPOINT stage2_full_seed0_navtest
```

`launch_stage2_reproduction.sh` is the controlled production entrypoint and
delegates lifecycle management to `launch_stage2_full.sh`. It starts training
and the completion/evaluation watcher under independent `setsid` + `nohup`
sessions, records their PIDs in `launcher_state.env`, and refuses to reuse a
nonempty run directory or GPUs owned by unrelated processes. Directly invoking
`train_stage2_full.sh` remains useful for foreground debugging, but a foreground
tool or terminal session must not own a multi-day run.

If metric caching already completed and only feature caching was interrupted,
resume directly with `./local_stage2/cache_navtrain_features.sh`.
Set `DRIVEVLA_FORCE_FEATURE_RECACHE=true` to atomically rebuild all feature
files after changing feature serialization.

The Stage-1 boundary can be audited independently with:

```bash
python local_stage2/verify_stage1_checkpoint.py \
  /mnt/project/DriveVLA-M0-modelscope/best-epoch_26-step_174312.server_merged.ckpt \
  /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B/model.safetensors
```

Feature cache files are atomically published. After an interrupted legacy
cache run, gzip integrity can be audited with
`python local_stage2/verify_feature_cache_integrity.py CACHE_ROOT` before
resuming.
Path serialization can be checked independently with
`python local_stage2/verify_feature_cache_semantics.py CACHE_ROOT SENSOR_ROOT`.
Use `verify_stage2_smoke_checkpoint.py` to confirm that a one-step checkpoint
kept all 1,005 VLM tensors unchanged while updating the random action head.

Set `STAGE2_TRAIN_CKPT=/absolute/path/to/last.ckpt` to resume an interrupted training run. The scripts intentionally disable heuristic auto-resume so an unrelated checkpoint cannot be loaded silently.

The following execution optimizations have been checked independently and can
remain enabled on the controlled reproduction path:

- four persistent forkserver DataLoader workers per rank decode/resize the nine
  images, cast them to BF16, build the exact prompt, and tokenize ahead of time;
- pinned, prefetched batches use non-blocking host-to-device copies;
- each rank is bound to the CPU threads local to its GPU's NUMA node;
- detached PDM proposal scoring uses sixteen forkserver workers and eight exact
  proposal partitions per scene;
- training-only scalar logging avoids unnecessary all-reduces and only records
  at the configured logging interval, while validation reductions remain
  distributed and exact at epoch scope;
- validation scores the selected trajectory and 64 proposals in one call; and
- a custom best/latest callback writes exactly once per epoch: a new best is
  linked as `last.ckpt`, while a non-best epoch writes a real latest resume
  state and preserves the older best.

Worker preprocessing and pretokenization produce identical hidden-state inputs,
proposals, predicted scores, and loss on the audited real samples. Sequential,
full-scene process-pool, and partitioned PDM scoring produce exactly equal
arrays. Fused validation scoring is also proposal-independent in the local
training scorer because progress is normalized against the cached PDM progress,
not against the maximum of the submitted candidate set.

The following earlier changes are not execution-only optimizations and must be
treated as explicit ablations:

- eager attention to FlashAttention 2 changes BF16 hidden states and gradients;
- forcing the frozen VLM to stay in `eval()` disables InternViT stochastic depth
  and LoRA dropout that standard recursive `module.train()` would activate;
- padding the dataset before `DistributedSampler` shuffles a different index
  space and changes essentially every optimizer batch; and
- excluding norm and bias tensors from AdamW decay differs from the released
  optimizer construction, which decays every trainable action-head tensor.

`train_stage2_full.sh` retains the previous defaults for compatibility and makes
each of these choices overridable. It should not be described as an exact
official reproduction. The private Stage-2 launcher was not released, so the
remaining ambiguous choices require A/B results and multi-seed validation.

The measured diagnosis, including the Flash, seed, learning-rate, and
checkpoint-displacement evidence, is maintained in
`reports/stage2_reproduction_diagnosis/STAGE2_REPRODUCTION_DIAGNOSIS.md`.

The additional read-only audits used to prioritize the corrected full run are
available as standalone commands:

```bash
python local_stage2/audit_stage2_feature_cache_semantics.py --samples 128 --logs 64
python local_stage2/audit_stage2_long_target_integrity.py \
  --output reports/stage2_reproduction_diagnosis/long_target_cache_integrity.json
python local_stage2/audit_stage2_score_loss_signature.py
python local_stage2/audit_stage2_optimizer_signature.py
python local_stage2/audit_stage2_lr_schedule_signature.py \
  /path/to/public_vs_constant_epoch0_update_direction.json
python local_stage2/audit_stage2_vlm_runtime.py --help
```

The LR-signature audit is explicitly a squared-LR random-walk approximation;
it prioritizes full-run controls and does not claim to reconstruct the private
optimizer state from weights alone.

The repository's older `cache_hidden_state=true` path is intentionally not used
for this controlled reproduction. Its feature builder constructs a backbone
from the base VLM path and does not restore all 1,005 tensors of the post-VQA
Stage-1 checkpoint, including the 160 LoRA A/B pairs. Both the released online
path and the worker-tokenized path use a fixed 2,800-token left-padded sequence,
so re-batching is not itself a dynamic-padding mismatch. A future hidden-state
cache is acceptable only after its Stage-1 checkpoint loading and online/cache
tensor equivalence are explicitly verified.

On this machine, the original 8-GPU path took approximately 1.24 seconds per
optimizer step. The first optimized pipeline measured 0.788--0.814 seconds per
step over full-run 100-step windows. Increasing the exact scorer pool to 16
processes and retaining eight partitions per scene measured 0.700, 0.704,
0.724, and 0.751 seconds per step over the final end-to-end probe windows
(approximately 0.719 seconds per step on average). This is about 1.72x training
throughput. The throughput result remains valid, but the completed run combined
the safe execution changes with the semantic changes listed above and therefore
does not by itself establish official Stage-2 reproducibility.

The prior `stage2_full_seed0_pipeline_v7` run was attached to a foreground tool
session and was terminated when that session closed at epoch 0, step 3,649. It
did not fail from OOM, NCCL, GPU, or training-code errors and produced no
checkpoint. The detached restart is stored under
`/mnt/project/DriveVLA-M0-stage2/runs/training/stage2_full_seed0_pipeline_v8_restart`.
After the automatic full Navtest evaluation, `evaluate_checkpoint.sh` validates
all 12,146 rows, checks that the candidate and public Base token sets are
identical, hashes the evaluated checkpoint and CSV, and writes a machine-readable
`comparison.json` under the evaluation experiment directory.

The sequential, full-scene process-pool, and partitioned scorer outputs were
checked for exact array equality. The scorer and post-CUDA forkserver behavior
can be audited with:

```bash
python local_stage2/verify_score_process_pool.py
DRIVEVLA_SCORE_START_METHOD=forkserver \
  python local_stage2/benchmark_score_partitioning.py
```

Set `DRIVEVLA_SCORE_PROCESSES=0`, disable the worker preprocessing switches, or
override the launcher environment variables to restore the corresponding
serial paths. Ray scoring remains disabled for this single-node DDP launcher
because starting a separate local Ray cluster in every rank is unsafe.

## Public-Base residual scorer cache and fine ranking

The scorer-only follow-up keeps the released proposal bank fixed and exports
only inference-time tensors. Independent GPU workers can write resumable token
shards:

```bash
CUDA_VISIBLE_DEVICES=0 bash local_stage2/export_public_base_scorer_cache.sh \
  --split all --shard-count 3 --shard-index 0 \
  --batch-size 16 --chunk-size 128
```

Attach offline PDM training labels in a physically separate tree:

```bash
bash local_stage2/score_public_base_scorer_cache.sh \
  --num-workers 24 --watch
```

Train the zero-initialized top-16 residual ranker on official complete-log
train/validation partitions:

```bash
bash local_stage2/train_public_base_residual_scorer.sh \
  --mode local --top-k 16 --epochs 20 --require-complete-cache
```

The primary modes are `local` and `scene_cross_attention`; `set_aware` and
`scene_cross_attention_set` are candidate-set interaction controls.
`--score-mode residual` learns a bounded utility delta, `factor_aggregate`
calibrates the six interpretable factor logits and reuses the released
PDMS-style formula, and `hybrid` combines both deltas. Final artifact selection
sweeps conservative shrinkage, a switch penalty, and factor-based safety gates
on held-out complete logs. Partial-cache runs are architecture pilots only.

The exported model artifact contains only the small residual head plus the
immutable public-checkpoint path and hash. Its deployable agent class is
`local_stage2.public_base_residual_scorer.PublicBaseResidualScorerAgent`.
