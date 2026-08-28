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
repository's `run_training_full.py` concatenation. Distributed padding at
global batch 16 yields 6,456 steps per epoch. Training through epoch index 26
(27 completed epochs) therefore produces exactly 174,312 steps, matching
`best-epoch_26-step_174312.server_merged.ckpt`.

The paper explicitly reports 16 H20 GPUs for Base Model training, AdamW, and a
`1e-4` learning rate, but it does not report per-device batch size, global batch
size, epoch count, precision, or wall-clock time. The public checkpoint name and
released data path provide a strong reconstruction of the missing schedule:

- released run (inferred): global batch 16; with 16 H20 GPUs this is most likely
  batch 1 per GPU
- local: 8 GPUs x batch 2
- global batch: 16
- optimizer: AdamW, learning rate `1e-4`
- epochs: 27
- proposals: 64

The repository's generic `default_training.yaml` values (`batch_size=64`,
`max_epochs=20`, and `devices=1`) do not reproduce the released checkpoint's
epoch/step pair and are not treated as the original run recipe. Local training
uses BF16 DDP and FlashAttention 2 for the frozen VLM; the paper does not state
its precision or attention kernel.

The local launcher pads both train and validation datasets to a multiple of the
global batch. In particular, 103,288 trainval samples become 103,296, preserving
the paper run's 6,456 optimizer steps per epoch under the local 8 x 2 mapping.

The log files come from `/mnt/project/DriveDreamer-Policy/navsim_raw`, while
sensor images default to the complete duplicate at
`/mnt/project/onevl_navsim_data/sensor_blobs`. The former sensor copy contains
at least one truncated log (36 rather than 168 frames per camera); shared files
were checksum-compared before selecting the complete copy. Override
`DRIVEVLA_SENSOR_ROOT` if the data is moved.

Run in order:

```bash
./local_stage2/cache_full_navtrain.sh
./local_stage2/smoke_stage2.sh
./local_stage2/launch_stage2_full.sh
./local_stage2/evaluate_checkpoint.sh CHECKPOINT stage2_full_seed0_navtest
```

`launch_stage2_full.sh` is the production entrypoint. It starts both training
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

The full launcher keeps the samples, prompts, optimizer steps, losses, and
floating-point model path unchanged while moving input preparation off the
critical path:

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

The repository's older `cache_hidden_state=true` path is intentionally not used
for this exact reproduction. Besides constructing the feature-builder backbone
from a different checkpoint path, it caches samples independently while online
tokenization pads dynamically per batch. The action Q-Former receives the full
hidden-state sequence without a padding mask, so re-batching independently
cached states is not numerically equivalent to the released online path.

On this machine, the original 8-GPU path took approximately 1.24 seconds per
optimizer step. The first optimized pipeline measured 0.788--0.814 seconds per
step over full-run 100-step windows. Increasing the exact scorer pool to 16
processes and retaining eight partitions per scene measured 0.700, 0.704,
0.724, and 0.751 seconds per step over the final end-to-end probe windows
(approximately 0.719 seconds per step on average). This is about 1.72x training
throughput without changing the global batch or training schedule.

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
