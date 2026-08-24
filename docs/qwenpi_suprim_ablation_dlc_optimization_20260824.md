# QwenPI multi-trajectory + score ablation: DLC throughput tuning

Date: 2026-08-24

This note records the diagnosis and the matched throughput configuration for
the 100k-step DriveSuprim OFF/ON ablation. It is intentionally separate from
the final quality comparison: the two replacement jobs still need to produce
their checkpoints and evaluation results.

## Jobs inspected

| DLC job | Arm | Allocated accelerators | Training ranks actually launched |
|---|---|---:|---:|
| `dlc1g021ai5qk7j8` | DriveSuprim OFF | 16 | 8 |
| `dlcizecqy8gy96ei` | DriveSuprim ON | 16 | 8 |

Both jobs ran the paired launchers from
`/mnt/zhangt_workspace/project/VLA-Drive-DDP-DRS`. The launcher logs show
`devices=0,1,2,3,4,5,6,7 processes=8`, so half of each 16-accelerator DLC node
was not participating in training.

## Before-tuning measurements

Measurements below are steady 20-step timing windows from the shared launcher
logs. They are per optimizer step with global batch 64.

| Arm / phase | Model time | Data time | Throughput | Peak allocated memory on an active rank |
|---|---:|---:|---:|---:|
| OFF, dynamic K=64 from step 0 | about 11.3 s | about 0.000 s | about 5.53 samples/s | 57.40 GiB |
| ON, static-only curriculum (<10k) | about 3.4 s | about 0.000 s | about 18.84 samples/s | 61.03 GiB |

The OFF rate projects to roughly 313 wall-clock hours on one allocated
16-accelerator node (about 5,008 allocated accelerator-hours) for 100k steps
if it stays unchanged. The ON progress-bar ETA is not a whole-run estimate: its
first 10% is static-only. At step 10k it begins dynamic K=64 sampling and at
step 20k it reaches the full joint path, so most of the run will be much closer
to the expensive dynamic regime.

## Root cause

1. The DLC node reserved 16 accelerators but the local launcher started only
   eight ranks. This explains much of the low node-level utilization directly.
2. Dynamic proposal generation dominates the model time. Each sample creates
   64 Flow-DiT candidates. The old chunk size of 8 required eight serial
   chunks, and every chunk ran 10 Euler/DiT denoising forwards: 80 serial DiT
   forwards per rank and optimizer step before proposal scoring.
3. Flow training deliberately repeats each ground-truth sample eight times.
   This is part of the experiment definition and remains unchanged.
4. The four trainable Q-Former blocks used activation checkpointing despite
   substantial free memory. That adds backward recomputation. On the installed
   PPU PyTorch build it also emitted one checkpoint deprecation warning per
   rank and step, producing avoidable multi-rank log traffic.
5. Data loading, static-score mmap access, and NAVSIM dynamic-label supervision
   do not appear in `data_time`: measured loader wait is effectively zero.
   However, online NAVSIM proposal scoring is inside `model_time`. It was
   configured as one synchronous CPU scoring worker per rank, leaving most of
   the 128-core allocation unused while each rank scored its local samples.
6. The online scorer converted every one of the 64 candidates through separate
   Python `Trajectory`, `StateSE2`, and `EgoState` objects before entering the
   already-batched PDM simulator/scorer. It then waited for all CPU labels
   before starting the trainable Flow forward, serializing CPU and accelerator
   work that have no data dependency.

## Matched optimized configuration

| Parameter | Old | Replacement | Experiment semantics |
|---|---:|---:|---|
| DDP ranks | 8 | 16 | Same model and objective; uses the full DLC node |
| Per-rank micro-batch | 8 | 4 | Changed to preserve global batch |
| Gradient accumulation | 1 | 1 | Unchanged |
| Global effective batch | 64 | 64 | Unchanged |
| Dynamic candidates K | 64 | 64 | Unchanged |
| Candidate chunk size | 8 | 32 | Chunking-only execution change |
| Local candidate inference batch | 8 x 8 = 64 | 4 x 32 = 128 | Verified on the same PPU type |
| Serial candidate chunks | 8 | 2 | Cuts serial denoising loops by 4x per rank |
| Euler steps per candidate | 10 | 10 | Unchanged |
| Flow training repeats | 8 | 8 | Unchanged |
| Q-Former activation checkpointing | on | off | Same forward/backward function; trades free memory for speed |
| DataLoader workers across the node | 8 x 3 = 24 | 16 x 3 = 48 | More ranks consume more independent batches |
| NAVSIM metric workers across the node | 8 x 1 = 8 threads | 16 x 4 = 64 spawned processes | Four local samples are scored concurrently per rank |
| Nominal rank + loader + metric CPU slots | 40 | 128 | Matches the DLC CPU allocation |
| Candidate pose preprocessing | 64 Python object pipelines per scene | One vectorized NumPy SE(2) transform | Equivalent float64 state arrays; identical metrics |
| CPU/GPU scoring schedule | Synchronous, before Flow | Async submission; wait after Flow + scorer preselection | Same labels and one combined backward |
| Optimizer steps | 100k | 100k | Unchanged |

The 16-rank topology halves each rank's samples while keeping the optimizer
batch and learning-rate schedule identical. A same-hardware PPU probe at local
micro-batch 4 measured candidate sampling at about 1.016 / 0.605 / 0.529 /
0.499 seconds for chunk sizes 8 / 16 / 32 / 64. Peak allocated memory was only
1.57 / 1.61 / 1.66 / 1.79 GiB for the standalone 803M-parameter action model.
Chunk 32 therefore captures most of the available speedup with minimal
transient memory growth and two serial sampling chunks. Disabling Q-Former
checkpointing is safe under the lower training micro-batch and removes its
recompute and warning flood. Four NAVSIM scoring processes per rank match the
new local micro-batch of four, while three persistent DataLoader workers retain
the already-observed zero-wait input pipeline. OMP, MKL, OpenBLAS, NumExpr, and
BLIS remain at one thread per process so these explicit workers do not
recursively oversubscribe the 128 CPU cores.

The scoring implementation now transforms all `K=64` relative trajectories in
one float64 homogeneous-matrix operation and passes the intact 65-trajectory
pool (reference + candidates) to PDM. Keeping that pool intact matters because
PDM progress normalization is proposal-pool dependent; splitting candidates
into independently scored sub-pools would silently change the training labels.
On a real metric-cache scene, the old and vectorized full scorer produced all
ten output arrays bit-for-bit identically. Three warmed runs measured median
total scene scoring of 0.596 versus 0.413 seconds (1.44x), including the
unchanged simulator/scorer rather than only the converted section.

A production-like real-cache benchmark used new tokens on every iteration and
rotated backend order to balance filesystem caching. At one replacement rank
(`B=4, K=64`), one thread took median 2.487 seconds and four threads took 2.582
seconds, confirming that the Python-heavy path does not benefit from threads.
Four persistent processes took 0.744 seconds, a 3.34x speedup over one worker,
with every metric bitwise equal. The process pool always uses `spawn`, never
`fork`, so no initialized accelerator state is inherited. Across the DLC node,
16 rank processes + 48 DataLoader processes + 64 metric processes account for
the 128 allocated CPU slots. The reproducible probe is
`tools/benchmark_navsim_metric_parallelism.py`. A second probe created the
input proposals on `cuda:0` before spawning and returned all labels to that
device successfully, exercising the same accelerator-to-CPU-to-accelerator
boundary as training.

```bash
python tools/benchmark_navsim_metric_parallelism.py \
  --metric-cache-root "$NAVSIM_METRIC_CACHE_ROOT" \
  --workers 4 --batch-size 4 --candidates 64 --repeats 3
```

Scoring is also asynchronous. Once detached candidates have reached CPU, their
four NAVSIM tasks run while the accelerator executes the Flow training forward,
DrivoR preselection, and (for the ON arm) DriveSuprim coarse scoring. The label
future is resolved only at the metric loss dependency. Candidate count,
candidate-pool normalization, labels, loss weights, and the single combined
backward are unchanged. Reordering independent stochastic proposal generation
before Flow changes the exact random-number draw sequence relative to an old
run, but not either distribution or the matched OFF/ON experiment definition.

The two fully resolved YAMLs are automatically checked to differ only in
`run_id` and `framework.hierarchical_scorer.joint.enabled`; all performance
settings are identical between arms.

## Delivered code inventory

The complete execution-only optimization is revision
`d72517ac65e88c29bb79f0e0bcc4f9ce2be042f1` (`perf: parallelize QwenPI DLC
scoring`). Its production changes are:

- `run_qwenpi_multitraj_score_suprim_{off,on}_dlc.sh`: use all 16 accelerators,
  micro-batch four, three loader workers, and four metric processes per rank.
- `train_qwenpi_drivor_suprim_dlc.sh`: enforce the global-batch and CPU-slot
  contracts, expose the resolved performance settings, and provide
  non-interactive DLC preflight diagnostics.
- `qwenpi_multitraj_score_suprim_{off,on}.yaml`: use candidate chunks of 32,
  the process metric backend, and disable unnecessary scene activation
  checkpointing.
- `QwenPI_DrivoRSuprim.py` and `hierarchical_scorer.py`: overlap detached CPU
  labels with independent Flow/DrivoR/DriveSuprim GPU work and keep one
  combined loss/backward.
- `navsim_metric_supervisor.py`: batch K=64 pose conversion, score the intact
  proposal pool, and run four persistent spawned NAVSIM processes per rank.
- `tools/benchmark_navsim_metric_parallelism.py` and the corresponding tests:
  reproduce the process-backend speedup and verify metric equivalence.

No candidate, denoising, training-repeat, batch, optimizer-step, checkpoint,
or evaluation count was reduced by this revision.

## Post-deployment timing measurements

The replacement jobs both resolved revision `d72517a`, 16 ranks, micro-batch
four, effective batch 64, K=64, chunk 32, Euler 10, and the 64-process NAVSIM
pool. Measurements were captured on 2026-08-24 after warm-up from 20-step
windows in the shared launcher logs.

| DLC job | Arm and active phase | Model / wall time | Throughput | Peak allocated per rank |
|---|---|---:|---:|---:|
| `dlc162hp1mlpo6lj` | OFF, dynamic K=64 from step 0 | about 3.50 / 3.50 s | about 17.8 samples/s | 32.91 GiB |
| `dlcwwv3eyjglo10w` | ON, static-only before step 10k | about 2.13 / 2.13 s | about 30.6 samples/s | 34.27 GiB |

The last six observed windows were stable: OFF model-time averages were
3.468--3.567 seconds (median about 3.50), while ON static-only averages were
2.089--2.161 seconds (median about 2.13). Data wait was approximately zero in
both jobs, and step overhead was approximately 0.001 seconds, so the remaining
cost is inside model/proposal/metric execution rather than the DataLoader.

Compared with the original jobs on the same allocation:

| Arm and phase | Before | After | Wall-time speedup | Throughput before -> after |
|---|---:|---:|---:|---:|
| OFF, dynamic K=64 | about 11.3 s/step | about 3.50 s/step | about 3.23x | 5.53 -> 17.8 samples/s |
| ON, static-only | about 3.4 s/step | about 2.13 s/step | about 1.60x | 18.84 -> 30.6 samples/s |

At the observed steady rate, OFF projects to about 97.2 wall-clock hours for
100k steps on one 16-accelerator node, or about 1,555 allocated
accelerator-hours. The original OFF rate projected to about 313.9 hours, so
the replacement saves roughly 216.7 wall-clock hours for the same experiment.
The ON static prefix projects to about 5.9 hours for its first 10k steps; this
must not be extrapolated to 100k because its dynamic path starts at step 10k
and reaches the full joint path at step 20k.

## Why OFF is currently slower than ON

This is a curriculum-phase difference, not an inverted DriveSuprim kernel
cost. At step 920 the logs recorded:

- OFF: `dynamic_enabled=1`, DrivoR loss active, 3.58-second wall time for that
  step, and all 64 generated proposals receiving online NAVSIM labels.
- ON: `dynamic_enabled=0`, static DriveSuprim coarse/fine losses active,
  2.09-second wall time, and no dynamic proposals or NAVSIM labels.

The B3/OFF branch has no static scorer, so the trainer deliberately activates
DrivoR and K=64 sampling from step zero. The B4/ON branch uses its cached static
vocabulary for the first 10% of training, then ramps dynamic supervision from
10% to 20%. Disabling DriveSuprim removes only about 14 million trainable
parameters, while immediately enabling 64-proposal, ten-Euler-step generation
and physical scoring. That much larger operation explains the apparent
inversion. ON is expected to slow materially after step 10k and may be as slow
as or slower than OFF once the complete joint path is active.

The YAML ablation is matched at the explicit configuration level, but
`joint.enabled` selects different B3/B4 curricula in the trainer. Therefore
the present experiment is a recipe-level DrivoR-versus-DriveSuprim comparison,
not a compute-matched isolation of only the additional refinement heads. A
strict head-only ablation should activate the same dynamic schedule from step
zero in both arms and add DriveSuprim only in ON.

## Remaining cost after optimization

The optimized dynamic path still performs K=64 proposal generation with ten
Euler steps in two serial chunks per rank, followed by physical scoring of all
64 proposals for each of four local scenes and the trainable Flow/DrivoR
backward. The process pool reduced a real B=4, K=64 NAVSIM batch from 2.487 to
0.744 seconds and overlaps it with GPU work, but it cannot remove proposal
generation or the physical-label objective. Material reductions beyond the
observed 3.23x therefore require changing experiment semantics (for example
K, Euler steps, scoring frequency, or staged/offline labels) or adding more
accelerators; execution-only changes such as moving chunk 32 to 64 have only
small remaining headroom.

## Restart acceptance checks

Before treating a replacement job as valid, its first launcher block must
contain all of the following:

```text
processes=16
batch=micro:4 accumulation:1 effective:64
action_horizon=8 flow_train_repeats=8 num_dynamic_candidates=64 candidate_chunk_size=32 euler_steps=10 chunks_per_sample=2
scene_gradient_checkpointing=off
cpu_parallelism=ranks:16 dataloader_workers:48 metric_workers:64 nominal_slots:128
navsim_scoring=vectorized_pool async_overlap=flow+drivor+coarse backend:process workers_per_rank:4
```

After compilation/warm-up, inspect at least three 20-step timing windows. A
reasonable first acceptance target is below 7 s/step for the OFF dynamic path
and below 2.5 s/step for the ON static-only path. These are operational targets,
not guaranteed benchmark results; the replacement logs are the source of
truth. When ON reaches step 10k, re-check its dynamic-path timing rather than
extrapolating from the static-only prefix.
