# Runtime overlap evidence — 2026-09-18

CPU official reward scoring now can run during independent GPU reference
inference. `runtime.overlap_reward_reference=true` is currently restricted to
bounded diagnostics. It does not change training samples, rewards, current or
reference conditions, G16/K10, loss weighting, optimizer, dtype or trainables.
The ongoing64-update experiment retains its original source/profile.

Only `RewardService.score` enters a single background thread per rank. All model
work and collectives remain on the main thread. Scoring errors are exchanged on
the main thread after reference inference. Cleanup takes exclusive ownership of
this job's scoring pool, terminates its own workers on failure, then joins the
thread. CPU worker counts do not increase. Timing marks overlapping spans as
nested and records the actual remaining wait.

Actual validation (no tolerance changes):

| Evidence | Result |
|---|---|
| Concurrent execution/cleanup and real two-process faults |10 passed, exit0; asynchronous rank1 scoring error propagates after a main-thread collective, both ranks exit nonzero within the test bound |
| Full affected CPU regression |262 passed,4 skipped, exit0; CUDA skips are not GPU acceptance |
| Two native BF16/ZeRO2 updates,4GPUs, global scenes16, accumulation4, G16/K10 |Both serial and overlap exit0; own visual/full reference unchanged |
| Initial complete bank |Chains, old probabilities, official rewards, advantages, reference means/stds and metadata exactly equal |
| Actual optimizer gradients |672/672 tensors exactly equal at each of two updates |
| Optimizer and forward weights |All Adam/master/model tensor values exactly equal at both saved boundaries |
| Exact inner-boundary resume |Update1→2,14 state/RNG/pending/model/optimizer files, zero-tolerance PASS, exit0 |
| Same-GPU inference scheduling |Four fixed real scenes × six ordered trials; all rewards/reference tensors exactly equal, exit0 |

The same-GPU serial versus overlap median score+reference times are respectively
7.492/5.789,7.087/5.789,7.164/5.789,6.999/5.795 seconds. This phase is about17–23%
shorter. It is **not** a17–23% whole-training speedup. Native cold first-behavior
score+reference spans were36.31 versus26.25 seconds; backward/update costs remain.
Whole two-update jobs were590.60 versus586.59 seconds including startup, gradient
dumps and two checkpoints; that small sample does not establish steady-state
throughput. `summary.json` retains per-phase and whole-update timings.

The measurements used F-SFT, gamma0.6 raw credit, rho0.8, BF16 Qwen/stored weights,
inherited FP32 action computation and FP32 ZeRO2 accumulation/master/Adam, TF32off.
Native source identity is
`6ad9faf18cca803e33508d1d5c1f2c1474e011cd98dbf9da383954ec81eb5f80`;
the native change was committed in `8370617`. Actual source environment, own
checkpoint hash, observed dtypes and parameter manifests remain in the run
directories and archived execution contexts. This is4-GPU evidence, not an
assertion that8/16-GPU deployment has been newly tested.

No production acceptance record is issued. Historical chunk1/2 BF16 failure is
unchanged. A proposed FP32 action-weight snapshot is separately preserved under
`rejected_snapshot`: it saved only about2% but changed velocity, chain and
logprob on all four fixed scenes. It is FAIL and is never installed by training.
An initial snapshot alias-restoration unit test also failed and is retained;
the corrected diagnostic helper's four tests subsequently passed. The actual
snapshot numerical failure remains FAIL.

Two observer/launcher errors are also preserved: an invalid `--spec` launcher
argument exited2 before starting a model, and the first gradient comparison
used a wrong flattened dictionary key and exited1. The latter was corrected
and the entire actual comparison completed with zero tolerance. A slow read-only
comparison attempt was interrupted explicitly: mmap incurred tiny page-fault
reads on the shared network filesystem. Streaming one shard pair at a time
completed the same full comparison. No actor, tolerance, checkpoint or training
job was changed by that observer optimization.

Reproduction from `/mnt/project/DriveDreamer-Policy-perf` (these existing output
directories are complete; use new explicit output/control directories for a new
run, preserving all existing evidence):

```bash
export PYTHONPATH="$PWD/navsim:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/root/miniconda3/envs/ddp/bin/python
$PY -m scripts.cluster_flow_grpo.cluster run runs/reward_overlap/serial_spec.json
$PY -m scripts.cluster_flow_grpo.cluster run runs/reward_overlap/overlap_spec.json
$PY -m scripts.cluster_flow_grpo.cluster run runs/reward_overlap/resume_spec.json
$PY -m scripts.analysis.checkpoint_efficiency \
  --on runs/reward_overlap/serial --off runs/reward_overlap/overlap \
  --difference overlap_reward_reference --output /tmp/new_optimizer_comparison.json
$PY -m scripts.cluster_flow_grpo.boundary_evidence \
  --continuous runs/reward_overlap/overlap/checkpoints/update_000002 \
  --resumed runs/reward_overlap/resumed/checkpoints/update_000002 \
  --output /tmp/new_resume_comparison.json --cpu-threads 4 --streaming-load
```

The four-scene probe ran on rl-zt4 GPU4. Native serial used rl-zt4 GPUs0–3;
overlap/resume used rl-zt2 GPUs0–3. All have exited; no other job was stopped.
No PDMS/EPDMS improvement is inferred from these runtime checks.
