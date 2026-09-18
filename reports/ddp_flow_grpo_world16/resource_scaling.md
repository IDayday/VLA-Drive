# Resource reallocation, 2026-09-17

The user explicitly authorized stopping GPU pressure scripts, restarting both
experiments and using more than8 GPUs. The original four-rank paired runs were
stopped only after **both update100 checkpoints and all1696-scene dev evaluations
completed**. Nothing is reset or overwritten. Their directory remains
`runs/paired_full_navtrain_v3`. The old controller PID1268540 was terminated after
its two evaluation children finished.

## Resources actually inspected

| Server | GPUs | Assignment | CPU availability |
|---|---|---|---|
| local `training-vla-zt` |8 A80080GB| F ranks0–7 |128 logical CPUs|
| `training-rl-zt2` |8 A80080GB| F ranks8–15 |128 logical CPUs|
| `training-vla-zt2` |8 A80080GB| U ranks0–7 |128 logical CPUs|
| `training-vla-zt3` |8 A80080GB| U ranks8–15 |128 logical CPUs|
| `training-rl-zt4` |8 occupied A80080GB| not allocated; no visible process identity allowing intervention |128 logical CPUs|

Five other SSH aliases were unavailable; exact probe outcomes are in the initial
inventory. The16 pressure-script parents on the two `training-vla-zt2/3` hosts
were identified by their exact `/mnt/project/gpu_stress.py` argv and separate
process groups before SIGTERM. Their PID/argv records and post-stop GPU readings
are retained under `runs/resource_reallocation_v1`. Other workloads were not
terminated. The existing9GB DDP dependency environment was copied into previously
absent environments on those two hosts; no environment upgrade was performed.

Each training rank uses one reward worker and one OpenMP/BLAS/MKL thread. Each
evaluation worker is assigned one GPU and8 physical CPU cores on its node. The
trainer keeps its existing data-worker setting0, avoiding repeated spawn overhead
for a one-scene microbatch. All run directories, control processes, ports, caches
and logs are isolated. Supervisors terminate only their own job's process group.

## Contract and qualification

The proposed world16 recipe keeps global scene batch16, scene microbatch1,
accumulation1, G8/K10, inner epochs2, seed42, LR1e-6, reference coefficient.01,
action replay coefficient.1 and the original checkpoints/visual freezes. No
model, reward, input, candidate or numerical tolerance is changed. New runs start
from their own original SFT sources; the world4 optimizer is **not** described as
exactly resumed at world16. The old100-update results remain separate evidence.

The training implementation/existing-test executable identity is unchanged:
`854dbc2ecb67fb28a235ccc7238729f7560a7ccafead08d9f5307a626152153c`.
The new orchestration layer has its own source hashes in each cluster run
descriptor. Its semantic gate calls are the same production functions used by
the prior controller. World4 releases cannot authorize world16 training. The
new configuration files remain fail-closed until both complete world16 evidence
bundles pass the existing semantic publisher.

The initial16-rank NCCL check passed on the U host pair, including exact reduction
values. Containers have no `/dev/infiniband`; observed transport is Socket, not
RDMA. The268MB all-reduce averaged about.171 seconds,1.57GB/s effective bandwidth.
This is a communication probe, not model acceptance.

U completed two real joint updates and a separate two-update repeat. Repeat
update timings were28.92/28.71 seconds; the initial rollout/reference/reward batch
took17.08 seconds and is counted once across two inner epochs. This gives37.35
seconds/update for this bounded run, versus about86 seconds/update measured in
the previous four-rank production run. Startup, checkpoint I/O and cold reward
worker overhead are separate. A final production ETA requires live production
timings after the qualification sequence completes.

## Evaluation acceleration and preserved failures

`scripts/cluster_flow_grpo/parallel_evaluation.py` invokes the unchanged original
evaluator separately on whole logs and merges validated transactions. NAVSIM
adjacency never crosses logs and its final denominator is per scene, so whole
logs retain all neighbors and the original weight handling. Inference keeps the
same token/seed noise, one candidate, resolution and10 ODE steps. The merge
checks complete token coverage, disjoint actual log identities, cache identity,
trajectories, every required artifact and finite scores before atomic COMPLETE.

F's complete1696-scene baseline was actually executed on eight remote GPUs.
The final exact comparison passed for all normalized/physical trajectories,
every CSV component and the complete adjacency mapping. Original EPDMS and
merged EPDMS are both.9362947686109893. Reuse-only merge timings are **not** reported
as inference speedups; the largest whole log contains556 scenes and limits dev
parallelism.

Two integration failures remain preserved: the first parent omitted the child
environment's numerical-identity flags, and the next merge used lossy default
CSV parsing. The parent now sets the same explicit numerical flags; CSV merging
uses round-trip parsing. Correctly completed child evaluations were revalidated
and reused, without regenerating trajectories. The final exact comparison is
`runs/resource_reallocation_v1/f_parallel_comparison_v3_complete.json`.

F's first16-rank startup timed out at the120-second control collective after
model loading, before any optimizer update. All failure logs remain at
`f16_full_cont*`. Its explicitly separate retry prewarms the same original
model files before the unchanged native initialization. No timeout or numerical
tolerance was widened. Sequential checkpoint reads also avoid slow random mmap
faults during read-only optimizer verification; they do not skip content checks.

The new numeric-shard observer fixes a diagnostic limitation in the existing
world1/4 comparison scripts: lexical filenames order rank10 before rank2.
It validates contiguous numeric ranks and retains the existing Adam and1%
scaling criteria. The historical BF16 candidate chunk1/2 failure remains FAIL.

## Current entry points

All commands run from `/mnt/project/DriveDreamer-Policy-paired` with
`PYTHONPATH=$PWD/navsim:$PWD` and `/root/miniconda3/envs/ddp/bin/python`.

```bash
# CPU orchestration regressions; not production CUDA evidence
python -m pytest -q tests/cluster_flow_grpo

# Explicit bounded multi-node job; the JSON contains actual nodes/GPUs/entry
python scripts/cluster_flow_grpo/cluster.py run runs/resource_reallocation_v1/u16_full_cont_spec.json

# Read-only collation, then the existing semantic release publisher
python scripts/cluster_flow_grpo/verify_profile.py --variant f \
  --output reports/ddp_flow_grpo_world16/frozen_visual_evidence.json
WORLD_SIZE=16 FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python scripts/flow_grpo/publish_acceptance.py \
  --config configs/flow_grpo/paired_world16_frozen_visual.yaml \
  --evidence reports/ddp_flow_grpo_world16/frozen_visual_evidence.json

# Both releases are required; no GPU acceptance is issued by this launcher
python scripts/cluster_flow_grpo/paired.py \
  --plan configs/cluster_flow_grpo/paired_world16.json

# Same world/layout and immutable identity, latest complete checkpoint selection
python scripts/cluster_flow_grpo/paired.py \
  --plan configs/cluster_flow_grpo/paired_world16.json --resume
```

Use a new job/output identifier for another diagnostic; completed control or
checkpoint directories are intentionally never overwritten. The complete
multi-node qualification results and actual formal launch status are recorded
in the final validation report, not inferred from this resource allocation.
