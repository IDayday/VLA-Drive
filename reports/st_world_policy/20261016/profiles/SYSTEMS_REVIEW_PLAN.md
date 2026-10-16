# Independent systems review: measurements pending

Reviewer: native agent `/root/st_systems`, a separate worktree and task from the
model and trainer authors. Read-only source starting point:
`b2df6b4bb775630156ab5af57151c639c321431d`. This document grants no formal launch
approval. It records the specific remaining measurements and the instrumentation
handoff. The source will be re-bound after the Lead integrates the hooks.

Both original v1.0 campaign contracts were read in full, along with project and
worktree `AGENTS.md` and the live dispatch policy. The user's later instruction
allows releasing positively identified owned pressure scripts; it does not
permit stopping real or unknown work. Only the Lead dispatches experiments.

## Current findings

1. `train_world.py` records its `slowest_rank_seconds` before profile all-gather,
   scene-ID all-gather, JSON logging, status fsync and checkpoint writes. The next
   timer starts after those operations. This is a useful inner update measurement,
   but undercounts complete training wall time. Throughput approval must use
   complete steady-window elapsed time, including the gaps between updates.
2. Every microbatch uses `collective_call` (including object all-gather), global
   counts are individually converted with `.item()`, and every optimizer update
   verifies sampled FP32 master changes. These protect real invariants. Trace the
   cost before proposing safe consolidation; do not remove them speculatively.
3. `STWorldTrainingDataset.__getitem__` verifies current JSON, ego bytes and three
   source-image hashes on each read. `TeacherCacheReader` has a bounded verified
   shard LRU. A 64-scene hot loop cannot establish full-population filesystem
   behavior. Report warm working set and first-access verification separately.
4. The runtime uses original BF16 execution, FP32 ZeRO2 masters/Adam state,
   bounded optimizer groups, repeat8 FM, and existing gradient checkpointing.
   No measurement may silently change these definitions or omit an auxiliary
   target. The four spatial controls intentionally have different W/target sizes;
   each final topology needs evidence for its real contract, including peak memory.
5. The current temporal implementation and FPO update pipeline still need their
   own complete measurements. Teacher-only and CPU-scorer-only rates cannot be
   substituted for their coupled model/CPU training throughput.

## Actual resource observations

Read-only inventories at **2026-10-16 21:17 UTC**, with a follow-up at 21:21 UTC,
are stored under:

`/mnt/project/st-world-policy-20261016/audits/systems/resource_20261016T2130Z/`

The directory label is an administrative name; the JSON's UTC is the actual
observation time. No process was signalled and no training was launched by this
reviewer.

All eight currently allowed hosts were reachable. Each has eight A800-SXM4-80GB
GPUs with NV8 GPU-to-GPU connectivity and 128 logical CPUs in the visible affinity
mask. They expose **cgroup v1**, `cpu.cfs_quota_us=-1`, `cpu.cfs_period_us=100000`,
and a memory limit near 2^63. These show no finite limit at the visible container
root; hidden parent limits cannot be inferred. They are not evidence that every
host CPU or all host RAM is available exclusively to this study.

The same Lustre project mount was visible on all hosts. `statvfs` reported
approximately 11.93 TB available at this observation. `lfs` is not installed,
so user/project quota is **NOT_VERIFIED**. Filesystem free space and quota remain
distinct. No full-cache allocation is approved by this observation.

| Host | Observed workload | Current action |
|---|---|---|
| vla-zt | Campaign 2-GPU smoke; owned pressure workers on some remaining GPUs | Preserve smoke; Lead may reverify/release exact pressure workers for profile |
| rl-zt2 | At follow-up, 8-shard FM-transfer RL5000 train diagnostic evaluation | Preserve real evaluation; do not use stale idle snapshot |
| tsinghua-zt | At follow-up, 8-shard FM-transfer RL5000 dev evaluation | Preserve real evaluation |
| tsinghua-zt2 | Existing natural-A training | Preserve |
| tsinghua-zt4 | Existing FM-transfer RL training | Preserve |
| tsinghua-zt6 | Existing FM-transfer SFT training | Preserve |
| tsinghua-zt3 / zt5 | About 70.7 GB/card, 100%, no identified owner | Unknown; preserve and do not signal |

The owned pressure source seen on vla-zt, rl-zt2 and tsinghua-zt was
`/tmp/zt-idle-pressure-v1/agent.py`, SHA256
`6f2dac15bd5649486528b22a69e8a297f71e0cad66cb2a9908baa4f86cc46bb2`.
Process identity, start ticks and exact worker UUID must be checked again before
any Lead action. On rl-zt2, NVIDIA reports host PIDs that differ from visible
container PIDs. A missing `/proc/<nvidia-pid>` is **not** an idle GPU indication.

## Bounded profile registration

Use one final integrated source, one fixed legitimate scene manifest and teacher
identity, common S0 initialization, no future label inputs, original normalization,
global batch32, microbatch4, repeat8 and original training precision. The candidate
clock is the registered 20k schedule; `stop_after=240` limits only new profile
updates. Do not compress cosine to 240 steps. The profiles' weights are discarded
for all scientific initialization.

On one freshly admitted 8-GPU host, run:

1. **Topology A:** one8-rank job, global32, micro4, one microbatch/rank.
2. **Topology B:** two real concurrent4-rank jobs on disjoint GPU sets, each
   global32, micro4, two accumulation microbatches/rank. They use the same fixed
   scene order/seed and compute the same objectives as A. Each has separate
   rendezvous port, run ID, output, full GPU-lock set and registration/lease.

Each job performs exactly240 updates:20 warmup+100 steady, then20 warmup+100
steady. The2x4 result is the sum of actual completed scene exposures divided by
their measured joint makespan; multiplying a solo4-GPU run by two is rejected.
Record individual job speed as well as aggregate node throughput and report any
phase skew. Exclude profiler runs from throughput tables.

Run a separate at-most4-update trace diagnostic with one or two representative
steps instrumented. Record CUDA/CPU operators, data/H2D, visual/language, sparse
readout, DiT/FM, backward/recomputation, communication, master verification,
optimizer, logging and checkpoint. CUDA event durations are inclusive and can
overlap across streams; do not sum them as total wall time. The full timeline and
slowest-rank window remain authoritative. Instrumenting the model in evaluation
mode or only the action head cannot substitute for training.

CPU workers initially remain2/rank; all BLAS thread pools are1, and the runtime's
PyTorch intra-op2 is separately recorded. For the2x4 topology this is16 loader
threads plus actual rank/BLAS activity, not “2 workers total.” Record own process
CPU and waited-child CPU. A live CPU scoring pool needs process-tree sampling or
a dedicated cgroup; do not call `wall*workers` measured CPU usage.

One final native checkpoint per job is required for actual save cost. Report it
separately and amortize at the unchanged proposed formal save interval. Do not
save every profile update or delete user checkpoint history. Inventory/statfs
must cover atomic-write headroom for all three profile runs before dispatch.

## Minimum trainer integration

`PipelineProfiler` in `tools/st_world_policy/profile_pipeline.py` does not launch
or change training. The Lead should wire a strict registered profile field; do
not let an unregistered environment switch change execution identity.

```python
profiler = PipelineProfiler(profile_output, identity=profile_identity,
    rank=rank, world_size=world, gpu_uuids=lease_uuids, mode="timing")
# profile_identity includes actual source_sha, scientific_config_sha256,
# data_manifest_sha256, teacher_identity, precision_policy, global_batch,
# and sample_order_sha256, plus the full runtime/registration/lease fingerprints.

# Immediately before the optimizer batch fetch:
profiler.begin_update(state["completed"] - initial_step + 1)
# Existing fetch, forward/backward, optimizer, all reductions, IDs, log/status.
# Existing checkpoint if due; measure its full elapsed time independently.
profiler.end_update(local_scenes=len(ids), checkpoint_seconds=save_seconds)
# Also execute end_update on the final iteration BEFORE its break.
# After the fixed profile budget and safe final save:
profiler.finish()
```

`begin_update`/`end_update` drain CUDA only at update boundaries. Call end after
logger/status and periodic save; using its end before those repeats the old
undercount. Setup/strict load/initialization and final save get separate total
allocation wall time. Normal update timing excludes steps containing checkpoint
events; complete window throughput includes them. `dtype_inventory` should run
after the first optimizer step for actual master/moment dtypes, not merely before
lazy optimizer state is created.

In trace mode, explicitly map model modules to `attach_modules`, e.g. actual
`model.base.qwen_vl_interface.model.model.visual`, its `language_model`,
`model.base.action_model` and `model.sparse_readout`. Hooks return no replacements
and preserve RNG/gradients. Add `region(...)` contexts around CPU/H2D/loss/reduction
paths where modules alone cannot expose the boundary. The hook call list records
recomputation as actual additional forwards, with input dtype/shape.

Actual tools, already present:

```bash
python -m tools.st_world_policy.profile_resources --help
python -m tools.st_world_policy.profile_pipeline --help
python -m tools.st_world_policy.profile_pipeline --output <new-summary.json> \
  summarize --ranks <all-rank-json-paths>
python -m tools.st_world_policy.profile_pipeline --output <new-aggregate.json> \
  aggregate-two-by-four --first <job1-summary.json> --second <job2-summary.json>
```

Angle-bracket values above describe future measurement outputs, not invented
existing paths. Runnable registered training commands can be finalized only after
the Lead binds the actual profile config, teacher population, lease and integrated
SHA. Existing `train_world --registration ...` is the single training entry.

## Instrumentation evidence and limits

Installed APIs were inspected directly in the original training environment:
Python3.10, torch2.5.1+cu124. `torch.profiler.profile`, `torch.cuda.Event`,
forward hooks with kwargs and `always_call=True` exist in this exact installation.
Official2.5 documentation URLs were attempted but unavailable through the browser;
the installed library is the direct implementation source. No library was upgraded.

CPU tests preserve output, gradients and RNG under nested trace hooks; reject
missing ranks/steps, mismatched identities/batches, trace timings, serial “2x4”,
overlapping GPU assignments, and dishonest multiplication of solo throughput.
They verify the instrumentation only. No actual8-GPU efficiency result is asserted
until the registered full pipelines have run and their records are reviewed.
