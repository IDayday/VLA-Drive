# Actual bounded indexed-provider64 observation

This directory's collector analyzes the already launched run at
`/mnt/project/st-world-policy-20261016/profile/world_indexed_provider_profile32_updates64_v1`.
It cannot launch or retry training. It reads only small JSON/JSONL and native
checkpoint receipt/file sizes, with an aggregate128MiB ceiling. It never hashes
or loads checkpoint/cache tensor contents. Existing evidence is not overwritten.

The runtime is pinned to `2d5e9483a8c8bc80b71348211756f3cab6aa21b1`.
Run scope is oneGPU, microbatch1/global2,64 updates,128 exposures from the exact
same32 scenes. Repeated epochs are not a larger unique population. Original20k
schedule/warmup1000, BF16 execution/FP32 master state, FP32 fixed8 teacher,
all original current/future/ego supervision and PCA basis are preserved.

## Timing interpretation

The trainer's `steps.jsonl` internal timer ends before some reductions, logging
and status/provider I/O. Its mean is retained only as an internal diagnostic,
never used as the end-to-end throughput denominator.

The Lead's read-only observer captures existing `status.updated_unix` at0.5s
poll intervals. For observed consecutive completed steps n→n+1, their original
timestamps provide a complete interval between post-update status points. The
poller's `observed_unix` is not used. Missing steps are listed and no interval is
invented. A checkpoint tag change contaminating a cross-step interval excludes
that interval from normal-step statistics. Same-step pre/post-checkpoint updates
are listed separately; exact native-save time comes from `checkpoint_costs.jsonl`.

The collector reports all observed normal intervals and, separately, intervals
ending after the first8 updates. This is an explicit analysis convention for
this finite stability run, not completion of the different240-update registered
timing protocol. Cold startup is launch→status0 only when status0 was observed.
No missing initial timestamp is inferred from a later sample.

Checkpoint-inclusive observed mean is available only when all64 normal intervals
were observed uncontaminated; it adds the single actual final-save cost and
divides by64. It is a finite diagnostic amortization at save_every64, not an
estimate of another production saving frequency. Allocation wall, cold load and
the normal window remain separately reported.

## Supply, memory and state

Provider counters report actual immutable hits, RAM hits, computed images,
fixed8 calls/padding, failures, bounded LRU and tensor disk bytes. The collector
checks final provider identity and fixed8 accounting. The default cache is only
the bounded teacher target LRU; it is not the old multi-TB vision feature cache.

RSS samples are the actual training rank's `/proc/status` VmRSS, not an aggregate
of the controller, torchrun or all descendants. Initial/final8-update medians
and all observed peaks are reported without claiming long-run leak absence.
CUDA allocator peaks come from the unchanged trainer's per-step records. The
teacher/data phase precedes its per-step reset, so these figures are **not a
certified whole-invocation/teacher peak**; no additional GPU profiler is started.

Finite per-task losses, effective denominators, gradient norms and sampledFP32
master-update evidence are checked on every actual step. The native engine
receipt documents BF16 execution/gradient accumulation and FP32 master/moments.
Sampled update evidence is not a full-parameter bitwise comparison. Exact resume
and scientific quality are not claimed or rerun.

Native checkpoint inventory/SHA fields are inherited from the actual writer's
complete receipt; file sizes are verified through `stat`. This observer does not
reread37GB of weights to claim a second independent content-hash verification.
One final native save is required; old two-update snapshots remain preserved.
The previous two-update run is provenance/integration evidence, not a speed
baseline for this64-update run.

## Execution

```bash
env PYTHONDONTWRITEBYTECODE=1 python \
  reports/st_world_policy/20261017/indexed_profile64_actual_v1/test_timing.py
env PYTHONDONTWRITEBYTECODE=1 python \
  reports/st_world_policy/20261017/indexed_profile64_actual_v1/collect.py
```

The collector requires `EXIT_initial.json`; before then it exits without
publishing a final report. Five CPU timing fixtures pass (see `TEST_OUTPUT.txt`),
including skipped steps and checkpoint separation. They are not real training
measurements. Actual final values, when available, are in `RESULT.json`; actual
failure remains failure, with no automatic retry.
