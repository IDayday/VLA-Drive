# FP32 frozen visual cache production qualification

Fixed task start: `3931fedbd30326c3a98f32fd0259e5d06a8588c5`.
Branch: `perf/iqe-fp32-vision-cache-production-3931fed`.
Workspace: `/mnt/project/DriveVLA-M0-iqe-fp32-cache-3931fed`. Shared cache/evidence: `/mnt/project/iqe-fp32-cache-production-runtime-3931fed`.
Parent's separate concurrent commit `2085627` contains the22k feature queue only; this work preserves it and does not change22k.

The measured production recipe keeps original FP32 frozen visual, FP32 payloads,
FP32 trainable parameters/gradients/Adam, original downstream BF16 autocast,
legacy repeat8 with normal dropout/DropPath, all four original objectives,
global batch32/microbatch4/world8, original100k LR schedule and sampler.
Retained runtime: ordered bounded prefetch, fused AdamW, DDP bucket views50MiB,
no language recomputation on qualified A80080GB. No new precision or architecture search.
Vocabulary head bypass and auxiliary `need_weights=False` were already present;
this task does not claim them as new speedups.

The cache-only paired comparison is T0 under this exact deterministic runtime:
200 updates and full final model/Adam/RNG/cursor are bitwise equal. Migration
from the still-running historical nondeterministic/default runtime is an explicit
T1 execution boundary; the200-step comparison does not prove historical cross-backend bitwise continuation.

| Decision | Evidence/result |
|---|---|
| Keep FP32 cache reader/builder | 6400 actual next observations, zero cache misses/image decode/visual calls on hits |
| Keep full training semantics | Four losses/valid denominators/sample order exactly matched all8 ranks |
| Resume | Actual new-process pause33001 ->33002 equals continuous reference |
| Checkpoint lifecycle |32 actual official scores,8 online/cache/clone/bundle checks PASS |
| Full cache | BLOCKED_SHARED_CAPACITY; no full build launched |
| Replace current Base | Not executed; no current stop authorization |
|22k incremental Base/ACTIVE | Unchanged |
| Scientific improvement | UNTESTED; no new navtest started |

Historical BF16/repeat1/master-weight experiments remain available as history,
but are excluded by `ProductionOptions` and the production commands. They are
not silently stacked with this cache result.
