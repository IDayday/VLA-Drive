# Real indexed student: one two-update smoke completed

**REAL_TWO_UPDATE_WIRING_PASS.** The Lead's single invocation exited zero at
2026-10-17 07:48:59 UTC. It consumed four real scenes from the exact pinned32
manifest and completed exactly two optimizer updates. This is a finite
image/teacher/PCA/student training and save-path result. It is not a scientific
result, sustained-throughput measurement, full-model bitwise comparison,
full-population qualification or formal launch approval.

Actual run root:
`/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1`.
Run ID: `indexed_provider_single_updates2_v1`.
Source: `31c3dca2fbc697b419bf343ce05d10be0a4fe13f`.
Hardware in the actual launch record: `training-vla-zt-worker-0`, physical GPU0.
No second training process, retry or continuous/split resume pair was started
by this observer. The existing training framework was unchanged.

The committed `RESULT.json` contains exact small-artifact SHA256 pins,
per-update records, native file inventories and observed output. `collect.py`
only read 285,910 bytes of JSON and used filesystem `stat` for native files.
It did not open an image, model, native state or RNG tensor, or acquire a GPU.
Observation began at 07:40:54 UTC and finished within the 15-minute ceiling.

## Identity and unchanged recipe

- Common S0 binding: SHA256
  `84544c2242c0c7ecdcbe4450073b23a3c556cad4d822860d5b8652ecee9f1e38`.
- Exact32 train manifest: SHA256
  `7c78c58c40456649a93d152f9925ea5e1faa7e08bb65e98cd6c4af3b0c52b9b8`.
- Actual config: SHA256
  `2ec8df5c996c51ef0b7b40707c0e66179643bcc5cac317ec4d53d31d0d85b13f`.
- Provider descriptor file: SHA256
  `bb53252f1aea91ed54fa7b3917314d2d7f6c1c251d1f89f6a079f0f582f3874d`.
- Descriptor semantic identity:
  `90a908a48eba2e3d94e1d07700178dd08c61cb4aa0943459bbf572bd1b17ff8b`.
- Actual provider identity, matching the qualified plan:
  `4e20806444638cf8ed18a767eabd75f69e8de406a96faa4925b23088223ba360`.

The last two identities are different objects. The earlier handoff table label
has been corrected from “provider identity” to “provider descriptor identity”;
no artifact hash or runtime identity changed.

This retains T2/PCA384, the original FP32 teacher and basis, current/future/ego
objectives, three views, W48/view, 96 sampled positions, world1/micro1/global2,
accumulation2, seed42, loader0, the original 20k timeline and warmup1000.
The first two learning rates were `1e-8` and `2e-8`; there was no two-step
cosine restart. Student execution and gradient accumulation are BF16 with
FP32 master weights and FP32 Adam states; the teacher alone is FP32 throughout.
This is not an all-FP32 student and not an IQE Query-head experiment.

## Actual losses, gradients and master updates

| Update | Current loss | Future loss | Ego Flow loss | Gradient norm | Changed sampled master elements | Maximum sampled update |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.6023471355 | 2.6160497665 | 0.00100183394 | 4.3509325981 | 2694 | 1.49011612e-8 |
| 2 | 2.7064008713 | 2.6973395348 | 0.00536051532 | 5.6075940132 | 2679 | 2.23517418e-8 |

All reported losses, gradient norms and sampled master changes are finite.
The unchanged runtime's actual post-update engine check verified Torch AdamW,
BF16 execution, 2,552,673,668 FP32 master elements and 5,105,347,336 moment
elements. Before the first update, moments were correctly absent. Both updates
passed `fp32_master_update_verified`.
This is sampled parameter-update evidence and the runtime engine check; the
observer did not independently scan every gradient or optimizer tensor.

Each update used denominators `current=221184`, `future_a=221184`,
`future_b=221184`, `ego_scenes=2`. Total counts are 442,368 for each target
element stream and four ego scenes. These are the actual valid sampled target
denominators, not the 32-scene population size or image-padding count.

Logged sample order:

1. Update1: `a63f004207525614`, `be625a27a9785d96`.
2. Update2: `7d5c00836fe55286`, `29741652e0705def`.

All four IDs are distinct members of the exact32 manifest. The final consumed
cursor is epoch0/offset2 and exposure4. No RNG tensor was reread and no new
exact-resume equality claim is made.

## Indexed teacher and bounded cache behavior

The actual four scene reads reused **48 immutable cache records** and computed
**12 missing image targets**. There were four fixed8 executions: 32 compute
slots, including 20 discarded padding slots. Padding is not extra training
exposure or extra target supervision. Runtime RAM hits were zero for this short
sample stream; partial-LRU behavior was covered separately by the real32
qualification and is not claimed to have been exercised here.

There were 52 LRU evictions, zero provider failures and **zero new teacher
tensor bytes on disk**. Peak retained tensor payload was 14,159,232 bytes.
The two metadata connections had a combined 33,554,432-byte SQLite cache
allowance, and `metadata_population_materialized=false`.
These bounds exclude teacher weights, activations, image buffers and batches;
they do not describe total process RAM/GPU memory.

## Complete native saves and finite stop

Two complete checkpoint receipts are present:

- `step_000001_1792223181796920591`: native save 76.909846783 seconds.
- `step_000002_1792223263666164659`: native save 72.888156496 seconds.

Each has model, optimizer/master/moments and rank0 RNG files totalling
**37,365,927,466 bytes**. The receipt semantic hash, run identity, world size,
native inventory and physical file sizes were checked. The writer's committed
file SHA256 values are recorded, but the observer did not rehash ~74.7 GB of
tensor payloads.

No unconditional step0 save or duplicate terminal step2 directory was created.
At `stop_after=2`, the trainer reused the just-committed native boundary and
reported `PAUSED`, `pause_reason=BUDGET_PAUSED`, `completed=2`,
`new_optimizer_updates=2`. The receipt's `termination_reason=RUNNING_CHECKPOINT`
is retained because the pause deliberately avoids rewriting identical state.
This is expected on a 20k schedule and is not an early-stop bug or a claim that
20k training finished. Both diagnostic checkpoints remain retained; their
weights are unusable as formal initialization.

The training run and checkpoints stay under the independently admitted
`world_indexed_provider_training2_v1` namespace. The metadata-only factory root
and its 32 MiB allowance are unchanged; no model weights were written there.

## Timing boundary and remaining work

The trainer reported allocation wall **529.271105148 seconds**, including cold
initialization and the native saves. Its internal two update timers were
5.614075851 and 4.929991633 seconds, with data/teacher portions 4.095943654 and
3.773258565 seconds. They exclude checkpoint work and are only two warmup
updates: **do not convert them into a sustained-throughput or cache-speedup
claim**. Per-update recorded allocator peaks reached 56,284,111,872 allocated
and 60,131,639,296 reserved bytes; they are not a complete lifetime device
memory census.

The new exact32 indexed provider now has real student wiring and native-save
evidence. Full-population metadata behavior, sustained execution efficiency and
formal resource/review admission remain separate requirements. This result
does not change a scientific configuration, approve a formal pilot, repeat
resume testing or authorize another run.
