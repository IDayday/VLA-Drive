# Small-disk execution update — 2026-10-17

The latest user forbids building several TB of new cached tensors. The chosen
path reuses the existing verified current-frame teacher bank, performs missing
future computations with the same FP32 teacher, and uses an explicitly bounded
RAM LRU. It does not prebuild a full future tensor bank or change student
precision, supervision, sampling, the data population or the learning schedule.
Existing weights, caches and held experiments are preserved.

## Verified cache boundary

The full-current metadata index occupies 381,931,520 B including its root;
the new 32-scene request and original64-scene PCA-fit indexes together occupy
1,470,464 B. These are indexes, not the storage size of the pre-existing teacher
bank. No PCA refit occurred.

The exact real32-scene cache/computation composition has passed full-tensor
bitwise comparison, masks, sampling and frozen/RNG checks. Its actual result is
`audits/indexed_provider_composition_run_v1/RESULT.json` under the private
campaign root, SHA256
`a4f4c86f0b5c09d18a58d02e8919dd73e659b48a53ed0d2a25c7d68e497b7a2f`.
It took 170.356905 s; RSS peak 3,158,171,648 B and CUDA allocated peak
1,416,995,328 B. New teacher tensor disk bytes: **0**. The actual receipt was
independently accepted by the unchanged production validator. Full-population
hit rate and sustained student-update throughput remain **NOT_MEASURED**.

`indexed_smoke_handoff_v1/` binds that real result, the exact32-scene manifest,
the original PCA384 basis and the existing trainer. The planned diagnostic is
only two updates, with the original20k scheduler and1000-update warmup,
world1/micro1/global2, official BF16 execution and FP32 master/Adam states.
Its preparation input is not yet a registered training config. This diagnostic
cannot provide formal initialization or scientific results.

### Subsequent factory completion and actual student launch

The preparation above subsequently completed: result
`afb0c77ccea1b6e368995ad68efa7140e30e1f3a8fa1ab97e716a04f27c704d4`,
7.737131 seconds, RSS644,300,800 B, child rchar2,994,053,170 B. It reuses the
original PCA basis and performs no refit, optimizer update or teacher tensor
write. The actual config is
`f6b8bd3ad59a763355130ef8bb0ef9228473f424516c62e630def80e87db948d`.
Independent closure is in `indexed_smoke_preparation_resource_v1/actual_closure/`.
Its later directory accounting is65,536 B allocated/root-inclusive versus
40,678 B logical file bytes; this is a later stat observation, not a replacement
for the wrapper's62,329 B completion snapshot.

Training does **not** write model checkpoints inside that32MiB factory root.
`indexed_training2_handoff_v1/` changes only the output directory and binds a
separate120GiB native-checkpoint allowance. Normal two-save logical size is
74,731,857,310 B; conservatively three slots total112,097,785,965 B. These are
model/Adam/RNG checkpoints, not visual feature caches. The fresh capacity check
retains the full other1983GiB registered-run commitments and another1TiB floor,
without subtracting already-written checkpoints. It is advisory accounting,
not a verified filesystem quota or permission to delete anything.

The first control invocation failed before creating a training child: inherited
`execute()` still lockedGPU1 while the new registration usedGPU0. The independent
review identified this exact defect. The original controller and preparation
remain immutable. `run_indexed_student_smoke2_lockfix_v1.py` maps only that exact
inherited lock path toGPU0, preserving every other lock, the trainer and config.
`CONTROLLER_LOCK_FIX.json` records the entrypoint and initial failure. There was
no training restart or lost optimizer progress.

The corrected one-shot invocation is now launched on the freshly verified idle
localGPU0, timeout PID448307. Actual registration and execution evidence reside
in `/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1`.
It requests exactly2 optimizer updates with the original20k schedule, then
native boundary save and exit. No automatic retry or resume arm is scheduled.
Actual student completion, sustained throughput and full-population supply rate
remain **PENDING** at this report's publication; launching is not a PASS.

## Bounded W diagnostic preparation

The controller lifecycle fixes passed50 owner and7 independent CPU tests;
see `spatial_probe_controller_review_v3/`. Actual runtime v2 is immutable
`581399eb309677c20fe7464791b4f18685e947a6` at
`/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017`.
The first real SP0 dev16 plan then completed once under independent admission:

- result SHA256 `dd765726eff1c937c31a1aca0f3cc47db8515f53790f84646ae1b94218857520`;
- signature SHA256 `3de51e64aec7f555f8d5c053695656346e1549796dad08c2c5bdafa88a26cff4`;
- wall43.687974 s, observed process-tree RSS693,870,592 B;
- native checkpoint hashing37,376,393,182 logical bytes;
- model load0, GPU0, no W tensor extraction, no retry or owned-process remnant.

The large number above is existing-file **read/hash traffic**, not new disk
allocation. Extraction and the separately capped frozen-W probe package remain
pending actual typed-ledger admission. No expected signature was passed off as
an executed plan.

The v7 cache ledger remains unchanged (7claims, SHA256
`7fcfc0c18fa69fc65a125ee36acc1204f9976e62926bede016269136f79cebbe`).
Its768GiB envelope includes existing caches; it is not permission to create an
additional768GiB cache. Checkpoint commitments and the separate1TiB free floor
remain protected. Administrative report reservations are separately named:
64MiB for the teacher composition reports,32MiB for SP0 controls,32MiB for the
one-shot provider-config preparation. Effective metadata reservation is
1264MiB /4096MiB. These report allowances are not falsely claimed to have been
admitted as cached tensor files or actual consumed bytes. Future migration must
preserve them and all existing claims.

## Independently running experiments

The formal training processes remain untouched. Full fixed-dev evaluation is
separate from cache qualification and never reads Navtest here. Current complete
snapshots are in `consequence_full_dev_snapshot_v1/` and
`spatial_SP0_full_dev_snapshot_v1/`:

- C0_512:90.989352 PDMS/100; C1_512:91.169425; C2_512:91.204624.
- C0_4096:91.138335, recorded separately from512-update comparisons.
- SP0_5000:90.799174, its original20k schedule's screening endpoint.

All have1696 scenes/16logs, zero invalid/unresolved. These are single-seed
results; C1/C2 early differences do not justify eliminating other arms. The
SP1 full-dev export is running. The already registered C3_512 full-dev export
was launched on verified idle local GPU2, PID445432, original10-step single
trajectory and FP32-master evaluation. Route SHA256:
`360ecd610b2539d2bea96d192b91238012e72646066d08798ebe92bb92ce536a`.
No held IQE/MOPD/G task was restarted, no training was stopped, no checkpoint or
cache was deleted, and nothing was pushed.
