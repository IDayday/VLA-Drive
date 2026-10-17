# Fixed T1 native 2 → 4 execution adapter

Status: **PREPARED, NOT DISPATCHED**. This is a Lead-run adapter for the already reviewed qualification, not approval or a new trainer. The e85 runtime remains clean and unchanged. No registration, environment receipt, lease, checkpoint, image/target read, GPU operation or training launch was made during this implementation.

## Scope

| Item | Fixed boundary |
|---|---|
| Original trainer | `tools.st_world_policy.train_world` in `/mnt/project/VLA-Drive-st-temporal-readiness-inventory-20261017`, commit `e85d11810bdf0240e7295d0994534f1df4ed21a4` |
| Mathematical recipe | Exact previous `official_T1_native4_preparation_v1/PROPOSED_CONFIG_STEP2.json`; T1 causal history + current target; no future index dependency |
| Execution | Official Torch AdamW / BF16 execution / FP32 master and Adam / ZeRO2 / FA2; original 20,000 update schedule and 1,000 warmup |
| Topology | Exactly 2 ranks, microbatch 1, global batch 4; CPUs 2–5 |
| Invocations | Only phase2 (fresh, updates 1–2), then phase4 (actual native parent, updates 3–4); cumulative 4 updates / 16 scene exposures |
| Resources | 96 GiB observed aggregate RSS, 512 GiB cumulative logical `rchar`, **new separate 120 GiB** checkpoint reservation, 3,600 seconds plus 180 seconds save/owned cleanup per invocation |
| Formal initialization | Always false; no profile/smoke weights promoted |
| Terminal reload | **NOT_RUN, outside this two-invocation qualification**; unchanged existing terminal tests remain historical evidence |

There is no continuous-four control, zero-step invocation, third reload or automatic retry. Failed attempts retain their one-shot marker and use a conservative remaining-read charge. Changing the recipe, source, topology, phase or parent is rejected.

## Reused implementation and exact changed boundary

`run_once.py` imports the existing e85 `supervise_once.py` OwnedSession, pidfd ownership, signal/subreaper handling, global-flock mount check, fresh-GPU check and locks. It imports existing `run_build_once.py` terminal wait accounting and bounded pipe helper. The small monitor loop is adapted for torchrun rather than the old single CPU metadata worker: CPU 2–5; live rank/descendant RSS and read counters; 180-second save/cleanup; original trainer text logs instead of a metadata JSON result envelope. Native trainer checkpoint receipts and RESULT supply completion evidence.

Pinned original metadata monitor is `8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644`. The **actual e85** owned-session helper is `d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902`; this differs from the older `a29527…` constant inside the metadata runner. The adapter explicitly pins the actual e85 helper, and does not call that runner's old admission/ledger/dispatch code.

The external native observer is pinned to `5fce51c4c3e682ff4d3c33c958d7cacad49d05d372a7c46a9d39123a8dea5b5d`. Phase4 wraps only actual `DeepSpeedEngine.load_checkpoint` and the trainer's first `restore_rng` after that load. It calls the **existing** `LiveAudit.after_load` and `after_restore`, collectively on both ranks. Full live model/buffers, FP32 master/moments, scheduler, engine metadata and Python/NumPy/Torch/CUDA/generator RNG are compared to the actual parent before update3. The observer's original `main`, `finish`, forbidden backward/step hooks, fixed old SHA/world4 and zero-update checker are deliberately not used. Subsequent checkpoint saves legitimately restore captured RNG and are not mistaken for another parent restore. No seed is reset after native restoration.

This is immediate restored-state evidence, **not** a claim of continuous-vs-interrupted FA2 bitwise update equivalence. The original `review_live_native_observer.py` hard-codes a zero-update/world4 exit contract and cannot certify this phase4 unchanged. A reviewer must examine the new per-rank `LIVE_RESTORE` receipts, native receipts, sample IDs and the two-update RESULT. No new full observer was copied.

Live resource limits are observed at one-second intervals, not cgroup hard ceilings. Summing live rank IO can conservatively over-count during a concurrent reap; final accounting reuses parent post-wait absorption exactly once. `/proc/rchar` is logical reads, not physical shared-storage traffic. Checkpoint monitoring counts partial files and has no deletion/cleanup policy for retained artifacts. Cleanup uses one absolute deadline, bounded by both first TERM + 180 seconds and invocation start + 3,780 seconds; `finally` cannot restart that grace. Failure to finish by that boundary is retained as a failure, never a cleanup PASS.

Resource-review correction to `cd55876b`: the adapter now pins Lead-issued `/mnt/project/st-world-policy-20261016/smoke/official_history_T1_native4_v1/controls/CHECKPOINT_COMMITMENTS_2303G_V1.json`, SHA256 `13f3f4b282db2aff24528bc5434db51a9351cf43cf02dcdea1fbef38d5434cb3`. It requires combined 2,303 GiB obligations (existing 2,183 + new 120), the 1 TiB safety floor, exact new output root, no reuse of old profile reservation, and expiry covering the entire admission. It checks actual shared `statvfs` before spawning and every monitor sample, including during cleanup. Available space must cover **both obligations and floor**. This is an advisory reservation with live enforcement, not a filesystem quota. No cache ledger/claims are modified. The control's historical `actual_free_bytes` snapshot is never used as live available space.

## Lead handoff (no dispatch performed)

1. Review this exact adapter and source hash independently. Resource review must confirm a **new 120 GiB** reservation at `/mnt/project/st-world-policy-20261016/smoke/official_history_T1_native4_v1/runs`, not borrow the old 80 GiB profile budget. Confirm two idle leased GPU UUIDs and CPUs 2–5; the adapter rechecks idleness and takes existing per-GPU flock names before launch.
2. Lead creates the actual phase2 config from the exact existing proposal, plus current environment/resource lease/registration at e85. No source/model/target changes are needed. Follow the existing native trainer registration schema and original independent review process. The adapter creates none of them.
3. Lead creates an immutable, checksummed `st_T1_native4_execution_admission_v1` control with `status=APPROVED_BOUNDED_T1_ONLY`, phase, exact `LIMITS`, current hostname/time window, runtime `{path, commit}`, `adapter` and `registration` pins, `reviews` pins, and `checkpoint_commitments` pinned to the actual control above. Review controls use `status=PASS_BOUNDED_T1_ADAPTER_ONLY`, `adapter_sha256`, and independent reviewer identity. Resource control also records `new_checkpoint_reservation_bytes=128849018880`, `checkpoint_root`, `uses_old_profile_reservation=false`, and a covering `valid_until_unix`. Admission has finite `issued_at_unix`, `latest_start_unix`, and `valid_until_unix`; lease must cover 3,780 seconds at start. These are **required actual controls, not supplied approvals**.
4. Invoke only the fixed adapter CLI below with the actual admission path/hash. It invokes original torchrun/trainer and preserves all original registration, data, target, receipt and native checks.
5. Only after phase2 succeeds, Lead may create phase4 config/registration: stop_after=4, new run ID, `continuation` pinned to the **actual** phase2 registration/tag/COMPLETE hash. Phase4 admission additionally pins `previous_phase` at `phase2/RESULT.json` and `parent_receipt`. It consumes the remaining shared 512 GiB budget. There are no speculative phase4 parent fields in this commit.

Real CLI, verified by `--help`:

```bash
R=/mnt/project/VLA-Drive-st-t1-native4-execution-20261017
/root/miniconda3/envs/ddp/bin/python \
  "$R/reports/st_world_policy/20261017/official_T1_native4_execution_v1/run_once.py" \
  --admission "$ACTUAL_REVIEWED_ADMISSION" \
  --admission-sha256 "$ACTUAL_REVIEWED_ADMISSION_SHA256"
```

The two environment variables must refer to actual Lead-created controls. No fabricated path or approved receipt is delivered. Run the same command separately for phase2 then phase4, never as an automatic chain. No `--resume`, arbitrary command, world-size, source, budget, retry or GPU-pressure action is exposed.

## CPU evidence and remaining real boundary

The focused suite runs with CUDA hidden, two CPU cores and no model, teacher payload, target tensor or image loading. It checks exact recipe/source/scope, two-phase limit, actual-parent pins, stale pins/symlinks, one-shot failure semantics, strict restore flags, preserved optimizer/backward calls, first-RNG-only observation, checkpoint partial-file accounting, owned tiny-child normal exit/TERM cleanup, no double-charge terminal IO, and the **existing** native comparator on small CPU tensors. `TEST_OUTPUT.txt` and `TEST_RESOURCE.json` preserve the current execution.

The independent resource review `14831196b440081e104c3ec2ab9a9ab4255d6a06` reproduced the prior deadline restart. Its no-child/no-model assertion is retained in `test_independent_deadline.py`, with only the newly required storage-guard fixture argument added. Additional fixtures reject absent capacity, stale/short reservations, the old 2,183-only amount, 80 GiB borrowing, wrong floor and mutated controls, and demonstrate a second fresh `statvfs` rejecting a disk that filled after the first check.

An initial attempt to use `/usr/bin/time` exited 127 because the binary is absent; retained as `TEST_LAUNCH_UNAVAILABLE.txt`. The successful test harness used Python process/RSS observation instead. This environment issue did not execute tests or training in that failed attempt.

Not yet exercised: real NCCL launch, actual e85 T1 forward/backward and native save, live FP32/Adam state comparison on a real phase2 parent, actual CPU/GPU/RSS/read/checkpoint costs, rank failure during a real native load/save, or phase4 sample/loss evidence. Those are precisely the pending Lead-dispatched two-invocation qualification. Preparation and fixture PASS do not grant real-runtime PASS or formal initialization.

The mathematical history implementation is unchanged from old T1 `9db31a…`; execution differs in the previously documented official optimizer/backend/master initialization contract. Do not assert old-backend continuation equivalence, and do not use a frozen-feature cache success as completion of T1–T4 or the spatial20k/probe path.
