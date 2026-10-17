# Independent T1 native 2 → 4 adapter correctness review

**PASS_BOUNDED_T1_ADAPTER_ONLY** for owner commit
`59c256ec7c387d4b822a45787074db55386e2116`, adapter SHA256
`fe7b418dadb683c143f61d51508df32dae0c9d92f2726a4aade1950f9d112f93`.

Reviewer: `/root/st_full_index_profile_correctness`. Implementation owner:
`/root/st_temporal_readiness_inventory`. The reviewer changed only this independent
report/test directory. No production code, frozen runtime, resource admission,
registration, lease, ledger, GPU job, model, image, target or native checkpoint
payload was modified or read by this review. No real invocation was launched.

This is a narrow source/control/CPU-fixture result. It is **not** an actual T1
native continuation result, a resource approval, formal initialization approval,
or continuous-four versus interrupted-four numerical-equivalence claim.

## Exact scope and inherited evidence

The original runtime remains the clean worktree
`/mnt/project/VLA-Drive-st-temporal-readiness-inventory-20261017`, at
`e85d11810bdf0240e7295d0994534f1df4ed21a4`.
The only planned invocations are phase2 (updates 1–2) and phase4 (native
continuation, updates 3–4): **four cumulative updates, 16 scene exposures**.
The native parent of phase4 must be the new actual phase2 result, not an older
legacy-backend T1 checkpoint. The 20,000-update schedule and 1,000-step warmup
remain unchanged. A new-process terminal step4 reload is **NOT_RUN and outside
this two-invocation budget**. There is no third trial, automatic retry, formal
run, or continuous-four comparison in this approval scope.

Preparation review commit `a8b9bf9487a5facd52053223b740b33e22f4cf63` previously
ran 80 CPU tests. Those results are inherited, **not counted as newly rerun**.
Its exact step2 configuration hash remains
`05c76e837d9259137a0f88da35fa1cf6aa3d9d1ef23d527207540b66b769d820`.
All nine reviewed trainer/extension/state/runtime source hashes still match
that preparation review; see `SOURCE_EVIDENCE.json`.

The existing real64 data/teacher/PCA assets remain the intended inputs. This
review does not reopen teacher-bank construction or change the user's no-TB
cache decision. It does not recheck real payload contents. Historical real
legacy T1 continuation and official world4 zero-update restore evidence remain
historical, with their original backend/topology limits; neither is relabeled as
new official T1 world2 training evidence.

## Actual restore call chain

The reviewed adapter's `observe_restore` (line 191) and `rank_main` (line 216)
retain the original trainer. They temporarily wrap only
`DeepSpeedEngine.load_checkpoint` and `train_world.restore_rng`. They do not
replace forward, loss, backward, optimizer step, scheduler, sampler or native
writer. Phase2 calls the original trainer without the external observer.

For phase4, the frozen trainer first verifies the actual phase2 registration,
unchanged scientific recipe, source/data/Base identity, complete native inventory
and checksums through `resolve_extension`. Its actual native load requires
strict model, optimizer and scheduler loading. The adapter rejects each partial
load option before calling DeepSpeed. The wrapper then calls the existing
`LiveAudit.after_load`, collectively on the two ranks. The exact observer source
is `/mnt/project/st-world-policy-20261016/audits/correctness/audit_live_official_native_load.py`,
SHA256 `5fce51c4c3e682ff4d3c33c958d7cacad49d05d372a7c46a9d39123a8dea5b5d`.

`after_load` compares actual live tensors against the actual parent native
payload: serialized model/buffers, frozen fragments, full ZeRO2 FP32 masters,
Adam state and options, engine counters, parameter-shape inventory and scheduler.
It checks receipt identity, native client-state return, tag and topology. Native
file checksum validation stays in the original extension resolver; the observer
additionally checks receipt/inventory/size/file stability and live tensor bytes.
It does not claim unpersisted buffers were serialized and compared.

The original trainer copies the exact parent consumed state, checks engine step,
then changes only the new continuation identity/lineage annotation. Fresh-run
master restoration is skipped for a continuation. The original native RNG
restore then executes. The adapter calls existing `after_restore` **after** that
restore and **before update3**; supplied and actual Python/NumPy/Torch/CUDA/task
generator states are compared. Only that first post-load restore is observed.
The writer's later capture/save/restore operations do not trigger another parent
restore audit. The original `epoch`/`offset`/`exposure` and deterministic sampler
drive the next batches. The trainer computes learning rate from restored
`completed`, original warmup and original 20k horizon.

No old observer `main` or `finish` runs. Those carry a different zero-update
contract. After updates 3–4, the adapter requires exactly one load audit and one
parent RNG audit and retains their immediate-restoration status; it does not
compare final step4 state against step2 or claim no updates occurred. Two-rank
actual restored-state receipts and the original trainer's two-update result are
required by the parent adapter. Their values still require the real qualification.

## Findings and owner fixes

| Finding | Evidence | Resolution |
|---|---|---|
| Old observer's zero-update exit contract cannot certify a 2→4 continuation | Actual observer `finish` checks zero new updates and unchanged step logs; adapter intentionally does not call it | Correctly excluded; no third reload or synthetic zero-update claim |
| Partial native restore must not proceed | Independent tests reject strict-model=false, optimizer=false, scheduler=false, module-only=true and custom loader before native load | PASS |
| Failed live master/RNG comparison must stop before update3 | Actual adapter rank path exercised with injected comparison failures; no backward/step occurs and wrappers are restored | PASS CPU integration; actual multi-rank failure not run |
| Cleanup deadline could renew after an earlier timeout, and combined shared checkpoint/free-space guard was missing | Separate independent resource review identified the initial `5da49c14` boundary defects | Owner `8b6a5777` added fixed cleanup ceiling and exact current 2303 GiB commitments plus 1 TiB free-floor checks |
| Early admission check did not account for setup time before monitor start | This reviewer identified a minimal-window expiry gap in `a04e0cee`: the late monitor could derive a later cleanup deadline | Owner `59c256ec` now maps actual admission expiry to monotonic time, rejects insufficient remaining 3780s before Popen, and clamps cleanup/spawn-abort to that bound; negative test PASS |

The cd558→59c delta changes `controls`, `monitor`, `parent_main` and adds
`shared_storage_guard`. Twelve pre-existing functions are AST-identical,
including `observe_restore`, `rank_main`, `validate_recipe` and `check_prior`.
No native-restore or mathematical fix was needed from this reviewer.
The actual storage allocation, capacity, GPU UUIDs and lease remain the separate
resource review/Lead responsibility. The monitor uses sampled observations,
not a kernel cgroup hard limit; this report does not claim otherwise.

## Current test evidence

| Snapshot | Tests | Scope | Wall | Peak observed RSS |
|---|---:|---|---:|---:|
| cd558 / `5da49c14…` | 32 passed, 0 failed, 0 skipped | Owner18 + independent14 | 2.764s | 445,337,600 bytes, pytest process group |
| 59c256 / `fe7b418d…` | **42 passed, 0 failed, 0 skipped** | Owner28 + independent14, including fixed cleanup and setup-expiry cases | **2.662s** | **432,230,400 bytes, observed owned descendants** |

The current pytest body took 1.95s. Both executions used CPUs 0–1, two library
threads and hidden CUDA. Current allocated test output/temp peak was 119,808
bytes. The 180s / 4 GiB / 32 MiB limits were not approached. One existing
`pynvml` deprecation warning came from importing PyTorch; no GPU call was made.
The known small-source/control inventory is recorded explicitly (160,420 bytes
for the final set; 255,938 for the earlier set). No large manifest or payload
scan occurred. These are file inventories, not a claim to measure all Python
library import I/O.

Full outputs and the exact commands are preserved in `PYTEST_FULL.log`,
`TEST_EXECUTION.json`, and the `HISTORICAL_CD55876_*` files. The archived initial
independent test source is kept so its original exact-SHA assertion remains
reproducible. The current tests add real adapter rank dispatch with injected CPU
objects, all partial-load flags, failure-before-update ordering, actual existing
RNG comparator on tiny synthetic values, and source/entrypoint constraints.
The tiny child monitor tests are explicitly fixtures; they do not substitute
for real DeepSpeed, native checkpoint, teacher or image integration.

Current command (already executed once in this review tree):

```bash
cd /mnt/project/VLA-Drive-st-t1-native4-adapter-correctness-final-20261017
/root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/official_T1_native4_adapter_correctness_v1/run_cpu_tests.py
```

The evidence wrapper refuses to overwrite its prior test directory/log. A new
test attempt requires a distinct evidence directory; this does not authorize a
new training invocation. `run_once.py --help` also passed, with output preserved
in `CLI_HELP.txt`.

## Remaining real boundary and handoff

Before actual dispatch, Lead must supply the original real registration/lease,
the exact current adapter, independent resource receipt, checkpoint commitments
pin and actual bounded admission. Only Lead may invoke the adapter, separately
for phase2 and then actual-parent-bound phase4. No actual admission path/hash is
invented here. The implemented CLI is:

```bash
/root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-t1-native4-execution-20261017/reports/st_world_policy/20261017/official_T1_native4_execution_v1/run_once.py \
  --admission "$ACTUAL_REVIEWED_ADMISSION" \
  --admission-sha256 "$ACTUAL_REVIEWED_ADMISSION_SHA256"
```

Pending evidence is actual T1 history/current-target forward/backward, strict
native save/load on both ranks, actual live model/master/Adam/RNG hashes, next
sample IDs at updates3–4, finite loss/gradients, complete original writer
receipts, resource costs and owned-process cleanup. `RESULT.json` in this review
is only a correctness receipt for the pinned adapter. No real-runtime PASS,
formal launch permission, new-process terminal reload, continuous4 equality,
teacher tensor bank, or full-data training readiness follows from it.
