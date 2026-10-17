# Fixed matched SP1/SP2/SP3 probe controls

These are executable, fixed-arm controls. Three metadata preparations and fifteen
resource-admission templates have been generated under `prepared/`. They are
**not APPROVED** and no cache, plan, extraction or fit was run. No ledger changed.
Original SP0 controls, failures, reconciliation and completed fit are untouched.

The runtime is the existing clean storage-only checkout:
`/mnt/project/VLA-Drive-st-w-storage-only-581399eb-v1-20261017`
commit `2a1e140a64d2817247ceb6214f7f2affa20d56c6`.
It differs from frozen `581399eb309677c20fe7464791b4f18685e947a6` in exactly
`metadata_storage.py` and `spatial_w_storage.py`, with both hashes pinned.
Every computation, native loader, observer, probe readout and fit source remains
identical. The explicit new source identity is recorded in each binding;
SP0's old identity is never changed. The original protocol path/hash and 64/16
populations, raw labels, 300 updates and seed remain exactly those of SP0.

## Actual preparation commands

Run in this clean control worktree after independent source review:

```bash
cd /mnt/project/VLA-Drive-st-matched-probe-controls-20261017
PYTHONDONTWRITEBYTECODE=1 taskset -c 2,3 /root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261017/matched_spatial_probe_controls_v1/control.py prepare --arm SP1
PYTHONDONTWRITEBYTECODE=1 taskset -c 2,3 /root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261017/matched_spatial_probe_controls_v1/control.py prepare --arm SP2
PYTHONDONTWRITEBYTECODE=1 taskset -c 2,3 /root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261017/matched_spatial_probe_controls_v1/control.py prepare --arm SP3
```

Already executed, idempotent immutable preparation. Existing paths and hashes:

| Arm | `prepared/<arm>/PREPARATION.json` SHA256 |
|---|---|
| SP1 | 2a59c1276bbb662e9f305cc64d115316d4fe207619f51bbb165ccc5c8eaaca69 |
| SP2 | c7e6cdeaf893b32e5e43ccb80dab4da365761c2b6faa42c45cce543615df0565 |
| SP3 | 3ad04538942f5c245f09b5a3f51000c56787af2f435cef43297d74ab1f5a20b1 |

Each contains actual train/dev bindings, expected signatures, a CPU fit
registration and five DRAFT resource templates. None fabricates a live GPU,
lease, typed storage claim, migration or checkpoint-space receipt.

## Stage dispatch after independent resource admission

Lead uses the same entry for each stage, with the actual reviewed admission file
and its hash. Only the fixed five stages inside that admission are accepted:
`dev_plan → dev_extract → train_plan → train_extract → fit`. Each output attempt
is created once; failure prevents automatic continuation or retry. The original
single-loader flock serializes native loads and remains held through cleanup.

```bash
CONTROL=/mnt/project/VLA-Drive-st-matched-probe-controls-20261017/reports/st_world_policy/20261017/matched_spatial_probe_controls_v1/control.py
/root/miniconda3/envs/ddp/bin/python "$CONTROL" run --arm SP1 --admission "$REVIEWED_ADMISSION_PATH" --admission-sha256 "$REVIEWED_ADMISSION_SHA256"
```

Use SP2/SP3 only with their own admissions. The two environment variables must
point to the **new actual independent review**, not a template or SP0 receipt.
The same invocation executes the fixed original extract or CPU fit when that
stage is admitted. No separate unbounded direct probe command is needed.
For bootstrap wall coverage, Lead may wrap with the already used GNU timeout
pattern using the independently reviewed wall+cleanup allowance; no resources
or approvals are invented by this script.

Resource templates preserve original stage limits: plan600s/RSS64GiB;
extract960s/RSS64GiB/GPUpeak24GiB; fit1260s/RSS2GiB, two CPUs. Train/dev each have
256GiB cumulative plan+extract logical reads; fit2GiB. Original 4MiB log and
5MiB per-directory control accounting remain active. Reviewer must reserve
control/fit output separately: W four-arm total546104992B under1GiB, new three
fits at most192MiB. Actual extraction requires a reviewed typed-W policy and
migration, current checkpoint commitments and exact single idle GPU UUID.
All original byte/hash/native/input/storage checks run unchanged.

## Exact narrow code changes

The wrapper loads pinned source bytes in a fresh process and binds only SP1/2/3.
It mechanically replaces the original prepare function's three literal arm
strings and one memory length, keeps the old protocol path, and supplies exact
registration/export/checkpoint/root constants from the pinned inventory.

The **only supervisor loop change** rediscoveries pidfd-owned active processes
after WNOWAIT confirms leader exit. A stale pre-exit active list is no longer a
failure; a real surviving descendant still produces
`OWNED_DESCENDANTS_AFTER_LEADER_EXIT` and is cleaned. Source mutation is exact
text/count checked, with a reverse-diff test. The original observer `run`,
`Counters`, GPU peak, hash admitted bytes, nominal load bytes, effective read
cap, mode, locks, result and cleanup logic are reused without modification.

Extra wrapper guards enforce the fixed arm, checkpoint, runtime, controller
hash, original helper hashes and independent reviewer. Old manual retry,
artifact reconciliation and storage-bridge exceptions are rejected for these
new groups. Legacy schema names `st_SP0_*` remain for original validator
compatibility; immutable preparation/checkpoint/root plus `matched_arm` bind the
actual group. They do not refer to reused SP0 outputs.

## Validation and limits

20 targeted tests passed, including a deterministic stale-active snapshot,
real surviving fork cleanup, normal exit, invalid arms, wrong controller/arm,
forbidden arbitrary observer command and all old retry/reconciliation fields.
`TEST.log` and `TEST_COST.json` record the actual run: 0.939s, VmHWM67.7MB,
rchar12.16MB, CPUs2/3. Test children were created and cleaned by the test only.
A first 19-test pass preceded the explicit stale-active regression.

Actual three-arm metadata generation took2.158s, VmHWM28.94MB, source/control
logical reads5,812,733B (rchar8,379,285B). A preceding preparation attempt
stopped at a source-replacement count assertion before writing outputs; the
source contains three arm literals, not two. That assertion was corrected;
no model/data or budget was changed. Total logical source/control reads and
fixture/control writes remain below64MiB; no payload/GPU/ledger operations.

This is implementation evidence, not an independent launch PASS. Source review
must focus on fixed bindings, the two already-reviewed storage files, stage
prerequisites and the exit race. Real native/image validation occurs in the
existing admitted plan/extract, not in these CPU fixtures.
