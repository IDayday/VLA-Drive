# SP0 dev W artifacts: additive downstream prerequisite

Implementation source starts at Lead `6faa9f83`. The independent real artifact
review is commit `4afda7af` (subsequently integrated by Lead). This patch changes
only the helper-level prerequisite for `train_plan`; it does not change the
frozen `581399eb` model/runtime, original validator, computation, supervision,
cache, process cleanup or resource limits.

## Actual evidence and failure

The manual extraction attempt's real RESULT remains
`STOPPED_NO_AUTO_RETRY`, SHA256
`57406c7728bd2937d6723ebe3b22596fd04dd45d0fcabaa22c64311daf11166c`.
Its subprocess exited0 and wrote matching CLI_RESULT and cache COMPLETE for all
16 requested scenes. Its supervisor reason is
`OWNED_DESCENDANTS_AFTER_LEADER_EXIT`, with8 verified owned processes cleaned and
zero remaining. This is not recast as a successful supervisor launch.

The original runtime's full16-record W validator passed in the independent
review. That exact review JSON is pinned to SHA256
`d17e397964d8cccd465fe09c8aa0d65ca63eaadcca0d304f69c28e70b65014b8`.
The actual cache COMPLETE/CLI_RESULT SHA256 is
`82fe8df47a082f1bce402827a524c327391e315ccc7fa0402f0ae815751d2fb2`;
cache identity is
`bccc958c07f1ff9ce1618221abe8763337cab742243975f2f7ae47180819b673`.
No cache payload, native checkpoint or teacher tensor was reread by this task.

## Consumer semantics

The new `reconcile_artifact_exit.py` produces an immutable additive receipt:

- status `ARTIFACTS_ACCEPTED_SUPERVISOR_FAILED`;
- eligible consumer exactly `train_plan`;
- original failed RESULT, successful CLI, cache COMPLETE/signature,
  preparation, exact independent review and all16 record metadata pinned;
- original role-budget accounting,2 already consumed extraction attempts and
  `additional_extraction_attempts=0`;
- `launch_approved=false`, `automatic_retry=false`,
  `new_resource_authorization=false`, `scientific_result=false`.

Only the known original failed attempt and independent review checksums are
accepted. A newly re-signed alternative review cannot substitute for that
independent evidence. The supervisor uses the failed RESULT itself as the
`previous_stage_result`; it does not synthesize a COMPLETE replacement.
An explicit `previous_artifact_reconciliation` pin is also required.

Original `stage_inputs` behavior remains unchanged without this pin. Only
train_plan can use it, and only alongside the exact manual-attempt path.
All old manual-retry identity checks still run. The newly admitted stage records
the additive pin and the preserved upstream failure in `read_accounting`.
The train role's existing separate budget remains unchanged; the upstream dev
budget and already consumed attempts are carried in the receipt, never reset.

The existing reviewer-counted dev role consumed68,166,209,851 prior bytes plus
153,770,280,621 manual-attempt bytes. The review adds37,758,464 nominal payload
hash/mmap bytes. Its remaining52,903,658,008-byte figure remains explicitly an
**upper bound before small metadata reads**, not newly granted capacity.
This patch grants no resource allowance and mutates no live ledger. A fresh
train_plan admission must use the actual then-current reviewed metadata policy;
it cannot inherit the failed stage's policy or a pending migration by assumption.

## Cache integrity boundary

The consumer rechecks the independent review's small JSON hashes and full
record coverage, exact shapes/dtypes/finite flags and identity bindings. It
does not rerun the independent18.9MB W payload validation. The new receipt also
captures regular-file, single-link payload stat versions (inode/device/size/
mtime/ctime); later metadata consumption rejects missing or changed versions.
This is an additive metadata transition based on validation at the recorded
review time, not a fresh content-hash assertion. A same-size payload mutation
before reconciliation creation is not proved absent by stat alone. The later
fit path still uses the original `read_world` full SHA/tensor validator and
rejects corrupted content before consumption. No hash check was removed.

Only small existing evidence is accepted by this helper; it never opens W
payloads, images, raw logs, model checkpoints, teacher tensors or GPU devices.
The independent validator's earlier `/proc` zero-live snapshot is preserved;
historical PIDs are never used as signal authority here.

## Source and admission integration

Owned implementation files:

- `reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/reconcile_artifact_exit.py`
- narrow additions to the same directory's `supervise_once.py`.
- `tests/st_world/test_reconciled_w_artifact.py`.

The helper lives beside the existing standalone supervisor so the immutable
runtime need not be modified to import it. A reconciliation-using admission
must pin four helper hashes: the original `supervise_once.py`, `observed_cli.py`,
`prepare_bindings.py`, plus `reconcile_artifact_exit.py`. Original admissions
continue requiring their original three helpers. A source/resource reviewer
must approve the new supervisor/helper before any real train_plan dispatch.

After that review, the metadata-only receipt command is:

```bash
cd /mnt/project/VLA-Drive-st-w-artifact-reconcile-20261017
env PYTHONDONTWRITEBYTECODE=1 python \
  reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/reconcile_artifact_exit.py \
  --independent-review /mnt/project/VLA-Drive-st-manual-retry-correctness-20261017/reports/st_world_policy/20261017/correctness/sp0_dev_artifact_reconciliation_v1/ARTIFACT_RECONCILIATION.json \
  --review-sha256 d17e397964d8cccd465fe09c8aa0d65ca63eaadcca0d304f69c28e70b65014b8 \
  --output /mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/reconciliation/dev_extract_manual_retry_v1.json
```

The output path is a planned new additive artifact, not asserted to exist.
The command was **not executed**. Source/resource review and fresh stage
admission remain pending. No W artifact validation, extraction, retry, plan,
fit, GPU dispatch, cache write or live reconciliation receipt was performed.

## CPU regression evidence

Actual command:

```bash
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_reconciled_w_artifact.py \
  reports/st_world_policy/20261017/sp0_dev_extract_manual_retry_v1/test_manual_retry.py
```

**38 passed in0.31s**, recorded in `TEST_OUTPUT.txt`; actual CLI `--help` also
passed. All payloads in tests are explicitly marked CPU fixtures. Tests cover
the exact accepted transition, unchanged failed evidence and consumed attempts,
old failure rejection without a receipt, other-stage rejection, wrong hashes,
re-signed contradictory status/identity/budget, unverified/incomplete cache,
nonzero cleanup, wrong exit/reason, changed/missing payload stat, idempotence,
and a guard proving the consumer never opens a payload. They do not replace the
already completed independent real16-record verification.
