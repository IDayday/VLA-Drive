# Exact prepared controls and remaining launch dependencies

These commands were not executed. They invoke existing APIs. Lead must first issue fresh
independent resource review, live writer/queue quiescence and checkpoint commitments.
The new train build has a four-hour ceiling; a plan-only short token cannot authorize it.

## Migration (same ledger)

The prepared QUIESCENCE_NOT_APPROVED.json is deliberately unusable. Supply a real
Lead-created QUIESCENT receipt with exact API `bindings`, all existing claim/legacy
writer roots, process evidence and a fresh time window. A separate CURRENT checkpoint
receipt must retain the full 2183 GiB commitments and 1 TiB floor for the entire build.

```bash
: "${QUIESCENCE_RECEIPT:?fresh Lead receipt required}"
: "${QUIESCENCE_SHA256:?}"
: "${CHECKPOINT_RECEIPT:?fresh full-budget CURRENT receipt required}"
: "${CHECKPOINT_SHA256:?}"
cd /mnt/project/VLA-Drive-st-metadata-build-dev-v1-20261017
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.metadata_storage migrate \
  --old-policy /mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/storage_preparation_dev_v1/PROPOSED_POLICY_V8_SP0_DEV_W.json \
  --old-policy-sha256 0bf0dc467a83a378e02ab4c52e00c7f2836011985edc5bf76d4fb2135957f191 \
  --new-policy /mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/prep_phase_v1/PROPOSED_POLICY.json \
  --new-policy-sha256 59371867175a65dd032cc27c88a050227d8ac4ba34a681c2e31b250c58a0978d \
  --expected-old-ledger-sha256 a4c1fb15f4799187b03035343b8582484d8feb02fd23c7b632b5bc968662789d \
  --pilot-inventory /mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/prep_phase_v1/NO_PILOT_ADOPTIONS.json \
  --pilot-inventory-sha256 1c3b4503a88d29dfdffe9e75415e7b8161e59f9d25c6891b2db913f3660c4d23 \
  --quiescence-receipt "$QUIESCENCE_RECEIPT" --quiescence-sha256 "$QUIESCENCE_SHA256" \
  --checkpoint-receipt "$CHECKPOINT_RECEIPT" --checkpoint-sha256 "$CHECKPOINT_SHA256"
```

## Exact guarded writer payloads

These are the child commands for Lead’s bounded supervisor, not permission to run an
unmonitored four-hour shell. Reuse existing process-tree ownership, CPU/RSS/read counters,
shared storage guard, bounded logs and the per-role resource limits from RESOURCE_PLAN.
Keep dev and train serialized under the existing ledger writer lock. No GPUs are used.

### dev

```bash
cd /mnt/project/VLA-Drive-st-metadata-build-dev-v1-20261017
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.metadata_storage run \
  --policy-path /mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/prep_phase_v1/PROPOSED_POLICY.json --policy-sha256 59371867175a65dd032cc27c88a050227d8ac4ba34a681c2e31b250c58a0978d \
  --registration-path /mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/prep_phase_v1/DEV_STORAGE_REGISTRATION.json --registration-sha256 4344bb546c21a649946c0b82514fe2159515abd74288c8e3d2c7869dc6d31a83 \
  --checkpoint-receipt "$CHECKPOINT_RECEIPT" --checkpoint-sha256 "$CHECKPOINT_SHA256"
```

### train

```bash
cd /mnt/project/VLA-Drive-st-metadata-build-train-v1-20261017
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.metadata_storage run \
  --policy-path /mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/prep_phase_v1/PROPOSED_POLICY.json --policy-sha256 59371867175a65dd032cc27c88a050227d8ac4ba34a681c2e31b250c58a0978d \
  --registration-path /mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/prep_phase_v1/TRAIN_STORAGE_REGISTRATION.json --registration-sha256 d97beeb39e23a56730b03f78610f13900fd4a9403a6205ce59756055bd19fa12 \
  --checkpoint-receipt "$CHECKPOINT_RECEIPT" --checkpoint-sha256 "$CHECKPOINT_SHA256"
```

## Index phase

Both indexes are WAITING_REQUEST_ID. After each matching real future COMPLETE exists,
prepare new input descriptors with phase=index, current actual policy/ledger pins,
the exact future COMPLETE and original fixed teacher contract receipt. The helper emits
exact request-only index registrations and an additive policy proposal; another live
migration admission is required. It never turns a PLAN identity into a request identity.

Original teacher contract receipt (unchanged):
`/mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_runtime_run_v1/RESOLVED_PLAN.json`,
SHA256 `a1de94f3e51b3e8ed050fa7f63b55d819cfa6453af7c1268b04d20c51f5e1d32`.

```bash
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/full_future_metadata_preparation_v1/prepare_controls.py \
  --inputs "$INDEX_INPUTS_PATH" --inputs-sha256 "$INDEX_INPUTS_SHA256" \
  --output /mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/index_phase_v1
```

All variables above are genuinely pending receipts; none are fabricated paths or approvals.
No teacher tensors, model weights or training are produced by these metadata APIs.


## Reviewed outer invocation (supersedes the raw child examples for launch)

Use the committed `run_build_once.py` through the exact independently pinned controller path. The two source runtimes and prepared registrations remain the paths above. Source review and the fresh role-specific resource admission must be complete. Variables name actual pending receipts, not fabricated approvals. Execute dev and train serially; there is no automatic retry. Do not rerun an existing migration or build.

```bash
: "${BUILD_ADMISSION_PATH:?independent exact role admission required}"
: "${BUILD_ADMISSION_SHA256:?}"
: "${REVIEWED_CONTROLLER_PATH:?absolute reviewed run_build_once.py required}"
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/ddp/bin/python "$REVIEWED_CONTROLLER_PATH" \
  --admission "$BUILD_ADMISSION_PATH" --admission-sha256 "$BUILD_ADMISSION_SHA256"
```

The admission pins: schema `st_full_future_build_resource_admission_v1`; role `dev` or `train`; matching `PASS_BOUNDED_<ROLE>_BUILD_ONLY`; hostname; independent reviewer; finite issued/latest-start/expiry timestamps; exact `limits(role)`; controller path/hash; immutable PREPARATION; proposed policy; role registration; actual MIGRATED receipt; fresh CURRENT checkpoint commitments; current dispatch policy; and the three immutable monitor helper pins. It must cover the full 1800/14400 second invocation and 40 second cleanup. A source-review report alone is not a resource admission.

Original prepared controls remain historical immutable evidence. The new separate administrative allocation is 4 MiB, making the final planned metadata budget 5620 MiB / 8 GiB. See `BOUNDED_BUILD_CONTROL.md` and its additive budget receipt; do not reuse the prior 128 MiB administrative scope.
