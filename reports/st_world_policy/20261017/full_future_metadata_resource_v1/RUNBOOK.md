# Actual PLAN-only commands

Only Lead dispatches these two fixed, serial, single-attempt operations. They
perform metadata reads and raw-log `stat`; they do not build future requests,
read image/raw-log payloads, load models, launch GPUs, or migrate the ledger.
The exact wrapper is committed and clean at `b0ae418cb51b3cc67259a80144d571aabd3a82b0`.
Its SHA256 is `feaadec9ccac6efc4a302fe7ca8bd2f55efa1f415ab77830befd571d7e5d3734`.
These commands explicitly supersede the preflight attempt rejected because of
an atime-only change. See `ATIME_REVIEW_ADDENDUM.md`; no earlier token is reused.

The fixed admission window starts at Unix 1792228270.1489353, permits startup
through 1792229250.1489353 (2026-10-17 09:27:30 UTC), and expires at
1792229470.1489353 (09:31:10 UTC). Each start must have 180 s running + 40 s
owned cleanup available. No automatic renewal or retry is authorized.

Run dev first:

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-full-future-plan-control-20261017/reports/st_world_policy/20261017/full_future_plan_control_v1/plan_once.py \
  --admission /mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/full_future_metadata_resource_v1/DEV_PLAN_ADMISSION_V3.json \
  --admission-sha256 4fc509288a07970521f80f23eca041b75df87d696b9e30c0243dbb2fe0bb7e02
```

Then train under its own exact source and budget:

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-full-future-plan-control-20261017/reports/st_world_policy/20261017/full_future_plan_control_v1/plan_once.py \
  --admission /mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/full_future_metadata_resource_v1/TRAIN_PLAN_ADMISSION_V2.json \
  --admission-sha256 f262084ae73b79090e4c3bdc7cc2b78b3e318b5a1d737bc4b310a6122091117a
```

The underlying exact CLI arrays and source pins are checked by the wrapper.
Do not invoke those arrays directly: the outer 180 s / 2 GiB RSS / 4 GiB read /
32 MiB output limits differ from the proposed later-build limits in PLAN.
Do not change the wrapper path after approval: its `__file__` is pinned.

Expected outputs, once actually executed, are:

- `/mnt/project/st-world-policy-20261016/caches/full_future_metadata_dev_v1/PLAN.json`
- `/mnt/project/st-world-policy-20261016/caches/full_future_metadata_dev_v1/control/RESULT.json`
- `/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_v1/PLAN.json`
- `/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_v1/control/RESULT.json`

`PLAN_COMPLETE` only establishes a bounded proposal. Failure leaves a spent
root; preserve it. The wrapper has no resume or automatic retry path. An
explicit later attempt needs its own reviewed output identity.

After actual plans, Lead can bind their real hashes in future-build
registrations and the versioned 8 GiB additive storage migration. The build
entry remains `python -m tools.st_world_policy.metadata_storage run`, wrapping
`prepare_future_metadata build --plan ... --registration ...
--registration-sha256 ...`. The index entry remains
`index_teacher_metadata build --requests ... --expected-request-identity ...`,
with the real completed request identity and pinned original teacher contract.
Those registrations, policy and complete request identities do not yet exist;
this report deliberately does not invent hashes or authorize their execution.
The proposed per-stage inclusive limits and exact future roots are in
`RESOURCE_PLAN.json`. Full build admission must use fresh checkpoint obligations
valid through its complete window, including cleanup.
