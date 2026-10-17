# Actual PLAN-only commands

Only Lead dispatches these two fixed, serial, single-attempt operations. They
perform metadata reads and raw-log `stat`; they do not build future requests,
read image/raw-log payloads, load models, launch GPUs, or migrate the ledger.
The exact wrapper is committed and clean at `e894f687ce65ae0ca2f4c6c16df1b1d9c1910faf`.
Its SHA256 is `bd3c1dadaed8c79ea5242bcf40e95d991b11851edfd6b61d47f1073f6f41159f`.

The fixed admission window starts at Unix 1792228071.9452727, permits startup
through 1792229051.9452727 (2026-10-17 09:24:11 UTC), and expires at
1792229271.9452727 (09:27:51 UTC). Each start must have 180 s running + 40 s
owned cleanup available. No automatic renewal or retry is authorized.

Run dev first:

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-full-future-plan-control-20261017/reports/st_world_policy/20261017/full_future_plan_control_v1/plan_once.py \
  --admission /mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/full_future_metadata_resource_v1/DEV_PLAN_ADMISSION_V2.json \
  --admission-sha256 c5a7c14b22ee07a3093d3f8cf65a2b5534e67f9994d3eb45eeaa71e623d78761
```

Then train under its own exact source and budget:

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-full-future-plan-control-20261017/reports/st_world_policy/20261017/full_future_plan_control_v1/plan_once.py \
  --admission /mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/full_future_metadata_resource_v1/TRAIN_PLAN_ADMISSION_V1.json \
  --admission-sha256 1860c18c9ee723e1d07b5f8a7c7f5a86c98e8d83214e27657aec2383be8443b0
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
