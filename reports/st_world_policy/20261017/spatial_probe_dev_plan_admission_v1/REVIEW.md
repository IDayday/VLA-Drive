# SP0 dev16 plan: independent resource admission

**PASS_BOUNDED_DEV_PLAN_ADMISSION.** Scope is one CPU `dev_plan` at immutable runtime `581399eb309677c20fe7464791b4f18685e947a6`. It does not authorize extraction, fitting, storage migration, teacher inference, temporal training, or a retry. Lead alone executes. Controller lifecycle correctness remains the separate reviewed `3b2f3eb2` record; this review did not repeat its 57 tests.

Actual preparation SHA is `34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431`. Exact native endpoint is SP0 `step_005000_1792217229597178910`, small COMPLETE SHA `1583c98960120ad2cde5859919507386b27259c4b3524a82b40de6ce0f1299fe`. Preparation evidence and dev/train expected signature pins match. Expectations do not prove payload integrity: the real plan must still hash native checkpoint and image bytes. No such hashes or model/image/tensor loads were performed by this reviewer.

The approved plan receives two CPUs (0,1), an empty CUDA mask, 600 seconds, and observed process-tree RSS <=64 GiB. The supervisor samples RSS/wall and owns only its newly created child session using pidfds; this is not a cgroup hard limit. Allow 40 seconds for abnormal cleanup, even though the entry lease check requires 30. Read allowance is 256 GiB for the **whole dev plan plus later dev extraction**, with plan consumption subtracted before extraction; it is not a new 256 GiB allowance for each invocation. Expected checkpoint inventory is 37,366,035,746 bytes; actual read cost is unknown until the plan receipt. No physical bandwidth cap is claimed.

The one-loader global flock serializes these five fixed stages. An admitted attempt is consumed even on failure; no deletion or automatic retry is permitted. The real plan saves `stages/dev_plan/SIGNATURE.json` and must produce `PLANNED_ONLY` with matching expected identity. A failed plan does not grant extraction permission.

At review, no `stages/dev_plan` existed. Bounded stat of the SP0 control root found 126,976 charged bytes, including directory blocks. `MemAvailable` was 853,601,896 KiB. Shared free space at evidence capture was 7,550,856,208,384 bytes, above the existing 1 TiB floor plus the pinned 1,983 GiB remaining checkpoint commitment and the small control reservation. These are observations, not a new quota guarantee. The checkpoint receipt expires at 1792223245.5291371; any later extraction needs a receipt covering its full outer window.

## Accounting preserved

- Live v7 policy SHA `b7062c4da878519c435db2e88b899bd476e5cd6a0e9f16dca191fe61522bb831` and live ledger SHA `7fcfc0c18fa69fc65a125ee36acc1204f9976e62926bede016269136f79cebbe` are unchanged. All seven claim rows match the policy exactly.
- Same 768 GiB campaign cap, original state directory/lock, immutable metadata baseline and 4 GiB metadata maximum remain.
- Existing metadata commitments: 1,136 MiB. Existing administrative report reserve: 64 MiB, pinned by `QUALIFICATION_REPORT_BUDGET_v1.json`, SHA `1ab0e8a8673b354ec81d99311444c68a502ac8d4d0f86e66e103373f90c0744e`.
- This admission conservatively reserves **32 MiB total for SP0 controls/stages/logs/temporary output**, including already present controls. Its effective metadata total is **1,232 MiB / 4,096 MiB**. The per-stage writer envelope is 5 MiB logical bytes, with a 4 MiB plain log and bounded final/temporary publication. Directory/rounding and existing controls fit within the 32 MiB administrative envelope.
- These report reservations are explicitly **not cache-ledger claims**. Lead must preserve them in the next migration/admission's aggregate accounting; neither silently disappears nor grants a second cache allowance. Fit outputs and typed W tensors are separately gated and are excluded from these control reservations.

## Independent checks

`check_metadata.py`: 1,610,476 bytes of bounded JSON controls read; one successful exact `validate()` and five rejection controls (expired lease, GPU supplied to plan, wrong runtime, changed helper, one CPU). No torch import, native payload/image/tensor reads, GPU calls or live policy/ledger writes. Current exact helper hashes, metadata pins and live stat observations are in `REVIEW_EVIDENCE.json`.

```bash
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py \
  --admission /mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/spatial_probe_dev_plan_admission_v1/DEV_PLAN_ADMISSION.json \
  --admission-sha256 9da2ff09d96fc7666352950dc9196f37eb6baa2ba4a6b03e946e07ff1b67648d
```

Admission expires at **1792223137.8968108 (2026-10-17 07:45:37 UTC)**. Start with at least 640 seconds remaining (by approximately 07:34:57 UTC). The entry checks fresh runtime/source, shared mount/global flock, available memory and lease again. This command was **not executed by the reviewer**.
