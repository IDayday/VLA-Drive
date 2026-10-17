# Bounded metadata admission: separate checkpoint reserve

Independent read-only snapshot: **2026-10-17 05:44:54 UTC**.
Source review worktree starts at `1162b955b3bf9abb6728823a84a50c38e1a19bb8`.

For this one short metadata package, retain **1,983 GiB
(2,129,230,036,992 bytes)** of separate remaining checkpoint/log commitments.
This deliberately keeps the full original commitments rather than deducting
already-written files: four spatial runs ×240 GiB, C0 ×187 GiB, four policy runs
×209 GiB. It is a conservative admission allowance, not observed usage or an
additional allocation request. It does not modify those runs' budgets.

The original commitments are documented in `FLOW_FORMAL_FIRST_PAIR_RUNBOOK.md`
and `FLOW_FORMAL_STARTED_v1.md`; each spatial registration also contains
`DATA_AND_STORAGE_BOUNDARY.json` (40 GiB per native payload upper bound, five
periodic writes and one extra allowance). The consequence launcher retains every
remaining 512-step/final boundary plus pause/partial allowance. No deletion or
retention change is proposed. C0 has completed, but its original allowance is
intentionally not released for this brief admission.

## Read-only observed inventory

| Run | Shared status step | Last complete receipt step | Native directories | Retained GiB |
| --- | ---: | ---: | ---: | ---: |
| SP0 | 4601 | 4000 | 4 | 139.200 |
| SP1 | 4163 | 4000 | 4 | 139.200 |
| SP2 | 3526 | 3000 | 3 | 104.400 |
| SP3 | 3181 | 3000 | 3 | 104.414 |
| C0 | 4096 (COMPLETE result) | 4096 | 9 | 90.063 |
| C1 | 622 | 512 | 2 | 15.265 |
| C2 | 504 | 0 | 1 | 4.580 |
| C3 | 212 | 0 | 1 | 4.580 |
| C4 | 53 | 0 | 1 | 4.580 |

Total: **28 directories, 606.279 GiB**, counting the greater of logical bytes
and allocated blocks for each direct file. No partial directory was observed
at this snapshot. All 28 small COMPLETE receipts passed internal receipt hash,
run identity, tag, four-rank native inventory and declared-file-size checks.
Latest pointers resolve to matching receipts. This is not payload checksum
revalidation or a model-loading test: **zero checkpoint payloads opened**.
Shared status RUNNING is not live process verification; Lead's refreshed host
census must establish actual writers and controllers.

Filesystem available space at the snapshot: **7,483.251 GiB**. It exceeds the
1,024 GiB free floor plus the conservative 1,983 GiB checkpoint allowance by
4,476.251 GiB before subtracting outstanding cache claims/metadata. Actual
policy, claims and per-write floor checks remain mandatory. Filesystem free
space is not proof of an undisclosed project quota; no quota guarantee is made.

## Receipt suggestion and final conditions

Lead may use `remaining_bytes=2129230036992` in a newly pinned
`st_checkpoint_storage_commitments_v1` receipt after checking the accompanying
snapshot, fresh writer census and exact policy package. Set real check/expiry
times when that receipt is created; do not backdate this review or use it as a
permanent CURRENT receipt. The validity must cover the invocation and be
revalidated by the existing storage guard before each bounded write/check.

Only nine explicit formal run roots were inspected. No recursive project scan,
weight rehash, tensor/image read, GPU work, live process signal, cache mutation
or ledger migration was performed. The collector source and full structured
receipt/stat snapshot are included for replay. This recommendation alone does
not approve a build; final exact registration/policy/runtime review remains
pending. It does not enlarge the 4 GiB metadata package, 768 GiB cache envelope
or create an independent second ledger.

```bash
cd /mnt/project/VLA-Drive-st-metadata-actual-admission-20261017
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 timeout 45s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/metadata_actual_admission_v1/collect_checkpoint_metadata.py \
  --root /mnt/project/st-world-policy-20261016 \
  --output /tmp/st_checkpoint_metadata_refresh.json
```
