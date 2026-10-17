# ST formal dependencies, 2026-10-17 07:45:11 UTC

Metadata-only snapshot of the nine registered formal runs. No SSH, training,
GPU execution, scoring, dispatch, weights hashing, partial evaluation rows or
Navtest access. Prior C0/C1/C2/SP0 result reports remain unchanged.

## Nine training runs

`RUNNING` below means a recently updated shared `status.json`, not an
independent remote-process inspection. Exact observation timestamps and
status-file hashes are in `STATUS.csv` / `SNAPSHOT.json`.

| Arm | Reported state | Reported updates | Latest complete native step | Host |
|---|---|---:|---:|---|
| C0 | COMPLETE / MAX_STEPS | 4096 | 4096 | tsinghua-zt4 |
| C1 | RUNNING | 1047 | 1024 | tsinghua-zt4 |
| C2 | RUNNING | 942 | 512 | rl-zt2 |
| C3 | RUNNING | 644 | 512 | rl-zt2 |
| C4 | RUNNING | 486 | 0 | tsinghua-zt4 |
| SP0 | PAUSED / BUDGET_PAUSED | 5000 | 5000 | vla-zt |
| SP1 | PAUSED / BUDGET_PAUSED | 5000 | 5000 | tsinghua-zt |
| SP2 | PAUSED / BUDGET_PAUSED | 5000 | 5000 | vla-zt |
| SP3 | RUNNING | 4691 | 4000 | tsinghua-zt |

C0–C4 retain the registered4096-update endpoint and source6714b1e9. SP runs
retain sourcefdf25f99 and a5000-update screening stop on the unchanged20000-step
learning-rate schedule. Paused SP0/SP1/SP2 is the registered budget boundary,
not a claim of reaching the20000-step common endpoint or permission to resume.

For all nine runs, this inspection validates registration/config/approval
metadata, source identity, latest pointer, exact receipt self-hash, native
inventory, world size, filename/tag and every declared payload's existence and
byte size. Fixed512/1024/4096 or5000 endpoint receipts are separately listed.
No multi-GB model/optimizer payload was reopened; full payload hashes remain
the export reader's required check before model use.

## New dependencies and actual active exports

The two newly available checkpoints were handed to Lead during this audit.
Lead subsequently launched their existing registered evaluations. Their
actual route and launch receipts have now been read and linked to the same
native COMPLETE identities; **do not launch duplicates**.

| Export | Native tag | GPU on vla-zt | PID | Campaign slot | Observed |
|---|---|---:|---:|---:|---|
| SP1_5000 | step_005000_1792219550866622410 | 1 | 440343 | 0 | Existing, process present |
| C3_512 | step_000512_1792220910111409135 | 2 | 445432 | 1 | Existing, process present |
| SP2_5000 | step_005000_1792222283615035511 | 3 | 449050 | 2 | Newly launched by Lead, present |
| C1_1024 | step_001024_1792222710626368133 | 4 | 449179 | 3 | Newly launched by Lead, present |

All four known processes were checked locally on the actual launch host
`training-vla-zt-worker-0`; command lines match their exact registered output
directories. Only named job metadata and command hashes are saved, not raw
process inventories. No process was signalled. No shell exit code is inferred.

**All four export slots are occupied.** C4 currently has no complete512-step
checkpoint. Its state is `WAITING_EXACT_COMPLETE_NATIVE_CHECKPOINT_THEN_EXPORT_SLOT`;
even if the checkpoint becomes complete, retain `WAITING_EXPORT_SLOT` until an
existing export releases a slot. This report neither expands C4 implementation
nor dispatches it. C2_1024 and SP3_5000 also await their exact complete receipts.

New pinned dependency records:

| Dependency | COMPLETE SHA256 | receipt_identity |
|---|---|---|
| C1_1024 | 5bcfb80fb1d656c99ef68c324cab5f5587584721f4c38eb3a5ac7d5f265541c7 | 8607eb1225b152155a2fc95af33f8535e976ee243a7b7c44f9717d168896eebb |
| SP2_5000 | 9e8b7694996130c1c6148519c27c9d562df106b46553357b4218e68245637183 | 784fa31313d8653e29142a0f0287ffa0abc4d9a47535a2d21f90892ff39714f0 |

C1 route SHA256:
`3fe30afe13b2259d55d22b34c873486f372cd3ac93e7d3b6ea386edf334857f8`.
SP2 route SHA256:
`9167c97bb1cb42f8d6f9fba683849a654742b8921cdff3c03adc1a5a59749fa8`.
Full receipt paths and run identities are in `SNAPSHOT.json`.

After these exports finish, the next executable dependency is their complete
prediction receipt for the **already registered** single-trajectory official
score stage. Lead must preserve the existing scoring concurrency/budget,
source/manifest/route and duplicate-run checks. A present checkpoint alone does
not authorize an extra export or scorer process.

## Existing completed dev evidence

The following COMPLETE-file hashes still match the earlier full-scene audits.
Their1696 valid/0 invalid/0 unresolved denominators are intact. This snapshot
does not redo ledger loading or scoring.

| Arm / updates | PDMS /100 | Zero scenes |
|---|---:|---:|
| C0 /512 | 90.989352 | 46 |
| C0 /1024 | 91.076647 | 48 |
| C0 /4096 | 91.138335 | 45 |
| C1 /512 | 91.169425 | 43 |
| C2 /512 | 91.204624 | 43 |
| SP0 /5000 | 90.799174 | 48 |

This is an inventory, not a cross-step or cross-architecture comparison.
Early diagnostic scores do not select winners or justify stopping any arm.
No in-progress export/scoring result is interpreted as complete.

## Reproduction and cost

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 timeout 120 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/formal_dependency_status_v1/inspect_status.py
```

Final metadata inspection:0.23seconds,16.73MiB peak RSS,two-CPU affinity.
The snapshot is timestamped, so later live progress must be checked afresh by
Lead before any dispatch. A mutable status SHA is evidence of what was read,
not a promise that the file remains unchanged while training continues.
