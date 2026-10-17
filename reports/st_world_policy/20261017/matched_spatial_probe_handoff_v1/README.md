# Matched SP1/SP2/SP3 learned-W probe handoff

Status: **metadata located; no registration, cache claim, extraction, fit or dispatch performed**.
This report does not select a spatial arm or change the experiment. The handoff
starts at `12d86714317ab1873dcf0103d7b8bdfb5bcbc2a5` in an isolated worktree.
`ASSET_INVENTORY.json` contains exact paths, checksums, receipt identities,
registered configs and stat-only native-file versions.

## Existing assets and fixed comparison

All four native checkpoints have complete small receipts at exactly update 5000,
scope FORMAL, from training source `fdf25f99b702754787904215bf1c2a2b7515ea75`.
Receipt semantic identities, export identities, checkpoint descriptors, matching
native-file sizes and the frozen runtime's protected deployment AST were checked.
**No native tensor was opened or rehashed**. The original `cache_plan` must still
perform its full native-checkpoint and image validation when separately admitted.

| Arm | Frozen checkpoint tag | Descriptor identity | Native bytes, from receipt/stat |
|---|---|---|---:|
| S-P0 | step_005000_1792217229597178910 | c87c59804b6237205e4b4a7a7e355b569c7cef5eaa499023f699111ec36d0beb | 37,366,035,746 |
| S-P1 | step_005000_1792219550866622410 | c088115fdeffa355d2b6ebf8a807ea01112a22c90059919b39f5372456af2d28 | 37,366,035,746 |
| S-P2 | step_005000_1792222283615035511 | 2de4bf93f817a82f673d5d0c53af20a264c2e5b0a76e0263336283091cc5bdb1 | 37,366,035,746 |
| S-P3 | step_005000_1792224403643960209 | 52beafa2273e3cb9900ce3aed824a0af19683a5e22e935cb2e4d6ef9d3fc5058 | 37,371,024,866 |

Each native root is
`/mnt/project/st-world-policy-20261016/formal/spatial/runs/SPn_fulltrain_seed42_official_5k_v1/checkpoints/<tag>/`.
The original small `COMPLETE.json` describes nine native files; it is not a claim
that the full checkpoint was independently loaded by this handoff.

Reuse exactly the original runtime
`/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017` at
`581399eb309677c20fe7464791b4f18685e947a6` and original protocol
`reports/st_world_policy/20261017/spatial_probe/PROTOCOL.json`, SHA
`7e2dcd4a938dba575f4000f641f59e2cd48a80ad1ece190b0a31000596fc956e`.

- Train: 64 scenes / 64 logs, manifest SHA `58387c3779b083cd83b55f2339e49dc99ff12b136e20ea672b9dc0c50e5b40c1`.
- Dev: 16 scenes / 16 logs, SHA `b544911f882b44631b515107f955c96bc43f6933a4c49bfdad1fc5dc4798d148`. No train/dev log intersection.
- Existing current-raw-label manifest SHA `5e43b13b382fce51fdd07dcf8e81f5168495bd66aeab2a535cdf90cceab93e9a`. Label tensors were not read or regenerated.
- Same seed 42, 300 updates, two scenes/update, 32 sampled patches per current view, three views, hidden 128, one layer, four heads, two outputs, LR .001, WD .001, clip 1. No dev tuning.
- Same 600 scene presentations / 57,600 sampled patch queries per arm; evaluation covers all 432 positions per view. These are recipe counts, not new observed execution results.

This remains a learned-W spatial capacity diagnostic, not full 1696-scene PDMS,
an untouched test or proof of generalization. The labels are current raw-LiDAR
depth and current raw-box/LiDAR-supported small-vehicle region proxies.

## W size and unchanged probe semantics

`spatial_probe.validate_binding` explicitly accepts all four arms. S-P0/P1/P2
use W `[144,2048]`; S-P3 uses `[288,2048]`, all FP32, with a Boolean valid mask.
`SparseFeatureReadout` projects the full W memory once, creates spatial/time/view
queries separately and cross-attends to every valid memory token. It accepts a
variable memory length without cropping, padding, query-count changes or a new
readout. The parameter count remains the SP0 observed **495,362**; SP3 has longer
K/V memory and higher memory/compute cost. Equal updates are not equal FLOPs.

The original extraction is observation-only, FP32 native-master evaluation with
TF32 off. No labels enter W extraction. Only the separate CPU fit reads current
labels. Source inspection is not substituted for actual SP1–3 output validation.

## Byte budget before registering three new groups

The estimates below exactly reproduce the original pure `cache_requirements`
function, checked without importing the runtime. Each split reserves output,
one staged record plus 4 MiB, and bounded metadata. Output includes 8192 bytes
of per-record overhead. These are conservative reservations, not measured new
cache sizes.

| Arm | Train64 reservation | Dev16 reservation | Combined |
|---|---:|---:|---:|
| Existing S-P0 | 86,254,736 | 26,085,776 | 112,340,512 B |
| Proposed S-P1 | 86,254,736 | 26,085,776 | 112,340,512 B |
| Proposed S-P2 | 86,254,736 | 26,085,776 | 112,340,512 B |
| Proposed S-P3 | 162,941,216 | 46,142,240 | 209,083,456 B |
| All four | | | **546,104,992 B (520.806 MiB)** |

New SP1–3 reservations would total 433,764,480 B. Under the frozen-W **1 GiB
aggregate** cap, all four leave 527,636,832 B. Only the SP0 cache root exists;
the SP1–3 roots are not yet created. No storage ledger or claim was modified.

The original CPU fit reserves at most 64 MiB per arm for two durable slots and
staging; three new fits therefore require a separate 192 MiB allowance, plus
their bounded control/log namespaces. These fit/control bytes are **not** hidden
inside the W formula. Fresh storage/resource review must identify their proper
envelopes before registration. Native checkpoints remain existing read-only
assets and are not duplicated into the W budget.

Small W output does not make extraction a small I/O task: each native checkpoint
is about 37.4 GB and the original checker repeats integrity reads. SP0 TRAIN64's
real extraction charge was 153,889,914,639 B plus 37,575,105,055 B for its plan;
wall 466.292 s, sampled RSS 26.805 GB, GPU allocated/reserved 12.838/13.288 GB.
These are historical SP0 measurements, **not** promises for SP1–3. The original
CPU SP0 fit completed 300 updates in 10.826 s supervisor wall, RSS 793,411,584 B,
zero GPU/native loads. Fresh per-stage read/wall/RSS/GPU admission is required.

## Original SP0 failure and the smallest affected control change

SP0 TRAIN64's original supervisor receipt remains `STOPPED_NO_AUTO_RETRY`,
`OWNED_DESCENDANTS_AFTER_LEADER_EXIT`, despite CLI exit 0 and 64/64 COMPLETE.
Its tensors were later independently verified and accepted through an exact
TRAIN64 reconciliation; the original CPU fit then completed. Neither receipt
is rewritten here. `verified_processes=6/live_owned_remaining=0` is not enough
to identify or infer the state of the five other processes at leader exit.

The old `_supervise` calls `active=owner.discover()` before `leader_done()`.
The leader can exit between these calls; a stale active list is immediately
classified as a residual process. The already-reviewed `4b2f...` monitor fixes
that narrow race by rediscovering after WNOWAIT confirms leader termination.
For the new isolated SP1–3 controller, preserve the existing extract observer,
GPU and logical-hash/load counters, ownership, limits and cleanup, and apply only:

```python
if done and active:
    active = owner.discover()
    if active:
        reason = reason or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
```

Real surviving descendants must still fail and be cleaned by the mature pidfd
owner. This is a prospective recommendation, not a signed launch PASS or proof
that the historical failure was solely that race. Do not repeat the old loop
and assume each arm will be rescued through another expensive reconciliation.

Additional changes are fixed-control bindings only: original
`prepare_bindings.py` and supervisor `JOB`/`STAGES`/paths and storage bridge pin
SP0 literally. New control source should enumerate only SP1/SP2/SP3 and the exact
checkpoint/registration pins in this inventory, use separate output roots, and
retain the original validators and step order. Do not permit arbitrary commands
or turn the exact old TRAIN64/DEV16 reconciliations into general failure bypasses.
No extraction/readout/fit mathematics or frozen runtime needs a change.

## Bounded evidence collection

Actual command:

```bash
taskset -c 10,11 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/matched_spatial_probe_handoff_v1/capture_metadata.py
```

Final capture passed all metadata assertions in 0.553 s, read 5,954,689 bytes of
explicit unique source/metadata (56 files), `/proc` rchar 6,847,146 B, actual
executable VmHWM 34,598,912 B. It used a 1 GiB address-space limit and 120 s timer.
No torch import, tensor, image, DB or full training manifest read; no dispatch,
GPU operation, cache write, ledger write or registration.

Two preceding metadata-only attempts reached the same final resource assertion
but failed because `getrusage.ru_maxrss` retained a pre-exec 1.70 GB high-water
mark; `CAPTURE_ACCOUNTING_FAILURE.json` preserves this failure. The final report
uses `/proc/self/status` VmHWM for the current executable and retains the raw
getrusage value. Three explicit captures read 17,864,067 source/metadata bytes
total; ancillary small source/control inspection kept logical source/control
reads below 64 MiB. Final OS `read_bytes` was 83,963,904 B and is reported
separately: filesystem/page-cache read-ahead is not the logical source-byte
budget or a payload read. No physical-I/O benchmark is claimed.

Required next reviews: fixed three-arm bindings and stage prerequisites;
stale-active rediscovery regression plus true surviving-child rejection;
unchanged GPU/logical-read accounting and bounded bootstrap; separate typed-W
and fit/control reservations; real plan signatures, then one bounded extraction
per fixed split. This handoff grants no launch approval.
