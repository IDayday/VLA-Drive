# One explicit SP0 dev extraction retry — preparation only

Base: `bdd750a4489f4a5019cd4d33d3bde9571da344d2`.
Only `spatial_probe_actual_preparation_v1/supervise_once.py` changes at runtime.
This is not a generic retry framework, admission, dispatch, new plan or model run.

The signed/pinned admission may contain `manual_retry` **only for dev_extract**:

- `failed_result`: exact original path and SHA256
  `18a351f58be8e1b9c561a7e585a7bd6e600118d674bf2beafb5192de337b9327`;
- `plan_result`: exact original dev_plan result path and SHA256;
- `attempt_directory`: exactly
  `/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/stages/dev_extract_manual_retry_v1`.

The failed receipt must still be STOPPED_NO_AUTO_RETRY / MONITOR_UNREADABLE,
have zero remaining owned live processes, and bind the same PREPARATION.
Existing plan receipt must be COMPLETE with the same preparation. Runtime,
input/checkpoint identity, actual plan signature, typed W claim, global loader
lock, GPU lock, fresh GPU and checkpoint/storage checks remain in force.

Exactly one fixed retry directory is created under the existing global lock,
using `mkdir(exist_ok=False)`. A duplicate call or failed retry cannot choose a
new timestamp or retry again. No original path is deleted or overwritten.

## Cumulative read budget

The accounting remains `max(hash_admitted + nominal_torch_load, proc_rchar)`
per attempt, summed across consumed attempts. It includes admitted in-flight
hash bytes, not just completed hashes. It is conservative recorded logical IO,
not exact physical storage traffic; the prior monitor snapshot limitations are
not concealed.

| Original receipt | Charged bytes |
|---|---:|
| dev_plan, SHA dd765726… | 37,533,999,575 |
| failed dev_extract, SHA 18a351f5… | 30,632,210,276 |
| Total previously charged | 68,166,209,851 |
| Original role ceiling | 274,877,906,944 (256 GiB) |
| Explicit retry remaining | **206,711,697,093** |

Negative, noninteger, absent or exhausted counters fail closed. The new worker
receives only the remaining cap. ADMITTED, success and failure RESULT receipts
record source result pins, consumed/read remainder, actual attempt directory
and retry lineage. There is no new dev_plan execution to reset accounting.

Small-control inspection confirms the failed final COUNTERS content matches
the snapshot in the immutable failed RESULT. The original unique cause remains
UNKNOWN_NOT_RECORDED; this retry preparation does not revise that history.

## Following the retry

`train_plan` must explicitly provide `previous_stage_result` pinning the new
retry's successful RESULT. It must be COMPLETE, same PREPARATION, stage
dev_extract, exact retry directory and valid original retry lineage. Without
this pin, the original failed dev_extract remains the predecessor and blocks
progress. The override is rejected on every other stage. Later train_extract
and fit follow their ordinary successful predecessor receipts, preserving the
explicit chain. The old failure is never masqueraded as success.

## Unchanged runtime / source separation

The actual PREPARATION remains SHA
`34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431`.
Its clean model runtime is still
`/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017` at
`581399eb309677c20fe7464791b4f18685e947a6`.
No frozen runtime/helper file there was edited. Lead can copy the three control
helpers into a new independently pinned control directory; the unchanged child
observer imports the original model runtime, whose HEAD/clean check remains.

No storage/resource cap is increased. The new attempt retains the 5 MiB bounded
control-directory envelope and original logs/error reserves; admission review
must account for this additional control directory inside existing administrative
capacity. No tensor/ledger claim is added here. All v7 claims and prior reserved
administrative reports remain protected.

## Tests and use

**72 passed in 6.03 seconds**: 50 existing lifecycle/admission cases, nine mutable
counter cases and 13 new manual-retry/accounting cases. New cases reject wrong
stage/path/hash/fields/preparation, consumed directory, exhausted/malformed
counters, failed predecessor and generic downstream overrides. The original
failure bytes remain intact in fixtures. CPU affinity `{0,1}`, 2 GiB address
space, empty CUDA visibility, 90-second suite timeout; no real GPU/payload/run.

`METADATA_PREPARATION.json` contains actual small receipt pins and the exact
fields to add after independent approval; it is explicitly NOT_ADMITTED.
No usable APPROVED admission is generated.

The CLI remains:

```bash
/root/miniconda3/envs/ddp/bin/python /path/to/independently-pinned-controls/supervise_once.py \
  --admission /path/to/lead-issued-admission.json \
  --admission-sha256 ACTUAL_REVIEWED_ADMISSION_SHA256
```

These placeholders intentionally identify artifacts that do not yet exist;
they are not an executable launch authorization. Lead must issue exact helper
hashes, approved stage/host/CPU/GPU/lease, current storage and checkpoint-space
pins, and the manual fields. Preserve the old admission and old failure.
This implementation does not retry or progress any actual stage.
