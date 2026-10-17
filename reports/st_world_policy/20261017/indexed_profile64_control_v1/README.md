# Indexed32 / 64-update registration and launch control

**PREPARED SOURCE ONLY — not dispatched or resource-admitted here.**
`controller.py` reuses the existing immutable ST trainer, registration verifier,
environment capture, native checkpoint implementation, small metadata archive
and advisory lock helper. It never calls the old controller's `m.execute`:
that path hardcodes SMOKE, two updates, GPU1's lock and a1020-second timeout.

This explicit launcher uses PROFILE,64 new updates, physicalGPU0 and
`/tmp/foresight_gpu_0.lock`,1620-second outer timeout plus180-second own-group
cleanup, with the actual trainer1500-second limit. No resume arm, automatic
retry, implicit budget extension, extra trainer, model change or teacher cache
write is added. A partial or failed output remains preserved. A successful
process that reaches an earlier boundary is not mislabeled as64 completed.

## Actual inputs

`INPUTS.json` pins these real completed artifacts, not proposed values:

| Artifact | SHA256 |
|---|---|
| Factory result | `6407b37cd7f457c57f60717c71ef47bcedc9e2d40c090e51239cdc2fcf8be6d1` |
| Final training config | `e72f767f4dfdc349508befb64665bb11f4e407b8e670fc3a31dd983f8c0c7676` |
| Provider descriptor | `e28c677f9d7799ff3db3ed7b62ad5252f336233e14448ffeb171f77f73f21885` |
| Source migration | `3e56f0380236c4ef9be76e5cb15bff75195fe700f28736f5f4a83337591a6976` |

The actual factory finished in9.223s, peak RSS642,953,216 bytes and
3,020,067,533 rchar bytes, zero teacher tensor writes/GPU execution. Those are
factory costs, not training throughput. Same32-scene manifest and original
Base/PCA remain pinned. Exact source is
`2d5e9483a8c8bc80b71348211756f3cab6aa21b1`, immutable checkout
`/mnt/project/VLA-Drive-st-indexed-profile-runtime-v1-20261017`.

Config preserves world1/micro1/global2,20k cosine/warmup1000, original objectives,
BF16/FP32-master/ZeRO2 execution and FP32 fixed8 teacher. `save_every=64` writes
one normal final native checkpoint. The existing trainer can save an earlier
boundary on interruption and exit; it does not write unconditional step0.

Independent output, which this task has **not created**:
`/mnt/project/st-world-policy-20261016/profile/world_indexed_provider_profile32_updates64_v1`.
Run ID: `indexed_provider_profile32_updates64_v1`.

## Real resource admission is external

`prepare` and `run` require a small Lead-created external admission receipt and
its SHA256. The default action is read-only `preflight`; it does not query GPU,
acquire locks, create a lease or make output directories. This controller does
not write the shared storage ledger or manufacture approval.

The explicit receipt schema is `st_indexed_profile64_resource_admission_v1`.
Lead fills actual times and records the reservation only after independent
source/control/resource review. Required fields are checked in `admission()`:

- `status=ADMITTED`, host `training-vla-zt-worker-0`, GPU`0`.
- Actual source SHA above, run ID/output root above.
- `controller_sha256` and `inputs_sha256` from the exact launched files.
- `checkpoint_reservation_bytes=85899345920` (80GiB).
- `other_commitments_bytes=2258079055872` (2103GiB).
- `free_floor_bytes=1099511627776` (1TiB).
- `max_invocation_seconds=1500`, `outer_timeout_seconds=1620`,
  `cleanup_seconds=180`.
- Finite actual `valid_from_unix`, `valid_until_unix`; at least1800s must remain
  at each prepare/run admission check. Fresh lease inherits the real expiry.

At both preparation and dispatch, storage checks require actual free space of
at least the full2103GiB existing commitments +1TiB floor + this run's remaining
80GiB. Existing owned bytes are counted by max(logical,allocated), with symlinks
rejected. No old reservation is reduced and no file is deleted. One measured
native directory is37,365,928,655 bytes; two conservative complete/partial slots
fit74,731,857,310 bytes, leaving11,167,488,610 bytes within80GiB. This is advisory
accounting, not verified filesystem quota.

Preparation takes an actual idleGPU0 UUID under its lock, captures fresh actual
environment, saves the precise config/target/lease and registers all seven
required artifact pins. No model/teacher is initialized. Run rechecks host
authorization, clean source, input hashes, environment, lease, GPU UUID, idle
state, storage and duplicate output/launch guards. Source/control changes need
new registration; the old two-update registration/lease cannot be reused.

## Commands

These commands are real CLI interfaces. Only `--help`, CPU tests and metadata
preflight were executed during this handoff. Set `CONTROL` to the reviewed
controller file and `ADMISSION` to an actual Lead-created receipt; no admission
path or approval hash is fabricated in this repository.

```bash
CONTROL=/mnt/project/VLA-Drive-st-indexed-composition-20261017/reports/st_world_policy/20261017/indexed_profile64_control_v1/controller.py
/root/miniconda3/envs/ddp/bin/python "$CONTROL" --help
/root/miniconda3/envs/ddp/bin/python "$CONTROL" --action preflight
```

After Lead review, pin the exact controller/adjacent `INPUTS.json`, create the
external reservation/admission and use its actual path/hash:

```bash
test -n "${ADMISSION:?Set the actual Lead admission path}"
test -n "${ADMISSION_SHA256:?Set its actual SHA256}"
/root/miniconda3/envs/ddp/bin/python "$CONTROL" --action prepare \
  --resource-admission "$ADMISSION" --resource-admission-sha256 "$ADMISSION_SHA256"
/root/miniconda3/envs/ddp/bin/python "$CONTROL" --action run \
  --resource-admission "$ADMISSION" --resource-admission-sha256 "$ADMISSION_SHA256"
```

If a private control copy is preferred, copy `controller.py` and `INPUTS.json`
together before recording the actual controller hash/admission. Preserve exact
source. No old scripts or evidence are overwritten. The child command remains
the existing `torch.distributed.run ... tools.st_world_policy.train_world`
with this new registration. The parent stays attached through result/archive
completion; an external launcher may background this single controller without
adding a second training dispatch.

## Tests and remaining evidence

Six CPU tests pass, with negative subcases for changed scope/update/population/
schedule, wrongGPU/expired or mismatched admission, busyGPU rejection, missing
2103GiB storage accounting, invalid byte counts and read-only default behavior.
The busyGPU/admission tests are explicit mocks, not actual GPU/resource checks.
`TEST_OUTPUT.txt` contains their output. Actual metadata preflight and CLI help
are separately recorded. No teacher/index/tensor/weight/image content was read.

Still pending: independent review of this controller, actual resource admission,
fresh registration/lease, and one real64-update run. Exact32 repeats cover128
exposures, not128 unique scenes; this is a supply/memory stability diagnostic,
not full-population throughput, formal profile qualification or model quality.
