# Independent review: no-sidecar controller and explicit dev attempt 2

**PASS for one exact explicit dev attempt 2.** No real builder was started by
this reviewer; no train, third attempt or automatic retry is authorized.

Owner commit `1ee0660c901547df23cc6f852834954ca112bd38`, clean. The exact executable
remains the owner path
`/mnt/project/VLA-Drive-st-full-future-metadata-preparation-20261017/reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py`.
SHA `a866ae4261cad36312355028cdfba3b24f52f64b174433635d0473420ddc84bf`.

## Changed scope

All background counter-file writes and the counter monitor thread were removed.
The original metadata builder, registered source/runtime, PLAN, 1,696-scene /
16-log selection, shard format and original storage guard were not changed.
Parent polls `/proc` I/O only for the current verified unreaped child, with
identity checks before and after reading. It captures terminal I/O before the
first reap. Child reuses the existing Counters only to create its final RAM
snapshot and stdout envelope; its write function is temporarily a no-op and is
restored afterward. Thus the observer no longer creates a temporary file while
the builder inventories its directory. There is no ignore-ENOENT workaround or
weakened path/hash check.

Missing or malformed final accounting is a failed result. Where neither terminal
`/proc` nor final child receipt supplies the final count, the remaining read
allowance is conservatively charged in full. Only newly owned processes can be
signalled. Finished/reaped children are not cleaned twice. All existing source,
checkpoint, free-space, directory and one-shot marker guards remain in place.

## Current-source CPU results

**53 passed in 4.99 seconds, no skips:** 33 owner cases and 20 independent cases.
Independent cases cover exact remaining budget and prior-failure identity,
refusal of attempt3/bool/reset-budget/train reuse, RAM-only final counters on
success and source failure, preserved original error, identity change rejection,
invalid counter types, and seven tiny owned CPU children. The latter exercise
success, nonzero child exit, malformed final envelope, RSS/read/wall limits and
pipe overflow; all use real process ownership/cleanup. Every observed child I/O
read preceded the first reap, cleanup found no live owned tasks, and no CONTROL
sidecars were written. No real cache builder, model, GPU, image, raw log payload
or ledger write was used in these tests.

Prior initial-controller 29/1 failure, later 31-pass revision, and the real
48.586986-second dev failure remain separate preserved evidence. The old tests
did not cover the inventory/atomic-sidecar race; this new design removes that
writer rather than claiming those old tests proved the full workload.

## Exact actual bindings and budget

The retained failed result SHA is
`60b7ba6177b51081fdb323b36a0821f3848e6ddf2c2bb625b97e9a8c850aefda`.
Its recorded charge is 2,202,810,181 bytes. The original role cap is 17,179,869,184
bytes, leaving **14,977,059,003 bytes** for this explicit attempt. The controller
verifies that exact result, charge and zero-live-child cleanup before entering
attempt 2. It writes distinct `CONTROL_BUILD_ATTEMPT2_STARTED.json` and
`CONTROL_BUILD_ATTEMPT2_RESULT.json`; the original START/RESULT/counters are
retained, as are two legal shards and the orphan partial. The original builder
must verify identities before reusing a shard. There is no attempt 3 route.

All nine original PLAN/control hashes were rechecked and match. Current live
ledger SHA is `f9099a472140e1b29daaaaf87bea9f297d924e29e9616fe450452b5361419377`.
Native migration, policy, registration, checkpoint receipt, dispatch policy and
three original monitor-helper hashes are unchanged. Current shared free space
is 7,254,744,834,048 bytes; quota remains NOT VERIFIED. Full 2,183 GiB checkpoint
obligations and additional 1 TiB floor remain reserved through 15:20:52 UTC.
The 128 MiB dev root includes existing partials and new attempt controls. The
same 768 GiB common cap and 5,620 MiB planned metadata /8 GiB envelope remain.

The invocation retains 2 CPUs, sampled process-tree RSS 4 GiB, at most 1,800
seconds plus 40 seconds owned cleanup, and two 1 MiB bounded pipes. Source
request pacing is 64 MiB/s, not a guarantee that physical block/Lustre traffic
is capped at that rate. Actual first-run child physical reads substantially
exceeded its logical `rchar`; both figures remain reported in the failure
closure. No performance claim or full-future tensor-cache approval is implied.

## One executable retry token

Token path:
`/mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/full_build_admission_v1/DEV_BUILD_ATTEMPT2_RESOURCE_ADMISSION_V1.json`

SHA: `a8aedf096e29a28552e5e2e8e9a981d53288ca3a8748d16001c1e2a42d14b15c`.
Latest start 2026-10-17 10:01:39 UTC; expiry 10:46:39 UTC.

```bash
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-full-future-metadata-preparation-20261017/reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py \
  --admission /mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/full_build_admission_v1/DEV_BUILD_ATTEMPT2_RESOURCE_ADMISSION_V1.json \
  --admission-sha256 a8aedf096e29a28552e5e2e8e9a981d53288ca3a8748d16001c1e2a42d14b15c
```
