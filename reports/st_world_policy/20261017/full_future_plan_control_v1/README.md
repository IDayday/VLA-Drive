# One-attempt full-future metadata PLAN control

Prepared from `b805b00e80e1cc4e28f8e4a8f6ba7b79894fb9ab` in an isolated worktree.
No real plan, build, image read, raw-log read, tensor read, GPU use, ledger mutation,
or new resource admission was executed by this implementation agent.

## Scope

`plan_once.py` is a thin parent/child entry around the existing
`prepare_future_metadata.plan`, `supervise_once.supervise`, and
`observed_cli.Counters`. It does not add a trainer or a retry framework.
The W-artifact reconciliation change is a separate commit and is not a dependency.

The final independently issued admission must pin:

- the wrapper's absolute path and SHA256 in `wrapper`;
- exact argv and runtime source files;
- the original three supervision helpers;
- live storage ledger and existing 2183 GiB checkpoint commitments.

The preliminary dev admission used navsim Python. This wrapper rejects that argv:
the final pinned argv must use `/root/miniconda3/envs/ddp/bin/python` explicitly.
No silent interpreter substitution occurs. Resource owner prepared
`DEV_PLAN_ARGV_V2.json`, SHA
`7c2c3332d7c530d8da54db42a7937996037364e54913bd2855e3bbaf1fb22632`.

For train, the same wrapper only accepts a separate exact train admission,
`full_future_metadata_train_v1`, the entire admitted scene count, and the proposed
192 MiB raw-log cap. The independently prepared train argv SHA is
`6019166052234371d80e7bf27fa0582956fdd548fcc4243f5b377b3c28b39f1c`.
It does not authorize dev-to-train inference or removing oversized logs.

## Execution bounds

The parent verifies helper/source/argv pins before creating the output root and
before spawning. It rejects existing roots and leaves a durable consumed-attempt
marker before `Popen`. Failed starts remain spent; no retry loop exists.
Existing shared one-loader lock and live-session/pidfd ownership logic are reused.
No historical PID is used for cleanup.

The outer invocation is CPU 0/1, 2 GiB parent-plus-child RSS, 4 GiB combined parent
and child `/proc/*/io` rchar including imports, 180 seconds, 32 MiB inclusive output,
and 4 MiB maximum file size. The child reserves 8 MiB of the read allowance for
parent control reads; the parent additionally checks the actual combined total.
Read counters are logical syscall/hash observations, not physical disk traffic.
RSS and time are sampled/cooperative bounds, not a cgroup hard ceiling.

Original supervisor cleanup is preserved: TERM grace 30 seconds followed by up
to 10 seconds for final owned-session closure. Admission validity therefore must
cover start + 220 seconds. No change to its signals or ownership implementation.

`COUNTERS.json` is the only mutable read handled specially: it is read under its
original writer `control_lock`. All other controls remain strict hash/stable-file
reads. Counter observation starts before runtime imports; the top-level proc IO
baseline precedes all monitored helper/runtime imports. The fixed plan imports
need no model or optimizer; a CPU-only `is_initialized=False` facade is sufficient
for the existing Counters interface. CUDA visibility is empty. The actual plan
source is pinned, and file-open audit rejects raw logs, images and tensor formats.

All outputs, logs, counters, locks and partials remain below the single output
root. The root census counts root/directories, temporary files and controls,
charging `max(st_size, st_blocks*512)` for each. RLIMIT_FSIZE protects
the child; existing supervisor bounds its log. A 5 MiB control reserve is kept
while the plan writes metadata. Original proposed *future build* limits remain
in PLAN and are not expanded into execution authority.

Fresh shared free capacity must retain the entire 2183 GiB obligations plus
1 TiB free floor plus this 32 MiB root allowance. The pinned live ledger must
still match. This script never migrates policy, issues a claim or rewrites ledger.

## Tests and exact invocation

CPU fixture-only negative checks:

```bash
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  reports/st_world_policy/20261017/full_future_plan_control_v1/test_plan_once.py
```

Actual result: **33 passed**. Cases reject wrong status/schema/time window,
NaN/bool timing, prohibited GPU/build/raw/model scope, arbitrary commands,
wrong interpreter, duplicate/unknown flags, changed population/budgets,
unavailable CPU, dev raw limits in train, symlinks, nonregular/oversized files,
root overflow, changed source hashes, and a preexisting output namespace.
`--help` was also executed successfully. No real workload or fake-result fixture
is presented as metadata preparation evidence.

After separate review, Lead runs the exact final admission supplied by resource
review (no admission path or SHA is invented here):

```bash
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/full_future_plan_control_v1/plan_once.py \
  --admission "$FINAL_ADMISSION_PATH" --admission-sha256 "$FINAL_ADMISSION_SHA256"
```

The only successful completion status is `PLAN_COMPLETE`. It means the single
metadata proposal exists and the outer monitor passed; it authorizes no build,
index, teacher extraction, training or scientific conclusion. Failure preserves
the root and evidence under `control/RESULT.json` as `*_NO_AUTO_RETRY`.

## Narrow read-atime correction

After the original commit's independent review, Lead's real preflight rejected
the correctly hashed admission before creating any output root or child. Lead
observed that Lustre changed `st_atime_ns` on read while content-related fields
were unchanged. Original `pinned()` compared whole `stat_result` objects, thereby
treating access time as a content change.

The correction compares only device, inode, mode, link count, size, nanosecond
mtime/ctime, and allocated blocks. SHA256 is still mandatory. Atime alone cannot
trigger rejection; a changed inode, size, mtime/ctime (including nanoseconds),
mode, link count, device, allocation or content still rejects.

The original 33-test log and failure evidence remain preserved. The added CPU
fixtures explicitly vary each field and retain the existing changed-content/hash
negative test: **42 passed in 0.10 seconds**, recorded separately in
`ATIME_FIX_TEST_OUTPUT.txt`. No real plan was rerun by the implementation agent.
Fresh resource tokens must bind the changed wrapper; old tokens are not reused.
