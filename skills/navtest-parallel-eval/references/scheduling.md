# Resource and completion decisions

Use a read-only plan when asked to plan. `--serve`, GPU probes, remote inspection
and asset builders have actual side effects and are not dry-run substitutes.
For execution, record the actual policy version and selected current routes.

## Admission

First identify the physical GPU UUID, the visible owner process and its
PID/start time, full executable entry point and applicable job registration.
Reconcile CUDA_VISIBLE_DEVICES and local rank with the physical device list.
Where the namespace hides a process, treat its identity as unknown; container
names and momentarily low GPU utilization do not identify a filler. Confirm
own training from its registered run and native worker command, not a permanent
host-name list.

Measure model peak memory on the real current-input path. Require repeated free
memory at least peak plus the registered reserve. A conservative larger guard
may be used, but record it rather than calling it the model's actual requirement.
Measure training update time before and during cohosting. A short profile on one
GPU can underestimate the sustained effect of a full inference deployment. Under
the user's fastest-completion preference, use these measurements to report total
cost and choose concurrency, not to throttle otherwise authorized inference.
Refresh routes while keeping the model, evaluation protocol and training recipe fixed.

| Device state | Decision |
| --- | --- |
| Idle, enough memory, runtime/input verified | Use it. |
| Own training, cohosting authorized, enough memory | Use full inference duty within capacity; retain training and report contention. |
| Genuine other work, repeated low utilization, sharing authorized, enough memory | Use within qualified sharing limits; retain that work. |
| Genuine other work with sustained high utilization | Exclude unless sharing that load was explicitly authorized and qualified. |
| Confirmed pressure script | Journal and release only the selected affected devices, remeasure, then admit. |
| Unidentified process or insufficient memory | Do not kill it; exclude until identity/capacity is established. |
| Alias UUID or unreachable host | Do not count another GPU or reuse stale availability. |

Count only eligible unique UUIDs. Derive concurrent model processes from that
count. Use small reclaimable work chunks; chunk size is an execution tuning
parameter, not a fixed GPU or shard count. Faster workers should claim more
scenes so busy cohosts do not force equal scene counts. Keep claims, lease
generation and per-scene publication fenced by one shared exclusive ownership
mechanism. Verify that locking works on the actual shared filesystem before
relying on it across hosts.

## CPU capacity

Check CPU affinity, cgroup quota and current load rather than os.cpu_count alone.
Do not double count CPU capacity from aliased containers. Reserve capacity for
native training, image preprocessing, staging and control. Canonical scoring
processes use one native numerical thread each. Start a bounded pool, observe
ready-scene backlog and score rate, and increase only within spare capacity.

The current streamer is a single-host CPU pool; do not run two full-population
streamers against one output directory. Multi-host scoring requires exclusive
token/variant partitions or a shared claim mechanism, plus one final merge
owner and matching numerical runtime. A local pool may already be sufficient:
concurrent CPU scoring does not require a separate pool on every GPU host.

## Pressure restoration

Keep a private journal of the original PID/start time, resolved interpreter and
script, argv, cwd, UID/GID, descendants, GPU UUID mapping and required environment.
Use mode 0600; publish only redacted status and counts. Acquire once, verify
release and record the obligation before GPU launch. Restoration reuses the
original parameters and limits, checks worker exit and whether new genuine work
has acquired the device, then verifies the replacement process and binding.
Keep deferred restoration assigned to an owner and report it. Failure/cancellation
also follows this lifecycle; absence of an original filler means no new filler.
The existing guard's default host list and fill-all-idle behavior are not this
task's authorization or restoration policy.

## Resume and final validity

Keep attempt-specific routing, worker reports and elapsed costs. On a changed
worker count, completion is based on the globally verified unique scene records,
not the sum of only the new workers' counters. Retain successful predictions and
scores; reclaim only expired or abandoned leases with generation checks. An
infrastructure retry may start a new attempt after the old owned workers exit.
Retain its failure record and classify it explicitly. Do not clear a scientific
error or delete valid outputs to satisfy a completion counter.

Require all of the following before marking the checkpoint evaluation successful:

1. Prediction COMPLETE identity matches the bank and checkpoint/input contract.
2. Exactly 12146 unique expected scene records and payload hashes pass.
3. Each of q0 and q_final has exactly 12146 corresponding scored records.
4. Both summaries have `valid=true` and `failed=0`, and their means match those
   records under the pinned canonical protocol.
5. Queue completion points to those validated score identities.

A COMPLETE filename can also describe a full diagnostic bank containing failures;
it is not evidence of a valid benchmark by itself. Retain its failures and mark
it invalid. Filler restoration has a separate status: score validity must not be
inferred from whether restoration succeeded, or vice versa.

Supervisory adoption of the existing training queue uses Linux pidfd APIs and
the tested `/usr/bin/python3`. The Qwen and canonical conda interpreters on this
installation do not expose those APIs. Do not change their environments to fix
that operator prerequisite; keep supervisor, model and scorer interpreters separate.
