# Independent endpoint-supervisor resource contract review

**PASS for the clean source contract; WAITING_CHECKPOINT_AND_FRESH_BINDING for
execution.** Final controller commit `82e8f9dcba7d3323ca15491fdfe7f6e5481922ff`,
file SHA256 `a2e24537b9749258417e390be705d4d5d4e6f775986ca479a1c67c4330759a3e`.
The inference/scoring/analysis runtime remains clean
`751fcfa2b41549ae16f7b3204aef8dd18b912016`. No runtime or running job was
modified by this review. This reviewer did not implement the controller and
did not dispatch inference, official scoring, GPU work or training.

## Exact resource and experiment boundary

The supervisor is a thin wrapper over sealed commands and SHA-pinned mature
v5 helpers. It retains one fixed4096 endpoint, DEV1696/16logs, four disjoint
424-scene batch1 exporters, seed42, one trajectory, Euler10, FP32-master eval
and original official reference protocol. Historical C1 is reused without new
control inference/scoring. No world-size, precision or population fallback exists.

The accepted topology is one future zt6 route with GPU0–3 and real, distinct
UUIDs. Initial export admission requires512GiB available RAM, fixed CPU0–7,
two Torch threads/rank and BLAS1. Exporter session RSS is capped96GiB each,
384GiB together and driver memory32GiB/GPU. At most two heavy model loaders
are admitted; a slot is released only by an owned current invocation's complete
strict FP32 restoration audit, matching rank, endpoint/source/manifest identity
and time, or actual process exit. Four ranks reconstruct/check the full model
separately; this is not a measured4x speedup or a new model-loading peak claim.

Outer timeout starts before native checksum/model reconstruction:
export7800+60s per rank, score4200+60s and analysis1800+60s. Each unit may have
at most two explicitly admitted invocations, counting partial directories,
unknown publication windows and failed launches. No automatic retry or phase
chaining exists. The same output and stable binding are retained for resume.

All four completed shards must pass the mature population/identity verifier
and every owned exporter session must be confirmed exited before CPU scoring.
Scoring is one coordinator plus four single-thread workers onCPU8–11 with
64GiB aggregate RSS. The original durable global ledger admits at most1728
calls across workers, processes and restarts for1696 primary scenes. Analysis
usesCPU8–9,8GiB RSS, no new score calls and the sealed2000-replicate protocol.
Other genuine jobs may occupy released GPUs during CPU phases; they are not
required to leave them idle and are never signalled by this controller.

Only current known policy/campaign authorization paths are accepted. Fresh
storage observations are at most300s old and must preserve1TiB shared floor,
other future write commitments and this evaluation's1GiB output allowance.
Dynamic policy/authorization/storage observations can refresh without changing
the immutable checkpoint, scene, source, route/lease, controller or output
identity. Each refresh is retained in its own resource admission receipt.

## Findings and fixes actually verified

The initial draft had real lifecycle faults. Independent real-CPU reproductions
showed a live thread-created child omitted by the tree walker, and a Popen
followed by LAUNCH-publication failure leaving an unregistered live child.
Both failed draft records are preserved. The corrected controller:

* registers the child immediately after Popen, before fallible close/stat/receipt
  work; log-open failures are separately retained as known not started;
* discovers all task children and tracks only the fresh session it created,
  covering reparented descendants, root exit and separate process groups;
* stops only those owned session groups and retains ownership/start stamps,
  invocation locks and exit/cleanup receipts;
* charges all partial invocation directories against the two-attempt ceiling;
* bounds nvidia-smi, findmnt and other mature helper commands to15s;
* checks RSS/ownership frequently, GPU queries every5s and output tree every10s
  plus final recheck, avoiding a Lustre tree scan on every0.25s loop;
* uses the actual current policy path rather than accepting an old allowlist
  copy, while allowing a fresh legitimate storage receipt on later phases.

These are process/sampled guards, not instantaneous filesystem/GPU quotas.
A query may take up to its declared timeout, and output growth can occur
between10s scans. Full-tree metadata scanning is intentionally bounded.
`waited_child_CPU_seconds` measures waited-child CPU; forced reparent/cleanup
can make it a lower bound for unwaited orphan CPU. The implementation documents
that limitation instead of claiming exact abnormal-process attribution.

## Single final independent CPU window

Against the initially sealed380c8f54 source, the controller suite passed **47 tests**,
3.69s pytest /4.937s subprocess wall. Total first review window5.045s,
4.774022 waited-child CPU seconds and sampled tree peak642,142,208B.
This includes exact loading slots/audits, driver-memory limit, all-stage
invocation accounting, stale/foreign admission refusal, terminal native
metadata, policy/lease/RAM/CPU/storage checks, owned-root/thread/orphan cleanup,
LAUNCH-write/log-close fault windows, preserved global score arguments and
CPU-stage release of owned exporters without evicting new GPU users.

The additional independent reproduction initially omitted a newly required
synthetic manifest field, so it stopped before Popen. This harness failure is
retained in FINAL_TEST_1_OUTPUT and the initial FAIL receipt; no runtime change
was needed. I corrected only that explicit fixture and reran the ownership
reproduction, **without rerunning the47 passing tests**:0.116s wall,
0.077073 waited-child CPU seconds. It confirmed both actual CPU descendants
were discovered, the deliberately failed LAUNCH retained one charged invocation
and left no live child, and two empty partial attempts were refused.

Those initial final checks used4.851095 waited-child CPU seconds and5.161s wall.

Lead then identified a catchable signal window not covered by those47 tests.
I reproduced it at the same380c8f54 source: a real child was created inside
Popen and self SIGTERM was delivered before the handle returned. The handler
raised before jobs registration, leaving the charged child alive; the harness
cleaned and waited only its own session. This actual failure remains in
SIGTERM_WINDOW_REPRODUCTION.json. The provisional source review was held,
not released as executable approval; its earlier contents are preserved.

Owner fixed only `execute` and `monitor` in82e8f9dc: the handler records the
first stop request, and monitor checks it before new dispatch and after child
ownership transfer. No signals are blocked/inherited by children, and no model,
command, budget or other runtime function changed. The original47-test result
is inherited for that unchanged scope; I did **not** rerun the full suite.
Independent final affected regression: **8 passed /41 deselected**,0.26s
pytest /0.690s wall /0.570645 waited-child CPU seconds, sampled tree69,279,744B.
It covers normal LAUNCH/log faults plus thread/root-exit ownership. My separate
real-CPU Popen-return reproductions for **both SIGTERM and SIGINT** now each
retain one charged/recorded invocation, leave no live child and confirm cleanup:
0.132s combined wall /0.104134 waited-child CPU seconds. No SIGKILL guarantee
is claimed. The measured bounded windows total5.525874 CPU seconds /5.983s
wall, within the120CPU-second /600s /2GiB /32MiB review budget. No weights, images,
DEV result payloads, metric cache, model execution, official score or GPU context
were used. Fixtures do not establish actual deployment equivalence or throughput.

## Remaining admission

The actual complete4096 checkpoint, UUIDs, resource lease and stable binding
digest remain null. REVIEW.json deliberately uses a **source-contract schema**,
not the executable admission-review schema; it cannot accidentally authorize
an unknown checkpoint/route. Lead must bind the real native receipt, sealed
analysis and fresh resources and obtain concrete independent binding reviews
before invoking the supervisor. No new GPU profile or unrelated model audit is
required merely because these endpoint metadata are not yet produced.

This PASS concerns resource controls only. It is not a mathematics approval,
driving result, formal evaluation completion, feature-side restart, new navtest
permission or proof of actual four-rank speed/RAM/IO performance.
