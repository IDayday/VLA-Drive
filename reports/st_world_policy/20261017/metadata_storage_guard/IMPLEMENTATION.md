# Metadata storage guard implementation handoff

2026-10-17. Owner `/root/st_metadata_storage_guard`; isolated start
`db615170e4ade6227056572d64d1f690677fad6a`.

**IMPLEMENTED, OWNER CPU TESTS PASS; independent Correctness pending. No actual
policy/ledger migration, cache preparation/index build, image/tensor access or
GPU workload was executed. This report does not approve a formal launch.**

The change is confined to the original storage ledger API, one metadata-only
helper and CPU fixtures. Teacher numerics, Base/runtime, data selection, index
implementation and future-preparation implementation are untouched.

## Same ledger and explicit additive migration

`StorageBudget.migrate_metadata_policy` calls `metadata_storage.migrate_policy`.
It acquires the existing `CLAIMS.lock`; no new global ledger/lock is introduced.
Old/new policy, old ledger, pilot inventory, Lead quiescence and checkpoint
commitments require exact raw SHA256 pins. Hash and JSON parsing use the same
bounded bytes. The migration preserves the old campaign/state directory,
cap, free-space floor, ledger allowance, reservation ordering, claims, charges,
observations, devices and other historical ledger fields.

The only allowed core extensions are exact inventoried pilot roots, their
MiB-rounded immutable legacy reservations and one metadata package. The package
contains the original declaration/reservation baseline and has a maximum of
4 GiB. Every new declared metadata root counts even before first admission.
Retained pilots, output, journals, staging, receipts and failures all share that
one allowance. Existing v4 remains byte-identical and safely rejects writes
against the migrated core. Unmodified cache readers are not affected.

Migration order is pending transition → immutable old raw snapshot → atomic
same-ledger state → immutable completion receipt. A retry can complete only the
same transition, retains all old accounting and rechecks pilot inventory. A
conflicting/stale transition fails. No missing ledger can be bootstrapped with
a metadata policy. No process is signaled or a quiescence result fabricated:
Lead must provide fresh complete process/controller evidence.

Migration files remain inside the existing ledger metadata allowance, including
interrupted atomic files. They are not another four-GiB package. Subsequent
claim additions must remain within the original metadata package baseline and
must bind actual immutable registrations.

## Writer guard and intentionally small runner

`metadata_writing(...)` admits a typed build-registration SHA, never a fabricated
final index identity. It precharges the **complete** bounded attempt before root
creation or metadata writes and holds the existing flock through the finite
invocation. `METADATA_ADMISSION.json` is distinct from teacher `identity.json`.
A resumed root must carry its own admission; its files and failed partials stay
charged. A previously empty root can contain only explicitly pinned small
pre-registration files (e.g. real PLAN.json and empty PLAN.lock). Unknown files,
links, hardlinks, mount escapes, excess depth/count and unregistered names fail.

`guard.check(before_commit=True, path=destination)` validates canonical parents
before IO, allocated and logical bytes (including directories), current process
**tree** RSS, elapsed deadline, separate free floor and current remaining
checkpoint commitments. It uses the already-held state, never nested flock.
Failures are sticky. The resource check must run at least every five seconds;
late callbacks fail before publication. Long reads and native unpickle calls
must remain bounded by the writer and Lead's independent finite process-tree
monitor/timeout. This cooperative hook is **not** a cgroup and does not claim
that Python can asynchronously interrupt a blocked native call.

`check_active_metadata_storage(...)` exposes the same check to bounded worker
threads through a process-local guard and RLock. Nested writers fail before
flock. The thin `run` command imports only the two real metadata CLI modules,
uses their existing parsers, and invokes `index.run` or `future.build` in the
same process. It adds no RPC, scheduler, subprocess or host dispatch. A writer
that never invokes the resource/publication hooks fails; the old unwired CLI
is not approved by this implementation. Owner/Lead must wire hooks before using
it for a real build. The final returned result is still subject to independent
artifact verification.

## Required owner hooks

In `prepare_future_metadata.PreparationBudget` / `write_chunks` and index long
hash/parse/SQL/write loops, call:

```python
from tools.st_world_policy.metadata_storage import check_active_metadata_storage
check_active_metadata_storage()  # at most five seconds between bounded chunks
check_active_metadata_storage(before_commit=True, path=destination)
```

Call before every creation/atomic publication, including temporary paths,
shards, final index/request and COMPLETE. Register all exact allowed names and
controlled directory patterns. Do not call another `StorageBudget.reserve`
inside this context. The independent monitor must stop only its own finite
process tree on a resource violation; it must not stop ongoing training.

## CPU evidence and remaining gates

Actual command, Python environment and full output are in `TEST_OUTPUT.txt`.
75 passed, one existing pynvml deprecation warning, 5.12 seconds. Tests include
all existing storage-budget regressions plus 48 metadata cases: migration race,
four crash windows, stale/incomplete quiescence, changed pilots/pins, cap/floor
and history preservation, aggregate package oversubscription, exact PLAN/lock,
partial resume, flat/nested symlink/hardlink/unknown-file rejection, process-tree
RSS/deadline/check interval/free/checkpoint limits, thread callback and nested
flock protection, and rejection of an unwired thin runner. Temporary fixtures
are explicitly not real resource evidence.

Not executed: actual six-pilot adoption; live writer/controller quiescence;
current-only index build; future qualification; full metadata; teacher/student
numerical work; GPU; formal launch. These remain Lead-gated after independent
review and owner callback integration.
