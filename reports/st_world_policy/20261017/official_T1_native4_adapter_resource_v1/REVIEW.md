# T1 native4 adapter: independent resource boundary review

**BLOCKED on two concrete issues** in `cd55876bc2214ac7c8e0bc97405979a3f07c5548`, adapter SHA `5da49c14d5552011a755ce981669b49f4d048242a9019667d77468eeefff5aca`. This inherits the prior resource feasibility review `490b288903b402b1499669b2ade7dc3fa15842ca`; it does not repeat model or scientific review. No registration, lease, resource reservation or real training was created.

## T1-R01: exception path restarts cleanup

`monitor()` raises at `term_at+180`, then its `finally` calls `owner.finish(..., deadline=time.monotonic()+180)`. The independent deterministic CPU fixture uses no child/model/GPU and reproduces first TERM at 3601, original absolute deadline 3781, but final cleanup deadline **3961**. The owner 18 tests pass; this additional original assertion fails. The terminal exception path can exceed the declared invocation 3600 + cleanup 180 window and its lease.

Required narrow fix: keep one absolute cleanup deadline after the first termination request and use it in normal, exception and finally cleanup. Unknown setup failures may establish a deadline once, but may not extend an already established deadline. Retain owned-session pidfd checks and no signals to foreign jobs.

## T1-R02: no live shared-space obligation check

The adapter limits files within its new `runs` namespace to 120 GiB, including checkpoint partial files. It does not validate a checkpoint commitment receipt or check shared free space against other runs' obligations. The resource-review schema only asserts a new120GiB reservation; a historical report snapshot does not protect the current shared disk. Lead confirmed that no external actual CP control/launch exists yet.

Required narrow fix: bind a genuine immutable combined commitment control supplied by Lead: preserved2183GiB + separate new120GiB =2303GiB, plus unchanged1TiB free floor. Validate hash, exact amount/root/window and live shared `statvfs` at admission and during the bounded write window. The control must cover the entire invocation/save/cleanup window and cannot reclassify earlier obligations as consumed merely to reduce it. This does not create or alter a cache ledger claim. Actual quota remains NOT_VERIFIED unless separately established.

## Preserved and acceptable boundaries

- Two possible invocations only: fresh phase2, then actual native-parent phase4. Stop endpoints2/4 and16 total scene exposures; no third reload or retry.
- Phase4 requires phase2 COMPLETE/native receipt/zero survivors and carries its observed cost into the same512GiB logical-read budget. Unknown failures conservatively consume remaining scope and preserve one-shot markers.
- CPUs2–5; world2; live process-tree RSS/rchar are sampled. Live reap races can conservatively over-count; final post-wait accounting uses the already reviewed8674 function. `/proc/rchar` is not a physical Lustre bandwidth metric.
- Exact e85 OwnedSession helper is `d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902`, explicitly different from the older helper constant inside8674. Only the pinned accounting/pipe functions are reused from8674; its obsolete admission/helper-loader path is not called. This mismatch is disclosed, not bypassed.
- New checkpoint120GiB is independent of the old PROFILE80GiB. Counts include native partial files; metadata/logical-file size monitoring is sampled rather than a filesystem hard quota.
- Live GPU UUID/idleness and cooperative GPU locks are checked by the original helper at actual admission. No GPU inventory or hardware claim was fabricated in this code review.

## Evidence and next review

CUDA hidden; pytest on CPUs0/1. `18 passed, 1 failed` in1.95s; the failure is the deterministic T1-R01 counterexample. Existing tiny children cleanly exited or received owned timeout cleanup; existing native comparator used only tiny CPU tensors. No real checkpoint/image/teacher/score payload was read. No production source was edited.

Fix only the two identified boundaries in a new owner commit. Independent follow-up should rerun the unchanged deadline assertion plus focused commitment/hash/expiry/free-space negatives. Do not rerun scientific qualification or issue actual resource admission before that follow-up and real Lead-created controls.
