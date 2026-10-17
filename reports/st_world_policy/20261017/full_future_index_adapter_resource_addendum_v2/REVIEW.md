# Exact corrected-pair resource addendum

**PASS_EXACT_FIXED_PAIR_RESOURCE_BOUNDARY** for commit `ea1b00bb3f00d3a96633325f04d8d08bc817b56f`. No actual index admission is issued.

Reviewed pair:

- Adapter: `eb9713666a84e8ffba7a55c75f656bb6133218dc626d772200189adad6d41a9a`.
- Shared monitor: `8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644`.

The adapter explicitly pins the new monitor. An AST comparison with the prior reviewed source verifies all adapter functions except `read_small_actual` are unchanged; all core functions except `monitor` are unchanged. The reviewed diff contains only the pin, bounded reader and post-exit ownership refresh. Budgets, argv, roles, same-ledger logic, old claims, source/runtime inputs, checkpoint floor and one-shot semantics retain the prior review `c6c41f25`.

The monitor refreshes owned-process identities after the leader is confirmed exited. A stale live-leader snapshot therefore no longer masquerades as a live descendant; a verified real descendant still produces failure and owned cleanup. Deterministic empty/descendant/identity-error cases and four real tiny-child exit0/exit3 cases passed. No unknown or preexisting process was signalled.

The COMPLETE reader now opens a nonblocking, no-follow descriptor, validates the opened identity, reads at most1MiB+1 in total, and rechecks fd/path identity and symlink ancestors before decoding/hashing. Growth cannot trigger an unbounded `read_bytes`; changed FIFO/symlink/file identities reject before payload reading. The extra one byte is the oversize sentinel, not an increased accepted file cap. Descriptor closure after a growth error is verified. This does not weaken existing source/hash checks.

## Actual bounded tests and preserved initial failures

**43 tests passed**, no failures/skips:12 owner bounded-reader tests,10 owner exit-race tests,21 independent resource/pin/fd/shared-filesystem tests. Final execution took1.14seconds, peak RSS38,617,088bytes, process read delta8,942,075bytes, CPUs0–1,2GiB address-space cap. No torch, GPU, actual index, live ledger write or dispatch.

The initial project-filesystem run remains **40PASS/2FAIL** in `INITIAL_TEST_OUTPUT.txt`; it is not rewritten as a pass:

1. One inherited independent test still intercepted `Path.read_bytes`, which the new implementation deliberately no longer calls. Its fault injection was updated to the actual `os.read` boundary. No production code changed.
2. The owner's timestamp test requested +1,000,000ns on the shared filesystem. `fstat` and path `stat` remained at1792232894000000000ns, so the requested fault was not observable. `SHARED_FS_TIMESTAMP_PROBE.json` records st_dev3789794614 and full before/requested/after values. A +2,000,000,000ns request changed the observed mtime to1792232896000000000ns and was rejected by the unchanged production check. The original +1ms owner test also passed unchanged on `/tmp`.

This distinguishes an ineffective shared-filesystem timestamp-only injection from a reader protection failure. It does not claim that version checks can detect invisible same-size in-place edits within filesystem timestamp resolution. Immutable producer receipts, expected schema/identity bindings and existing no-rewrite assumptions remain required. Growth, inode replacement and actually observed version changes remain independently covered. No production threshold or checksum rule was relaxed.

## Inherited boundary and remaining actual dependencies

The two code blockers identified for the old pair are resolved for this exact pair. The earlier113PASS/1FAIL combined evidence remains historical; it is not erased. We did not repeat unrelated model or index correctness suites.

Dev still has4GiB reads/900s+40s/128MiB inclusive root; train64GiB reads/3600s+40s/2GiB root. Both retain4GiB RSS and CPUs0–1. The5620MiB planned metadata allocation,8GiB envelope,768GiB common cap,2183GiB checkpoint obligation and separate1TiB free floor are unchanged.

Actual operation remains pending: genuine train completion with independently reviewed terminal accounting; explicit1MiB administrative allocation for the new index-preparation root (5621MiB total if granted); actual prepared registrations/policy and current ledger/source pins; fresh per-role lease/checkpoint commitments. This addendum neither allocates that space nor refreshes an expired receipt. No active TRAIN state, budget, controller or historical charge was modified.
