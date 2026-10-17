# Owned leader exit snapshot race: narrow fix

Source baseline: `5b956b05df21e0b3cf194b5d0ac7f6218a596de3`.
Isolated worktree: `/mnt/project/VLA-Drive-st-metadata-exit-race-20261017`.
Changed controller SHA256: `8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644`.

The first `discover()` can observe the still-live owned leader. It can exit before
`leader_done()` returns true through `waitid(WNOWAIT)`. Treating the earlier
snapshot as surviving descendants therefore falsely rejects an ordinary exit.
The fix re-runs the existing identity-checked discovery only in the
`done and active` case. A verified live descendant still fails with
`OWNED_DESCENDANTS_AFTER_LEADER_EXIT`; a changed spawn identity still raises and
uses the existing owned cleanup. Nothing treats process identity errors as exit.

The pidfd/session helper, immutable runtime, timeout, I/O limits, once-only
cleanup and pre/post-wait accounting are unchanged. In particular, terminal
parent `/proc/io` already contains the waited child and is not charged twice.
No active TRAIN source, old START/RESULT, index dependency pin or live policy was
edited. The original index combined regression (113 passed / 1 failed) remains
in `full_future_index_execution_v1/COMBINED_REGRESSION_INITIAL_OUTPUT.txt` on
commit `15967d1de0615ed104a2c326a09b4ac7dd63d2ea`. This is a source correction,
not a reclassification of that failed run.

## Current bounded CPU evidence

`EXIT_RACE_TEST_OUTPUT.txt`: **71 passed**, 4.27 seconds pytest time.
`EXIT_RACE_TEST_RESOURCES.json`: 4.818 seconds wrapper elapsed, 67,469,312 bytes
peak RSS, CPU affinity 0/1, 2 GiB address-space limit, 180-second wall limit.
Existing accounting fixtures write 12 MiB total payload; new race fixtures
write only tiny temporary fd files and pipe JSON. No images, model weights,
teacher tensors, real builder, GPU or live ledger are used.

New deterministic cases inject the exact `discover -> done` interleaving for
both return codes 0 and 3, then exercise an empty refreshed snapshot, a real
owned-descendant snapshot, and an identity-change exception. All require
exactly one `finish` call. Four additional real tiny CPU process cases cover
exit 0/3 with and without a forked live descendant; surviving descendants are
rejected and reaped, and each resulting cleanup has zero live owned processes.
The original terminal-accounting, pipe/read/RSS/wall, malformed receipt and
metadata preparation tests also pass.

This patch awaits independent review and explicit downstream source rebind.
It does not dispatch or authorize any metadata indexing/build job.
