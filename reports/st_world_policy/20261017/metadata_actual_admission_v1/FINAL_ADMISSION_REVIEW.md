# Final finite metadata admission

**PASS_BOUNDED_METADATA_ADMISSION** for the exact v6 preparation and controller
in `RESOURCE_APPROVAL.json`, until the receipt's explicit expiry. This permits
only the finite CPU metadata sequence, not teacher extraction or training.

The final package supersedes the prepared v5 review. v5 files remain untouched;
neither v5 nor v6 was migrated or built by this reviewer. The original v4 ledger
was still unchanged at the exact-package check. The runtime stays clean at
`1162b955b3bf9abb6728823a84a50c38e1a19bb8`.

- PREPARED_V6 SHA: `1bd1b7cc4045ad5febf6ac831c5013263ce1af7c1cbda450117cdbea959e1f58`.
- Policy v6 SHA: `7adf268a2c17437f011728dbd041f76c8caa99286d815cd0d36974d12034cf1c`.
- Controller SHA: `a222cc453d236db5ffde0e21f42936b40f7f30108a3648cad23050ffe5df18d7`.
- Checkpoint receipt SHA: `1bf12212b29202cd3f5cf80e73425ab5d663c778e7ba35df43a83bb2c27289ad`.
- Quiescence V2 SHA: `46544ce8368d916a9d53b892fbdc380b174f771fbf67222f23a56bd1a15494e0`.

## Narrow controller finding and verification

Source inspection found the original one-shot exists check occurred outside
the storage lock; unconditional START publication could overwrite a competing
controller's marker and admit a second child. Lead moved the second check inside
the existing `metadata_writing` global lock, covering check and atomic
publication. Both START and RESULT are now immutable.

The independent CPU test injects a competing marker exactly at lock acquisition:
the current source rejects before Popen and preserves it. A source-only negative
mutation removing this check reproduces overwrite and duplicate launch. No
production source was modified by the reviewer. Five further bounded Python
child fixtures check success, nonzero exit, RSS limit, deadline and pipe limit.
All failures terminate only their own fixture child, retain FAILED output and
charge the full4 GiB unknown read budget. **7 passed in0.18s**, full log retained.
No real worker, ledger or cache was used in those fixtures.

Lead also corrected atomic-temp filename patterns from hex-only to the actual
`[a-z0-9_]` mkstemp alphabet using new V2 registrations/v6 policy. The resource
review verifies both final patterns accept named CONTROL/COMPLETE partials.
Original preparation files are retained; budget and source/scene identities
did not change.

## Exact live evidence and bounded scope

The pinned eight-host observation at05:46:51 UTC reports reachable allowed
hosts, no unreadable/oversize process records and only four known read-only
training observers. Their script identity is pinned. Lead explicitly owns and
disables future cache admissions/queues; the independent review does not infer
queue shutdown solely from absence in a process list. The renewed V2 receipt
binds the v6 policy without resetting its original observation time.

The checkpoint receipt retains the full1983 GiB original nine-run commitments.
Available filesystem bytes still exceed that allowance plus the entire768 GiB
cache envelope plus the1 TiB free floor. No existing allocation is subtracted
to make this admission fit. Quota is explicitly unverified, not silently claimed.

Run **serially**: future32 build, successful-build reference, then current index.
At most one invocation per stage, no automatic retry. Each child uses CPU0–1,
empty CUDA visibility and a new owned process group. Parent+child RSS is checked
every0.25s against4 GiB. Each output pipe is limited to1 MiB. The finite deadline
includes controller import/startup; five seconds are reserved for local cleanup.
The failure path signals only the new owned child group. The exact current
builders use threads only; generic daemonized descendant monitoring is outside
this admission.

Current index: ≤1200s, total1 GiB including staging/control. Future build/reference:
≤450s each, ≤4 GiB reads each, ≤900s/8 GiB combined,32 fixed scenes, worker1,
all persistent output/control/partials within the same32 MiB root. Seven old
pilots are reserved16 MiB. Added package1072 MiB is within the single4 GiB package
and the unchanged768 GiB campaign cap. The storage guard keeps all precharges,
callback/free-floor/deadline checks and checkpoint commitments; outer monitoring
does not replace those checks.

No full future metadata run, teacher tensor creation, GPU use, formal temporal
training, active-model runtime change, file deletion or existing-job signal is
authorized. The machine-readable approval is an admission decision, not a claim
that the actual build succeeded. Lead must record actual execution separately.
