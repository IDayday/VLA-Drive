# Single formal rollout-scale run: systems draft

**NOT_APPROVED_WAITING_PROFILE240.** This draft binds the already prepared `C1_rollout_std_fulltrain_seed42_official_4096_v1` registration (`928b9788404c4dcb9316b8457dc1a627010f892b85308f52aacf07b2cbb0783f`) and runtime source `751fcfa2b41549ae16f7b3204aef8dd18b912016`. It approves neither training nor evaluation. The running 240-update PROFILE remains the only approved next execution. Its complete two windows, resource totals and final artifacts must be reviewed before an exact formal efficiency PASS.

## Execution to reuse

Reuse `/mnt/project/st-world-policy-20261016/resources/launch_reviewed_flow_v1.py` (SHA `6a03442b26eb01a5de9e2ea5e3e1f508ca69d2959ba18cf37903e73295ce3db2`) through a separately pinned narrow version. Keep the reviewed 8-hour invocation, 600-second TERM grace plus 180-second KILL backstop, native safe-boundary saving, explicit same-identity resume, six-invocation ceiling and fifteen-native-directory budget. Preserve the original launcher for old results. The final adapted launcher path/hash is still pending; this draft does not pretend the old hardcoded runtime can launch the new source.

The one candidate starts from the original S0, never the old C1 endpoint or diagnostic weights. Use zt5 GPUs 0–3 only after PROFILE exits and fresh allocation checks pass, CPU 0–15, world4 / micro4 / caller accumulation2 / global32. Retain 4096 scheduler updates, warmup100, G4, MC50/chunk5, Euler10, original repeat8 BC and all registered loss/optimizer settings. Global CPU scorers remain four, loader workers two per rank, BLAS/OMP/NUMEXPR one. The full run has 131,064 scene presentations and 524,256 primary scores; the fixed admission cap is 532,448.

A new 72-hour lease bounds when the same experiment can be invoked; it does not grant 72 hours of extra optimization. At most six 8-hour invocations provide 48 hours of trainer time and never more than 4096 completed updates. Formal speed and expected invocation count remain unfilled until the complete PROFILE. The four-step SMOKE is not used as a sustained-speed claim.

## Storage accounting

The old launcher required 3,130 GiB free: 1,024 shared floor + 123 cache headroom + four 240-GiB spatial future plans + 187 C0 + four 209-GiB C plans. That was a multi-wave admission, not the correct remaining-write requirement for this single new run.

Small immutable RESULT records show all C0–C4 completed4096. The spatial 5k runs and PCA control are stopped at their budget endpoints; SP2 extension is paused at11762 and SP3 at12071. Their stored models remain untouched. Feature development is currently paused by the user, so these plans are not automatically restarted or funded as current future writes. These are artifact-status findings, not a claim that a historical PID is a live or dead process; final dispatch must inspect actual processes.

The three large training-cache COMPLETE receipts are all present: native384, pool3 and legacy_c1, each covering101592 scenes. Together they declare 629,345,297,104 bytes of retained cache payload. The old plan's `CACHE_NOT_STARTED` snapshot is historical; those later COMPLETE receipts resolve its state. Their occupied bytes already reduce current filesystem free space. They must not be subtracted again as if about to be written. The ledger and cache cap remain unchanged, and no cache/checkpoint is deleted.

The new run reserves **209 GiB**:180GiB for at most15 native directories (12GiB each),24GiB for feedback/rollout records and5GiB for control/log/observer records. The actual recent SMOKE mature native payload is 11,473,306,504 bytes; fifteen mature payloads are 160.28GiB, within the native reservation. This is a capacity estimate, not an observed formal output total. Partial/failed native directories count and are retained.

Four old guarded metadata/probe claims lack a canonical completion at the checked path. Conservatively retain only their unmaterialized declared headroom, 3,493,398,928 bytes, rounded to4GiB; do not add their already observed bytes again. No new work is authorized for those claims. The provisional fresh-free threshold is therefore **1,237 GiB =1TiB floor +209GiB new run +4GiB unresolved headroom**, plus any other currently authorized active write claims discovered at launch. The local shared-filesystem snapshot was 3303.92GiB free. It is a point-in-time filesystem reading, not a quota guarantee or an assertion that other jobs cannot consume space.

Do not overlap the formal run with the selected-GPU PROFILE. Once PROFILE completes, its retained files also count only as occupied space, with no additional future-write reservation. If any other writer remains active, add its remaining bounded writes before admission.

## Narrow changes and admission

The adapted launcher must pin the new runtime/source and this one new registration; replace only the stale multi-wave storage guard with the above immutable remaining-write account; and carry over CPU0–15 plus the already tested selected-UUID compute-process check, including0MiB/0util startup processes. Preserve fresh RAM/maps/free checks, UUID/lease checks, dispatch/perGPU locks, exact independent reviews, duplicate-run rejection, safe native resume and explicit stop acknowledgement. Pin the bounded CPU observer before launching; do not expand its scope into a new monitoring platform.

Before each continuation, resolve the latest complete compatible native checkpoint with checksums. Recompute remaining periodic512 states, account for already saved and partial directories, and reserve the bounded pause/partial allowance. Refuse if15 directories or six invocations would be exceeded. Preserve Adam/master, RNG, consumed sampler and the original4096 scheduler clock. Do not delete evidence, reset progress, restart warmup, choose diagnostic weights, or auto-acknowledge an operator stop.

Fourteen metadata/arithmetic checks pass in this draft. No model, large manifest, raw training ledger or DEV scene scores were read. No host was remotely dispatched, no new job was started, and no running source or ledger was changed. Final measured efficiency, the adapted launcher, current resource claims and exact independent review fingerprint remain required.
