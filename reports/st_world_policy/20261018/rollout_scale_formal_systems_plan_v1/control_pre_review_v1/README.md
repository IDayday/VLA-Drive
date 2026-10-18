# Formal control pre-review

Status: **PASS_CONTROL_CHECKS_PENDING_PROFILE240_NOT_LAUNCH_APPROVAL**.
This review pins the actual external launcher, monitor, CPU observer, storage
reservation and formal registration in `REVIEW.json`. It does not sign the
formal EFFICIENCY gate or launch a task.

Ten CPU synthetic tests passed in 0.033 s. They execute unchanged pure-function
ASTs from the pinned launcher and the actual monitor stop-file helpers, avoiding
launcher main, model imports, GPU discovery and dispatch. Negative cases cover
wrong registration, inadequate free space, negative third-party commitments,
partial checkpoint counts, changed UUIDs and a selected GPU with a compute PID
even when memory/utilization are zero. An unrelated GPU process is left alone.
Existing operator stop reasons are preserved; absent/foreign runs are untouched.

The actual controls preserve the reviewed 4096-update recipe, four ranks,
CPU affinity 0–15, native-state resume, six-invocation and fifteen-native-directory
ceilings. Admission requires 1529 GiB free. Runtime monitoring protects 1320 GiB
for the shared floor, old unresolved commitments and protected IL future writes.
The latter job is neither stopped nor constrained by this launcher.

Recursive run/registration size accounting occurs initially, every 300 s, and
at completion. This avoids an O(number of accumulated feedback files) scan every
five seconds. GPU process ownership, RSS, free space and native directory counts
remain sampled every five seconds. The 209 GiB output guard is sampled, not an
instantaneous quota: detection may lag up to 300 s plus scan time, and an in-flight
native save / boundary save may finish. The native count budget, realistic 12 GiB
per native allowance and large free-space reserve remain required. No result may
claim hard byte-accurate enforcement from these sampled measurements.

The monitor sends no signals and writes only the current run's existing
`STOP_REQUESTED` mechanism. Its outer timeout owns the spawned process group and
has a 600 s TERM grace after the 8 h invocation plus 180 s final cleanup. Native
resume is explicit, completed states are refused before GPU dispatch, and partial
invocation directories consume the six-attempt budget.

Pending: actual complete 240-update PROFILE results and both fixed steady
windows, immutable final exact controls, independent scientific/correctness
reviews, then fresh UUID/compute-app/CPU/RAM/storage/lease admission. The historical
zt4 timings are reference only, not a pure causal normalizer comparison.
