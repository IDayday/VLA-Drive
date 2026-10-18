# Independent Systems review: real DDP worker ownership, v3

PASS only for the exact source and actual metadata binding in REVIEW.json.
This finite SMOKE resource approval does not assert model/replay correctness,
approve another formal run, start a process, or change runtime751/endpoint code.
Lead still needs distinct Correctness PASS and actual fresh launch guards.

## Failure and cost retained

V2 did launch one trainer: ADMISSION and LAUNCH identify timeout573016 and
elastic573017. Its real output lists workers573024–573027 and elastic's explicit
SIGTERM cleanup. The source SID-only membership omitted those setsid workers,
so GPU monitoring rejected real owned workers as unknown. Zero arm run dirs,
native COMPLETE and official ADMITTED verify0optimizer updates and0official
calls. This is **one actual failed trainer invocation**, not zero dispatch.

The prior independent review tested same-session children and catchable-signal
ownership transfer, but did not cover installed elastic workers' setsid topology.
That resource assumption was insufficient. Earlier approvals, failed logs and
all timestamps remain preserved; their same-session tests are not relabelled
as cross-session qualification. The old source EXIT cleanup=true alone does not
prove those omitted worker processes exited. Lead separately verified actual
ps/NVML clear; this reviewer does not claim to have repeated that remote query.

SID-only samples were405,000,192B RSS,16,348,730 rchar and24,576 read_bytes.
They omit workers and are lower bounds, not full-tree peaks or complete costs.
Zero run directories do not prove no parent-checkpoint input was read. Charge
60s conservatively for the whole failed attempt; retain observed IO lower bounds
in the new cumulative monitor. No unmeasured GPU/CPU/input cost is invented.

## Source-bound fix

Clean source e631600f13f3a54f9f7bc43f9989a089314b7f9e, controller SHA
 dcaf2c3e66113b20ac9349b147537f210745664f8fc0a1eed6a67b7ff3166228.
New ROOT is rollout_scale_native_resume_v3; v1/v2 evidence remains untouched.

owned_processes starts with the actual Popen PID/start, follows the PPID closure,
then retains individually verified PID/start identities and proven child sessions.
It reuses the exact stat reader from mature monitor3d01323. A session integer
alone is insufficient: session expansion needs a matching live leader identity
or a currently reverified non-zombie known member. Historical SID/PID reuse cannot
seed an unknown process. Unobserved/unproven processes remain unknown.

The tracker is used consistently for launch, group initialization gate, whole
owned-tree RSS, GPU driver membership and cleanup. Cleanup verifies each PID/start
again before signalling, including retained observed orphans across setsid.
The GPU monitor permits exactly one ownership-only refresh for a snapshot/fork
race. A still-unproven GPU process is rejected; it is never used as an ownership
seed. This is not a trainer or official-score retry. Unknown real jobs remain
protected. Resource scans remain sampled guards, not hard allocation quotas.

AST comparison confirms worker, observer/save/load/hash functions, comparison,
native selection and owned STOP marker logic are unchanged from v2. The actual
751 model/trainer, optimizer/loss/scheduler and inference source were not edited.
This supports the narrow scope; it is not an independent numerical replay result.

## Affected tests only

On the frozen v3 source, the independent stdlib-only harness ran:

- A real root → setsid child leader → leaf tree, all included in ownership/RSS.
- The same observed tree after root/leader exit: retained orphan found and cleaned.
- Historical SID and reused PID/start negative fixtures: foreign state not adopted.

All passed in0.220s wall/0.209s CPU. Six real owned processes across two cases
exited, while two separately created foreign fixture processes survived owned
cleanup and were then cleaned by the harness itself. No known residual remained.
Post-test RSS18,018,304B; child peak18,018,304B. getrusage's inherited pre-exec
high-water2,095,808,512B is disclosed, not claimed as synchronized test RSS.
Torch/model/GPU/scoring were not invoked. Old four-case tests/profile were not rerun.

Tiny synthetic monitor fixtures additionally verify one refresh admits only a
newly proven child, still-foreign remains an error after two monitor calls, and
unrelated GPU-cap errors are not retried. No actual GPU query occurred. The
implementer's eight affected tests are separately attributed, not independent
runs by this reviewer.

## Actual binding and unchanged total experiment

Binding file SHA f4c57bcf93be1d2a5a3712322b87385c725d68e5c698a3dcd6611d87751db46a;
binding identity4be799d83fff2f896a61a9dea65f9a0f88af76580e0e7886f92b012797c65805.
The small-file audit checks all source,prior-attempt,lease,policy,environment,
manifest and resource pins. It verifies exact eight zt6 UUIDs, world4 groups,
CPU0–15/16–31,32 fixed TRAIN scenes, original4096 time axis, same software,
no preparation yet, and lease remaining over3900s. It read206,459B of controls.
No new remote/GPU profile or model/native/cache payload read was performed.

Total trainer cap is now4 = prior failed1 + the original three planned launches.
There is no automatic extra attempt. New training remains A4 and B2+resume2,
**eight total updates,1024 primary official calls, zero retry**. Full accounting:
prior60s + new3540s trainer/controller + comparison120s + ownedcleanup180s =3900s.
The1TiB sampled rchar guard starts with the prior observed IO counts and still
reports lower-bound/mmap/short-lived process limitations.

104GiB future new-output allowance, eight native directories including partials,
12GiB per native, launch1634GiB shared free/runtime1530GiB,640GiB fresh RAM,
256GiB/group and512GiB aggregate RSS,48GiB/card are unchanged. Old tiny metadata
and logs are occupied space already reflected in df, not multiplied into a new
host/namespace budget. No checkpoints are deleted. Group-level heavy loading is
still staggered by actual native0/initialization completion.

Lead's fresh203939Z snapshot reports zt6 all8cards free and shared free
3,422,508,969,984B, with other genuine jobs protected. That is attributed Lead
resource evidence, not a repeated live query by this reviewer. Actual launch
must still pass current GPU/RAM/storage/software/mount/policy/lease guards.
Diagnostic weights remain forbidden as formal initializers. This approval is
resource/lifecycle only; real native resume correctness remains to be measured.
