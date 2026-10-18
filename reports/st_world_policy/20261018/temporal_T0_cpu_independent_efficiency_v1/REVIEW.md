# T0 full-population metadata preparation — independent resource review

**PASS for one CPU-only metadata invocation.** This is neither permission for a
GPU profile nor approval for formal training. The reviewer did not dispatch the
command, run either preparation CLI, or change the live resource ledger.

The reviewed implementation is owner commit
`c48f3faf601d66cd28f033c41273ca15192bebe0`, using clean frozen runtime
`38f86a15a92171e583bb5019cce89a63fb7a1d43`. Actual source, plan, admission and command
hashes are recorded in the adjacent JSON files. The lead must use the exact
command and launch before **2026-10-18 03:14:18 UTC**.

The adapter retains the existing two-CLI supervisor. It prepares an explicit
source migration and then a T0 descriptor over the already fixed full training
population. T0 attaches no causal history. The generated PROFILE configuration
describes four ranks, microbatch 1, global batch 32 and at most four updates; no
such training is executed by this CPU invocation. The command does not call a
trainer or profiler and does not produce teacher tensors or checkpoints.

The independent review exercised the real adapter's pure admission validation
and verified rejection of eleven changes: GPU access, CPU affinity, output cap,
automatic retry, checkpoint allocation, implementation-owner self-review,
output root, and each of the four resource limits. Both original CLI calls and
their arguments were inspected; they were not run by the reviewer. The original
model test suite was not repeated.

The finite limits are CPU 12/13 with two computational threads, 64 GiB logical
read, 4 GiB aggregate RSS, 1,500 seconds plus up to 40 seconds cleanup, and 8 MiB
output including controls. The shared control lock was available. A three-second
sample saw CPU 12/13 busy fractions of 1.33%/1.66%; these are shared CPUs, not an
exclusive lease. History's CPU 10/11 and existing GPU workloads are untouched.

The original full-population metadata preparation actually completed in
178.623 seconds, charged 9,664,517,081 bytes of logical reads, and reached
707,788,800 bytes aggregate RSS. This is inherited sizing evidence, not a
measurement of the new T0 invocation. Its source receipt is pinned in REVIEW.json.

The new 8 MiB is an additional administrative allocation: 10,098 → 10,106 MiB
within the 10,240 MiB envelope, leaving 134 MiB. The native disk guard was called
read-only with actual current typed claims, including the active history build.
The stronger reservation still retains all 768 GiB potential cache, 10 GiB
metadata, the 1 TiB free floor and 3,701,166,716,312 bytes future checkpoint
commitment. At review, 5,883,994,607,616 bytes free exceeded the resulting
5,636,049,483,160-byte requirement by 247,945,124,456 bytes. No additional cache
credit, deletion, quota assumption or four-profile allocation was used.

The output parent exists and the metadata root was absent at admission. The
one-shot supervisor must create it exclusively, retain failures and refuse
automatic retry. Completion, actual read cost and cleanup remain to be measured
by this invocation. Future T0 GPU capacity/timing and T1/T3/T4 preparation require
their own actual inputs and appropriate resource decisions.
