# Draft independent resource-control findings

Scope: external endpoint supervisor only. Runtime751 model/export/official
scoring/analyzer math is unchanged and was not executed by this reviewer.
The reviewed draft SHA256 was
`60b3111bc7b025308489aaae689212530070441cc33c98065c13ff4dc5d6625e`.
This document grants no execution permission.

1. **Post-Popen ownership gap.** `execute.launch` creates a process, then
   performs process inspection and immutable LAUNCH publication before returning
   the job to `monitor`, which appends it to the cleanup list. A publication
   exception leaves the actual child outside that list. Independent real-CPU
   reproduction returned PARTIAL with zero reported invocations while its one
   child remained alive. The review harness cleaned up and waited its own child.

2. **Thread-spawned child omission.** `process_tree` inspects only
   `/proc/PID/task/PID/children`. A child created by a live non-main Python thread
   is exposed under that thread's task. The real-CPU reproduction found only the
   parent, missing its live child. Resource totals and exit proofs therefore
   need all tasks/session membership, including owned descendants after a root
   exits or is reparented. No unrelated process may be signalled.

3. **Unknown partial invocation directory.** Counting only ADMISSION.json files
   misses the failure window between durable directory creation and receipt
   publication. All unit attempt directories must consume the two-attempt
   ceiling. Missing metadata is unknown progress, not permission for a free
   replacement.

4. **Unbounded monitor helper call.** `gpu_monitor` invokes nvidia-smi without
   a subprocess timeout, which can stall all memory/output supervision and
   lease handling. The exact mature helper functions also contain external
   nvidia-smi/findmnt calls; a narrow bounded subprocess wrapper can reuse their
   unchanged source while bounding actual invocation time.

5. **Shared metadata traffic.** Every 0.25 s, a full rglob/stat of all prediction
   files and receipts can create thousands of Lustre metadata operations per
   second. Keep fast RSS/ownership checks separate from a bounded, less frequent
   output-size scan, with final recheck and the sampled-guard delay documented.

Findings were sent to the implementation owner. No runtime source was changed
by this reviewer. Initial reproduction attempt incorrectly omitted DEVNULL and
STDOUT in its subprocess test double, preventing a Popen; its JSON is retained.
The corrected second attempt is the evidence for finding1. Both attempts
confirm finding2; neither used weights, DEV payloads, metric caches or GPU.

Actual endpoint, GPU UUID and lease remain unavailable. A final source-contract
review can be sealed after fixes; concrete runtime binding and fresh resource
approval cannot be represented by these synthetic tests.
