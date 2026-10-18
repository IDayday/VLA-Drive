# Cumulative logical-read budget adjustment

The early controller draft had a512GiB sampled rchar guard. The implementer
stat-checked the original S0 native initialization payload at approximately
37.60GB. Each of the three world4 invocations reconstructs that original
initializer before its engine initialization/resume load. Twelve rank loads
plus rank0 binding/checksum reads already imply more than560GB of logical
input reads, before framework loading, native2 restore and observer controls.
No original runtime751 reconstruction path is changed to reduce this cost.

Lead explicitly accepted a **1TiB cumulative sampled rchar guard** for this
whole SMOKE: initial A4 and B2, plus B resume2, including their observers.
The counter must preserve per-process start identity and sampled monotonic
high-water marks, and inherit the completed initial phase's totals on resume.
read_bytes and rchar are reported separately. mmap reads and short-lived
processes which exit between samples can be missed; these are observation
lower bounds, not physical IO quotas or filesystem bandwidth guarantees.

This does not allocate1TiB of new storage. The104GiB output allowance,
1530GiB protected runtime free space,1634GiB launch threshold,1024calls with
zero retry,8updates,3600s wall and original751 semantics are unchanged.
Hashhooks' full logical tensor bytes/copy/hash wall are separately reported;
they are not equivalent to disk reads and must not become a throughput claim.

The original512GiB draft and its reason remain visible in DRAFT_SOURCE_FINDINGS.
Actual original-S0 stat evidence and final controller source are to be pinned
in the final independent review; this adjustment alone grants no dispatch.
