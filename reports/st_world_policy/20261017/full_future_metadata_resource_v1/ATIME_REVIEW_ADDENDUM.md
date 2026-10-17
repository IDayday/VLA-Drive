# Access-time preflight failure and narrow correction

Lead attempted the previously approved dev PLAN wrapper. It failed in preflight
before output-root creation or child spawn. Lead reported that the expected
admission SHA matched and that only the file access timestamp changed between
the two stat calls. The original admission and wrapper evidence are preserved.
No real PLAN or builder result is claimed by this review. The reviewer confirmed
both new output roots still absent and the live ledger unchanged before issuing
the explicitly versioned replacement admissions.

The old pinned reader compared whole `stat_result` objects, which include atime.
Reading a file on the actual shared filesystem can legitimately advance atime.
Owner commit `b0ae418cb51b3cc67259a80144d571aabd3a82b0` makes the sole source
change a stable tuple containing device, inode, mode, link count, size, mtime_ns,
ctime_ns and block count, while retaining the expected SHA check. Access time
is deliberately excluded. This changes neither content validation nor resource
limits and does not permit a source change during reading.

Final wrapper SHA256:
`feaadec9ccac6efc4a302fe7ca8bd2f55efa1f415ab77830befd571d7e5d3734`.

Targeted CPU tests: **54 passed in 0.19 s**, consisting of 12 independent cases
and 42 owner cases independently rerun. Independent additions verify atime-only
acceptance and rejection of mtime_ns, ctime_ns, inode and size changes. The
owner's other fields and changed-content cases also passed. No real PLAN was
called. The first independent fixture revision accidentally counted the
`safe()` lstat as the initial pinned stat, masking injected changes. That test
fixture was corrected to leave `follow_symlinks=False` calls alone; its first
failed output is retained and is not misreported as an implementation defect.

Only the replacement `DEV_PLAN_ADMISSION_V3.json` and
`TRAIN_PLAN_ADMISSION_V2.json` may be used with this fixed wrapper. They bind the
same immutable dev/train runtimes, exact manifests/argv, original helper hashes,
2183 GiB checkpoint obligations and ledger hash. The outer scope remains a
single CPU-only metadata/stat PLAN for each role, 180 s plus at most 40 s owned
cleanup, and 32 MiB root-inclusive storage. This is an explicit corrected
preflight attempt, not an automatic retry loop or build authorization.
