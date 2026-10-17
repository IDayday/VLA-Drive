The private v3 evaluation helper candidates bind the sealed C3 preparation and add explicit routing across training-rl-zt2, training-vla-zt and training-vlawm-tsinghua-zt. Original ten-job and exact C2 definitions, complete native/export validation, dev1696, source6714, world1 inference and the 1728-call scoring cap remain fixed. The old unresolved C3 preparation remains rejected.

Candidate export SHA256: e58f58e3d0e3fb05e05dcd28c5d04142c4f3a4ba2002e5cc8ea9b3a5bf95217f.
Candidate score SHA256: 9217c5c0b5362453a86843bfe8dd9c6ad1609b7440c5441be4bd1af6511e8d7c.

[HANDOFF.json](HANDOFF.json) binds the exact candidate, baseline, diff, test and runbook bytes under [the private audit directory](/mnt/project/st-world-policy-20261016/audits/evaluation/C3_ROUTE_HELPER_v1). The private [README](/mnt/project/st-world-policy-20261016/audits/evaluation/C3_ROUTE_HELPER_v1/README.md) gives the invocation forms; [ROUTE_SCHEMA.md](/mnt/project/st-world-policy-20261016/audits/evaluation/C3_ROUTE_HELPER_v1/ROUTE_SCHEMA.md) defines the required immutable job route. These candidates do not alter any deployed helper or frozen runtime.

Implementer validation passed 99 metadata/negative/lock checks; Lead independently repeated all 99 in 3.756 seconds. Both --help invocations exited zero. Validation used one bounded CPU lock child and mocked GPU metadata. No experiment was launched, no model/checkpoint payload was loaded and no evaluation result was generated.

The four-export cap applies campaign-wide only when all new exports use v3. Lead confirmed the prior C0 export/score jobs completed and future new exports will use v3; preserved v1/v2 helpers must not dispatch new exports concurrently. Existing score slots already share the two-job cap. Actual host availability requires fresh admission, with no preemption or sharing of occupied training GPUs.

Cross-host flock operation and actual host-specific export/score latency were not measured. The candidate rejects a changed shared Lustre namespace or missing global flock semantics. Lead alone deploys the reviewed bytes and creates actual checkpoint routes after complete receipts and current resources are available. No new scientific selection or analysis rule is introduced.

This report commit is based on 6ea815bb0ed7feeaccf626ea7723970ad26d25e4 and owns only this new report directory. The candidate implementation remains in the private audit directory; this report contains small evidence pointers rather than raw host inventories. The pre-existing untracked smoke_dev16_aa directory is preserved.
