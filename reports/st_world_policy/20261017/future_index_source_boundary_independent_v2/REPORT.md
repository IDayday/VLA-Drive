# Independent source-boundary follow-up

The two previously demonstrated defects are fixed at `e8ac21550ae0d38bf8414486282c0b41908d3db3`. This review is independent of the implementation and does not inherit its 14 passing tests.

Actual narrow rerun: **2 passed, 0 failed, 0 errors**. A real empty FIFO is rejected in 0.000110 seconds. The actual `main` prelude interrupts a deliberately slow first source loader after 0.048923 seconds, respecting an already partly consumed absolute budget; it never reaches `run`. CPU affinity stays 8/9. Test process wall time is 0.1047 seconds, `/proc/self/status` VmHWM is 24,539,136 bytes, and process `rchar` is 2,126,077 bytes. No Torch, GPU, real payload, actual ledger mutation, or actual dispatch occurred. Synthetic payload read/write is zero.

The original report remains unchanged: 9 passed / 1 failed / 1 error. Its FIFO `TimeoutError` was the original resource-boundary defect, not a harness result to erase. The new bootstrap guard need not use the later adapter's function name: the functional test verifies the same absolute deadline directly. The unchanged monitor and post-wait accounting are not reimplemented or broadly retested.

The actual Lead bindings (`c15e7262…`) were independently checked against the wrapper, preparer, TRAIN runtime pins, current policy and read-only ledger; 152,400 explicit bytes were read. Both proposed output directories were absent. The exact argv uses the new wrapper hash, so it does not rely on the older descriptive SOURCE_PINS wrapper entry. The fresh 2303 GiB checkpoint reservation remains in force without reducing existing obligations.

`ADMISSION.json` permits exactly one CPU metadata-control preparation invocation: CPU0/1, 120 seconds plus 40 seconds owned cleanup, 1 GiB RSS, 384 MiB total process logical reads including imports and nested validators, and 1 MiB combined new output/control space. The Lead must freshly verify the recorded pins/ledger and output absence before launching. This is not permission for migration, index building, model execution, retry, or training.

The subsequent `run_index_once.py` still reads its pinned core at module import before an absolute alarm. A hash checked after an unbounded read does not itself bound that read. That separate build entry needs the same small reader/bootstrap adaptation or a verified whole-process outer deadline; it is not executed by this preparation and does not invalidate this limited permission. No actual full-population read cost or throughput is claimed here.
