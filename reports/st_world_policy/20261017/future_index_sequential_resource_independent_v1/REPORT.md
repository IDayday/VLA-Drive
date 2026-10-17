# Independent index-preparation wrapper resource review

**FAIL** atbf67fad2. This is a narrow source/control review, not an index build.

`prepare_once.py::read_source` checks path and size but does not reject non-regular files before `read_bytes`. An actual temporary FIFO with size0 blocks in open until the independent200ms test alarm fires. The initial adapter load is outside the wrapper's invocation-alarm context, so that path is not bounded by its declared120-second deadline.

Eleven independent cases produced9PASS,1FAIL and1ERROR. The error is the concrete FIFO timeout, not a model or dependency failure. All evidence remains inTEST.log. The small test process used CPU8/9 for0.648s, sampled peakRSS19,288,064B and1,559,482B once-charged logicalread. Fixture writes were two four-byte regular files plus a FIFO and symlink. No real data, live ledger, GPU, workload signal, or dispatch occurred.

Exact fixed monitor and finite-adapter hashes, BOOT_IO placement, once-charge reuse, limits, regular pin acceptance and wrong-hash/relative/symlink rejection passed. The already reviewed finalized TRAIN binder and unchanged DEV route are not being redesigned. A narrowly bounded source-open/deadline correction is required before an actual preparation admission. No implementation or author test result is counted as our PASS.
