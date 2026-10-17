# Independent bootstrap compatibility check

**PASS for the narrow behavioral delta** at source `e8ac21550ae0d38bf8414486282c0b41908d3db3`, wrapper SHA `49b21405d1dacee17405598c1a7165727db9855af4de66c2feff70ab30b93ce5`.

Five independent tests passed. Exact AST comparison establishes that `run`, `options`, `load` and all existing top-level assigned constants are unchanged. Therefore the prepared argv, output namespace, monitor invocation, cleanup path and budgets retain the previous reviewed definition. The new bootstrap context wraps option parsing and initial adapter loading, then exits before the existing finite context starts with the **same original absolute deadline**. It does not restart the 120-second timer.

A regular source returns exactly its pinned bytes and rejects a wrong hash. The real CLI `--help` exits 0 with the expected options; malformed arguments exit 2, before worker affinity or any actual preparation. The observed help text is preserved in `HELP.txt`. The independent resource reviewer owns FIFO/race/timeout negative tests; they are not duplicated or claimed by this functional report. The old FIFO failure remains historical evidence for the earlier source.

The actual Lead binding `future_index_preparation_actual_v2/BINDINGS.json`, SHA `c15e726202a788eb027a8ce42f3c484dee1320dfb9338c9abd79f9fbf94a1cd0`, matches this wrapper and the existing preparer/TRAIN source pins. Its output remains the same isolated finalized-phase path, with 120-second work/40-second cleanup, 384-MiB read, 1-GiB RSS and one-MiB combined output budget. It grants no ledger mutation, index build, model execution or retry. The referenced policy/ledger were not opened in this review.

Tests ran on CPU6/7 in 0.088948 s, parent peak RSS 24,088,576 bytes. Metered parent and child run-path reads totalled 2,744,072 bytes. These counters begin after stdlib/bootstrap imports in the tiny CLI subprocesses, so they are not advertised as complete interpreter-startup IO accounting. Only known small source/control files were inspected; no Torch, payload, ledger or actual worker was loaded. The external review timeout was 55 seconds. No resource or launch admission is signed here.
