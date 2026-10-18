# Cold-import and output-namespace delta

Parent controller19a5 actually failed during cold `route_checks → fresh_environment_v3` with `ModuleNotFoundError: tools`. The earlier51 CPU fixtures mocked that route and did not cover this import. The actual failed driver log and all v1 prepared JSON remain intact. Eight v1 prepared metadata files exist; no START, attempts, run, model or admitted scoring artifact exists. Lead confirmed that the outer driver exited. Consumption is0 updates/0 scores.

Final clean source01d8dc4733feb98e562eb9a4f66bd4e56c9db114 (controller763aa490...) has exactly two controller AST edits: add frozen RUNTIME to sys.path before mature helpers, and use a new v2 root. All other controller AST, original751 runtime, math, state hooks, pause protection and budgets remain unchanged. No protected source or old evidence was rewritten.

The independent delta check ran only two isolated `python-I` cold processes, from a temporary unrelated cwd and without tools/RUNTIME pre-imports. Old19a reproduced the actual import failure. New01d8 executed the actual frozen helper and actual CPU environment comparison successfully, then stopped deliberately before mount/GPU/model/trainer/resource probes. Wall4.20s, childCPU3.93s, maximum child RSS657,649,664 bytes. The old51 suite was not rerun.

The new binding396accfd... was verified from small metadata only: exact source/controller, v2 root, unchanged budget and32TRAIN manifest, fixed software, authorized host, disjoint4+4 UUIDs and current leases.194,488 bytes of referenced small metadata were rehashed. Correctness does not replace Systems' fresh availability checks.

`BOUND_REVIEW.json` is usable only for that exact v2 binding and limited8-update/1024-call/zero-retry SMOKE. The new run has not yet been executed by this reviewer. Diagnostic weights remain prohibited as formal initializers. The historical v1 review is retained as history and cannot admit the new source.
