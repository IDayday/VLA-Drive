# Explicit second real4 diagnostic attempt

The original attempt remains FAILED_NO_AUTO_RETRY. Its CuBLAS deterministic-workspace error is recorded; the receipt has no traceback locating the exact failing call. No loss/gradient result or failure-post-state equality is inferred.

The minimal fix preserves the original math, precision, four training scenes and RNG seeds. Child Python receives `CUBLAS_WORKSPACE_CONFIG=:4096:8` before torch import. The actual child environment and pre-CUDA receipt were verified. Independent correctness ran 53 CPU tests; independent controller review ran 28. These are separate suites with overlapping cases, not 81 unique tests.

A fresh clean frozen runtime f212c96c and explicit second-attempt admission bind the reviewed bytes. The new limits are3000+40 seconds,208GiB logical reads,96GiB aggregate RSS,16MiB output,CPU2/3,GPU0. Prior actual416.604673s and51,404,923,553B plus these limits remain within the original3640s/256GiB envelope. No automatic retry, optimizer, model checkpoint or new teacher tensor cache is introduced. The old output is retained.

LAUNCH.json is a timestamped running snapshot, not a completed measurement. Collect the actual terminal receipt independently. No measured coefficient is automatically applied to formal T2/T3/T4, and this diagnostic does not establish throughput or same-backend training resume.
