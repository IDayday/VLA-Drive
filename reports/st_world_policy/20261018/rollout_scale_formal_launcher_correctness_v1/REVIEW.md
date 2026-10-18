# Independent external formal control review

The frozen launcher and monitor pass 29 independent CPU negative/control tests. This is a correctness review of the external controls for the single registered rollout-scale arm; it does not approve live resources, completed PROFILE efficiency, or dispatch. The existing model correctness signature is preserved.

## Fixed issues

1. The previous six-attempt check counted only `LAUNCH.json`, so interrupted attempts could disappear from the count. It now counts every retained immediate invocation directory, including partial attempts, before either fresh or resume handling. The directory is created before dispatch.
2. Python booleans previously passed the remaining-byte integer check. The reservation parser now requires an exact nonnegative integer.

## Current evidence

29 tests passed in 3.28s pytest time (4.48s process wall, 4.37s child CPU, 652596KiB peak RSS). No GPU queries, model loading, training, scoring or real dispatch occurred: process/GPU interfaces are explicit fixtures. The mature three-role gate was executed against real tiny files and SHA256 values.

Tests exercise each external-control hash rejection; wrong role, reused reviewer, bad status, stale fingerprint, corrupted evidence; selected GPU startup PID at zero memory; wrong UUID and malformed query; six partial attempts; fifteen-native budget; terminal refusal; resource mismatch; only the owned stop marker; protected free space and 300s recursive-stat reuse. The native and stop helpers operate on isolated temporary paths.

The immutable storage plan reserves 209GiB for this run, 1024GiB shared floor, 4GiB unresolved prior writes, and 292GiB other active writes: admission requires 1529GiB free. The monitor retains a 1320GiB free floor as this run writes. Actual writer estimates and availability are independently verified by Systems; this review checks arithmetic and enforcement.

## Retained environment failure

The initial run imposed a 2GiB *virtual-address* limit. Torch could not map CUDA shared libraries, so sixteen launcher cases failed during import and twelve other cases passed. This was an audit harness limit, not a launcher failure. The original full log remains. The final run instead monitored the authorized 2GiB physical RSS budget (no GPU use); all 29 tests passed and source hashes were unchanged.

## Boundaries

RSS/GPU/free/native checks run every 5s, output-size scans at start/every300s/end. These are sampled guards with a safe-boundary stop request, not hard instantaneous quotas. The monitor sends no process signal. The launcher uses the existing outer timeout, does not loop or auto-resume, and cannot change the registered optimizer clock. The six-attempt count deliberately includes incomplete dispatch records.

No change was made to the model source, runtime registration, existing correctness signature, or real training artifacts. Exact executable and storage hashes are in `REVIEW.json`; any change requires affected-delta revalidation.
