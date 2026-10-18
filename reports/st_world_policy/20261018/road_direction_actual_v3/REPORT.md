# Corrected fixed32 road direction diagnostic

All32 original TRAIN cases retain their original population, trajectories, step size, context and1e-9 geometry tolerance. Geometry now passes for every case. Seven have a defined direction and complete official scores;16 report ZERO_OR_NONFINITE_GRADIENT and9 report MOVING_CORNER_BOUNDARY_OR_NEAREST_TIE. These25 are retained as undefined, never silently dropped or called valid zero scores. The existing error code does not distinguish zero from nonfinite gradients, so no more specific claim is made.

On the seven common valid cases only:

|Variant|PDMS percent|Change vs original, percentage points|
|---|---:|---:|
|negative|87.42908785|-0.00330635|
|original|87.43239420|+0.00000000|
|positive|87.43567651|+0.00328231|
|random|87.44136100|+0.00896680|

Negative-gradient executed proxy cost is lower than positive in 7/7 and fixed random in 7/7. Mean differences are -3.473805740146088e-5m and -1.3002902314237781e-5m, respectively. This is the custom margin cost after execution, not an official PDMS gradient.

The negative direction slightly reduces official score through progress; NC/DAC/TTC/comfort do not change on the seven scored cases. No new zero score or high-to-zero event occurs there. All32 originals had DAC=1: the road-repair denominator is zero. Neither these seven results nor their mean can establish failed-road recovery, whole-population score gain or held-out policy improvement. This does not authorize proxy-driven formal training.

Actual execution:21 new single-trajectory calls; original labels reused. Prior v2 12 calls remain as separate failed-version evidence; total33. Wall61.161s, aggregate parent/worker/tracker peak1,520,934,912B, worker exit0 and cleanup confirmed. No GPU/model load/optimizer update. The CLI returns2 because25 directions are undefined; this is not disguised as complete scientific success.

The initial v3 admission omitted maps_root. It failed with0 started scenes/0 scoring calls, and its report remains intact. Lead supplied the existing /mnt/navsim/maps and resumed only untouched cases, with the same registered source/population.

Identities and raw artifact pointers are in RESULTS.json. Source0d0d19dd; registration3bb6805165186638ac69e01c43e94655a4ee412955a761ab9844c8657366a3a8. Feature research remains paused.
