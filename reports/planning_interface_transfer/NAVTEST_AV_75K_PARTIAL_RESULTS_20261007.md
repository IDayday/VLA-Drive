# A/V75k–80k Navtest progress — 2026-10-07

Observed 2026-10-07T15:01:09.644211+00:00. A_ACTION75k is COMPLETE; the other five requested evaluations continue in parallel.90k/100k await their exact checkpoints.

| Arm |75k PDMS |75k exported scenes |80k PDMS |80k exported scenes |
|---|---:|---:|---:|---:|
|A_ACTION|88.870727|12146/12146|PENDING|5837/12146|
|A_NO_MAE|PENDING|5635/12146|PENDING|5820/12146|
|V_QUERY|PENDING|5849/12146|PENDING|5926/12146|

A_ACTION75k: full12146 scenes/136 logs,0 failed, 580 official score==0 scenes. PDMS 88.870726940; ego ADE 0.862886m, FDE 1.840808m.
Its50k PDMS was88.373807694, a temporal checkpoint difference of+0.496919 points; zero scenes decreased from636 to580. This is not an isolated method comparison against75k A_NO_MAE/A_W, whose matching results are not yet available.

| NC | DAC | TTC | EP | Comfort | DDC |
|---:|---:|---:|---:|---:|---:|
|98.085790|97.077227|94.590812|83.279769|99.991767|98.365717|

Native FP32 optimizer-master restoration and FP32 inference, TF32off,10Euler,one candidate,scene-bound samplingseed42. The original official full-precision environment and all objects are preserved. Complete population/CSV SHA/checkpoint/four shard validation was rechecked; the applicable full official backend qualification is reused and eight fresh distinct-log official spots have maximum seven-metric error0. No new full per-model reference replay is claimed. Trainseed42/samplingseed42 only; no multi-seed stability claim.

Checkpoint SHA256 `ed7e80df12f013b38dd913886febd5e868e10b4202b8403daf5e366dc1f62740`. Training/native evaluation source `f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882`; parallel execution source `4b566a6c6c6d3b4b6212ed19d7af018485d5298d`. Report commit is separate from those immutable sources.

The other prediction counts above are incomplete populations and are never reported as PDMS. The five inference exports and bounded CPU scorer queue are RUNNING; no task restart, method change or test-driven recipe selection was performed for this progress report.

Pinned original-trajectory S3 reference (same-update comparisons only):

|Updates|PDMS|
|---:|---:|
|50000|88.503739|
|60000|89.020719|
|70000|89.280841|
|80000|89.214882|
|90000|89.295769|
|100000|89.408179|

Original S3 has no exact75k score. Its80k89.214882 is a reference awaiting completion of the new80k arms; do not interpolate75k or compare different updates as a matched method gain.

[Aggregate evidence](evidence/NAVTEST_AV_75K_PARTIAL_RESULTS_20261007.json). Prediction banks, raw private scene/log rows, metric caches and weights stay outside Git.
