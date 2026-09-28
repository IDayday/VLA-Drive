# Shared foresight implementation stage0

New isolated branch, starting from validated local DDP current-camera/Qwen/flow plumbing at a684207. Historical joint modules stay in repository history/legacy files; the new student does not instantiate or import their generator, detector, graphs, or trained weights. Reference official DDP: https://github.com/youngzhou1999/DriveDreamer-Policy . This is a defined local Action-Only implementation, not an unverified exact reproduction of the paper's ablation.

Implemented: lazy framework/video/depth imports; registration only of state/action tokens plus optional64 W; original ego-only FM head with optional explicit noise/time arguments; independent deterministic auxiliary readouts from SAME post-Qwen W; current-only train/inference path and removable heads; random GT trajectory MAE with target future removed before encoding; role scheduler; globally normalized masked losses; raw vehicle-only data preparation; atomic teacher trainer/resume/metrics.

CPU checks:12 passed (teacher target-future invariance, genuine visible-peer input, padding NaN/large-value forward/backward isolation, illegal valid inputs, batch1/odd role sequence/resume, ego-only/save-load, both readout gradients to W, detached labels, masked empty supervision, current-only vehicle selection, timestamp lookup). Import check: DDPForesight imports no WAN/PPD/video utility, DDPVehicle or historical joint flow modules.

Real64training-scene data:0failures,1381source vehicles,246selected neighboring vehicles,245with some future labels,1945valid neighbor time points;22ego-only scenes retained. Raw current object filtering precedes capacity16. Graph is built before future extraction. This is teacher-only GT data, never student input.

Not yet verified at this stage: real Qwen gradients/deployment equivalence, distributed normalization on actual ranks, teacher learnability/full training, full future-VAE cache, full student training or planning results. These are active work, not READY claims.

The official FLUX.1-schnell revision741f7c3ce8b383c54771c7003378a50191e9efe9 metadata was inspected. Default VAE artifact hash is f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3. Download currently returns gated401; requested authorized access/local path, no alternate teacher substituted. Qwen source hash/revision is preserved in GENERIC_SOURCES.json; only that verified generic model will initialize the student.

The new provisional campaign ceiling is6000GPUh on existing local/vla-zt2 devices, with teacher/data/short-test allocations separately limited. Formal student launch requires measured Action-Only throughput, fixed nonzero gradient-calibrated auxiliary weights, and a frozen complete plan. Old stopped experiments are not resumed. No new formal model results exist yet.
