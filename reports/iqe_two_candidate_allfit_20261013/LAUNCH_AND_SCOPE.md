# Full-fit expansion, 2026-10-13

The user requested full training-population mining, then explicitly selected
Scorer fitting on all 90,892 legitimate training scenes. This is a new experiment;
the running smaller experiment and every Base training process remain unchanged.

## Frozen identities and data

- Source commit: `b42dd30edb449c59702af96222b75e6284ec5266`.
- Branch: `feature/iqe-two-candidate-all-fit-70k`.
- Experiment: `two_candidate_base70k_allfit_seed31415_v1`.
- Root: `/mnt/project/iqe-two-candidate-baseline-20261013/base70k_allfit_v1`.
- Base: `/mnt/project/iqe-remote-rolling-cache-20261012/continuation/step_070000.pt`.
- Base SHA256: `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`.
- Base binding SHA256: `27525c4c8b012b0a94922fc8a45a06a835776deb3314f62dc36481cc3f03fa46`.
- Scene manifest SHA256: `95a5908732c9888cd3c52d6a9e115ad9d1134558573f1006584e5c6ed91c7872`.
- Partition SHA256: `1ec91db34fec3d713de3e4963a98f9668bc7b6e6b7ae1681739dca275ef53487`.

Expert mining covers all 74,311 scenes in D_E; D_S retains 16,581 scenes excluded
from expert fitting. These immutable log/source-group assignments are the same
as the smaller run's full partition, with its scene-count caps removed.
Stage validation remains the same 1,024 scenes; calibration remains 1,024;
development remains the complete 1,696. No final_test/navtest scenes are included.
There are 94,636 total contexts in this experiment.

The Scorer scope amendment was recorded at `2026-10-13T05:40:28.371092+00:00`,
before either new head consumed an optimizer update. Its immutable SHA256 is
`30262251de3d631ed3264d0b7d285d937bfcf74432578fe392e05b5c21031281`.
Both Scorer sampling streams now use D_E union D_S, exactly 90,892 scenes.
Validation/calibration/development remain outside gradient fitting.

D_S is held out from the new expert's training, but is now Scorer training data.
It must not be reported as held-out Scorer generalization. Reports distinguish
the D_E/D_S source pools and, separately, actual optimizer-consumed expert scenes
from never-consumed scenes. Merely belonging to D_E does not mean the expert was
trained on that scene. Base exposure is inherited and recorded in the binding;
post-training partitions are not automatically unseen by Base.

## Actual execution

Verified 23,744 complete same-Base contexts were atomically hard-linked from the
smaller run. Full SceneRecord, target provenance, runtime, schema, checkpoint,
partition and every payload checksum were checked. No score labels were imported:
official labels use the destination's own locked request identities.

Reused logical bytes: 30,081,985,920. Observed payload size: 1,266,930 bytes/scene.
The 70,892 new contexts imply approximately 83.65 GiB additional payload storage;
this is a capacity estimate, not a claim of completed output. Shared storage had
approximately 1.43 TiB free at launch, with Base checkpoint/pool reserves checked.

On `training-vlawm-tsinghua-zt2`, GPUs 1/2/3 are generating three disjoint context
shards. All three were observed computing at 99–100% GPU utilization, with
`W=[4,144,2048]`, `A=[4,8,2048]`, planner memory `[4,16,256]` and a CUDA allocation
peak of 10,357,075,456 bytes. A concurrent CPU pipeline scores Base trajectories
and audits GT on D_E/stage_val failures. The head workflow waits for complete
cache and reference-scoring receipts, then trains continuously without artificial
pause/resume tests.

GPU0 preserves the previous smaller Scorer experiment and its final audits.
GPUs4/5 have two additional Base dynamic-cache producers, already committing
windows. GPUs6/7 run the separately authorized 80k Navtest; none of its labels
enter this experiment. All eight GPUs have real tasks.

The Base continues its unchanged four-objective run on vla-zt2. At the resource
snapshot it reached step 81,335. Cache waits remain observable; the monitor's
recent 200-update mean was 1.5931 seconds including wait. This is a live sample,
not a controlled speedup comparison. G-series GPU owners, the reserved Flow RL
host, and unknown external GPU owners were left untouched. No process signals,
GPU resets, Base restarts, ACTIVE changes or push were performed.

## Retained training and tests

- One independent expert, 10,000 updates, global batch 128, 80% audited hard and
  20% ordinary samples. Thresholds and the maximum 2,048 ordinary-scene budget
  are unchanged. Final H/N counts await the complete full-fit audit.
- One random-initialized Scorer, 20,000 updates: 5,000 pointwise, then 15,000
  mixed natural/comparison updates. The comparison distribution uses actual
  absolute score gaps without multiplying the gap a second time.
- AdamW lr 1e-4, wd 0.01, warmup 50/100 then constant, clip 1.0, validation every
  250 updates. Expert dropout inherited; Scorer dropout 0.0. No external scorer.
- Stage validation selects both checkpoints. Optional margin uses selector_cal;
  raw argmax and the full fixed development results remain separate.

The targeted CPU suite passed **81 tests**, with one existing pynvml deprecation
warning (1.62 s). This covers immutable context reuse, corrupt/changed identity
rejection, scope locking, train-entrypoint coverage, actual two-stream population,
role exclusion, partitioning and budget behavior. Tensor/unit evidence does not
stand in for the running full-fit training or its scientific results.

Full-fit context extraction and official scoring are **RUNNING**. Expert and
Scorer training are **WAITING_INPUTS** in their already-launched workflow queue.
Full-fit specialization, selected gain and safety results are **UNTESTED**.

Runtime evidence lives under the experiment's `runtime/`, including
`RESOURCE_LAUNCH.json`, `CACHE_LAUNCH.json`, `BASE_PREFIX_STATUS.json` and
`HEAD_QUEUE_STATUS.json`. This report intentionally excludes the private dynamic
cache engineering implementation from the source commit.
