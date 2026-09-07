# Fixed-epoch Navtest comparison, 2026-09-07

This campaign evaluates the completed BaseInit Task-Future Lite epoch27 against
the previous BaseInit PlanReg-WM epoch27. It does not select an epoch, change a
loss/sampler, or start further training. The previous epoch33 continuation is not
the equal-epoch primary comparator. No VQA model is added to this campaign.

## Locked artifacts

- Lite training source: `3b7d1665de4d2230abbc94213fc6c3116de0bea0`.
- Working source before evaluation helpers: `b32b1dee06fb42003a2f69490366cc46db419195`.
- Lite student checkpoint SHA-256:
  `41f582e67d5d37e2c9691ac6a200db6d07aa17db0b7324e97afd41fe669f0f8c`.
- Lite full training checkpoint SHA-256 (export manifest):
  `856b69c1ed572d73796c8c6fdf09c7aba81ce721710fde370ff912ef08a79d29`.
- Old epoch27 candidate-bank SHA-256:
  `615538bb38934654290d0eff043e0598e8341690d90dfe516def0ae696f871a9`.
- Both are VLM-only BaseInit, 103,288 trainval scenes, GB128, 27 dataset epochs,
  21,789 optimizer updates. Architecture/auxiliary losses differ; this is not a
  matched no-WM causal experiment.

Artifacts are stored under:
`/mnt/project/DriveVLA-M0-formal-runs/evaluation/lite_epoch27_navtest_20260907`.
No old checkpoint, proposal bank, metric cache, or audit directory is overwritten.

## Current-only inference

The exact checkpoint class is
`navsim.agents.EpisodeDrive.episodedrive_agent.EpisodeDriveAgent` from the Lite
worktree. Student loading requires all 1,135 state keys to match. EMA, the physical
query decoder, and the legacy future predictor must not be instantiated.

`scripts/export_planreg_current_only_navtest.py` reuses the existing agent forward,
Lightning candidate export, and NPZ serializer. Its dataset only constructs
`AgentInput`, never a full `Scene` or a target builder. It returns empty targets.
Future/target feature keys are rejected. Official scoring is a separate process.

Precision: `trainer.params.precision=32`, `vlm_config.compute_dtype=float32`, all
floating model parameters checked as FP32, TF32 disabled. Thus this particular
evaluation is full FP32, unlike BF16-VLM training. Read-only attention uses the
already-audited `split_sdpa` backend. Batch size is one per GPU; four hosts each
use eight GPUs, with disjoint whole-log shards and atomic per-rank output files.
Each inference manifest records the checkpoint SHA, actual class, dtype counts,
helper source SHA, and proposal-bank SHA. Neither targets nor auxiliary answers
are appended to the scorer input. Selection remains the scorer's log-score argmax.

## Scoring and acceptance

The existing validated two-stage scripts in
`/mnt/project/DriveVLA-M0-stage2-repro-fix/local_stage2` evaluate all 64 immutable
proposals in batches and independently check standard single-trajectory scoring
for each selected trajectory. Per-log Ray tasks resume without GPU re-inference;
each host has 64 physical cores, with one configured Ray worker per core and all
nested BLAS/OpenMP thread pools limited to one.

The old bank is repackaged without changing its coordinates, logits or selections
and rescored through the same CPU workflow. This is a fresh scoring audit of old
model outputs, not a fresh old-model inference run.

Required gates: 12,146 unique tokens, 136 logs, 64 candidates per scene, zero
invalid scenes, identical token sets, NPZ/CSV reconstruction, regret identity,
and batch/single score parity. Float32 NPZ persistence tolerances are kept
separate from the 1e-8 official double-precision scoring parity threshold.

Before full inference, a fresh four-scene public-weight gate matched the archived
released-evaluator CSV exactly (maximum absolute difference zero). The archived
full public audit was also revalidated: all 12,146 tokens matched, maximum
difference `8.777423232686488e-12`, below the unchanged 1e-8 threshold.

Paired differences use log-cluster bootstrap confidence intervals. Report selected
PDMS and offline Oracle@64 separately: Oracle is not a deployable model score.
Comparisons of different generated banks do not isolate scorer architecture
quality or prove that the WM alone caused a performance change.

## Runtime scoring sources

Both CPU scoring campaigns use the actual imported classes in the validated
`DriveVLA-M0-stage2-repro-fix` tree:

| Class/function file | SHA-256 |
|---|---|
| `navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py` | `02ab075017d4edb1c8bba6476ef1674b84d43dbca0f599df87cfe052b2dd5211` |
| `navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py` | `6908be76ade1831ad8386a9927c879477ab83f5629a4013d71125f1ab013bec7` |
| `navsim/planning/metric_caching/metric_cache.py` | `b994b5c9221c0796c3c511f5c339102fc1b720a48be08a5d7aaa0ad2108b7185` |
| `navsim/agents/EpisodeDrive/score_module/compute_navsim_score.py` | `b40bed94e0ab9b60dd982f0aed8d2d388cdcb354131709a3cbe288f20d24dead` |

The model's fixed-source scorer decoder, six heads, loss, detach boundary, and
aggregation are not modified by this campaign.
