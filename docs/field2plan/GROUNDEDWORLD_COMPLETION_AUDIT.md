# GroundedWorld-VLA completion audit

Audit date: 2026-08-10 UTC  
Repository commit at audit start: `30505ee3a86326892f8be6c2cc04ca30ab18c93f`  
Branch: `main`

The worktree was already dirty and contains user changes and prior Field2Plan
artifacts. No reset, checkout, vendored NAVSIM edit, or external download was
performed.

## Algorithm contract

| Revised-plan requirement | Actual code entry | Tensor/interface | Status |
|---|---|---|---|
| Isolated complete framework | `QwenOFT_GroundedWorld.py` | standard framework output | implemented/tested on CPU fakes |
| Legacy baseline unchanged when new framework is not selected | framework registry/config isolation | legacy `QwenOFT` config unchanged | implemented/tested |
| Calibrated current geometry | `GeometryFieldWriter` + camera contract | `[B,Cg,Ny,Nx]` | implemented/tested |
| Multi-scale ego memory | `MultiScaleGeometryMemoryWriter` | 3 levels, factors 1/2/4 | implemented/tested |
| History ego-motion alignment | `CurrentDynamicsEncoder._warp_history_to_current` | history `[B,4,Cg,Ny,Nx]`, transforms `[B,4,4,4]` | implemented/tested |
| Current/history external dynamics prior only | `cache_current_prior_adapter.py` | frames 0--3; `[Th,V,Ct,Ht,Wt]` | protocol implemented; real adapter external/missing |
| No fake JEPA patch-to-BEV correspondence | `ExternalPriorAdapter` + `global_alignment_losses` | target/prediction `[B,Cd]` | implemented/tested |
| Action-free future | `PredictiveMemoryForecaster` | `[B,8,Cd,Ny,Nx]` | implemented/tested |
| Shared teacher-independent future target interface | `FutureTargetCacheReader` | student-EMA `[B,8,Cd,Ny,Nx]` | implemented/tested; full cache not generated |
| No GT future action in writer | `GroundedWorldCore.forward` | no action argument | implemented/tested |
| Internal first trajectory | `_planning_forward` | normalized `[B,1,8,4]` | implemented/tested |
| Swept-tube physical read | `MultiScaleTrajectoryTubeReader` | points `[B,M,8,K,2]`, context `[B,M,8,Cr]` | implemented/tested |
| One bounded zero-init refinement | `TrajectoryRefiner` | delta `[B,M,8,3]`, final `[B,M,8,4]` | implemented/tested |
| Shared visual gradients in III-B | `_run_baseline_method(batch,"forward")` reuse | captured `[B,V,Cv,Hv,Wv]` | implemented/tested |
| Training-only physical consequence grounding | `PlanningConsequenceHead` | candidates `[B,K,8,3]`, values `[B,K,6]` | implemented/tested |
| No aggregate EPDMS head/reranker | consequence manifest/schema | six separate components | implemented/tested |
| Strict checkpoint chain | framework declared loaders + launchers | Stage I -> II -> III-A -> III-B | implemented/tested with fake checkpoints |
| GroundedWorld inference load | `infer.py` | strict combined checkpoint load | implemented/static tested; real GPU not run |
| Same-checkpoint access removal | inference intervention | reader context/gates zeroed | implemented/tested |
| NAVSIM-v2 navtest/navhard evaluation | `05_eval_checkpoint_navsim_v2_16gpu.sh` | one-stage / two-stage | launcher/result validator tested; full eval not run |

## Scientific attribution contract

Implemented matrix interfaces:

- B0--B5;
- real/no-teacher/random/scene-shuffled/GT-task/generic-VJEPA controls;
- supervision with no access and same-checkpoint access removal;
- common future-target manifest across controls;
- module learning diagnostics from actual JSONL records;
- non-aggregate physical consequence labels.

Not yet implemented from the later research agenda:

- dense/sparse/intent future representation comparison (dense is the current main path);
- static-slot vs dense-field storage comparison;
- online-teacher upper bound;
- dynamic-agent occlusion, visual nuisance, and geometry intervention suites;
- Bench2Drive/CARLA reactive branch rollouts;
- automated experiment submission to a specific cloud/DLC API (the repository
  provides fail-fast container entrypoints and an actual-result statistics
  generator, but does not assume a site-specific scheduler API).

These are experiment extensions, not silently claimed as completed.

## Launch gate

| Item | Local state |
|---|---|
| CPU GroundedWorld contracts | PASS: 50 GroundedWorld tests; 175 combined Field2Plan/GroundedWorld tests |
| 16-GPU smoke | not run |
| VGGT full cache | not run/validated in this work |
| Driving-JEPA repo + checkpoint + adapter | missing external input |
| Driving-JEPA full current/history cache | missing because adapter is absent |
| Stage-I EMA future cache | not generated |
| navtrain metric cache | smoke cache only; full cache not generated |
| B5 consequence cache | not generated |
| navhard processed metadata | not generated |
| NAVSIM-v2 evaluator caches | not generated |

Consequently, code construction is substantially complete, but a formal B5
job should not be submitted until every required manifest passes launcher
preflight. B1/B2 can proceed independently once the VGGT cache exists.
