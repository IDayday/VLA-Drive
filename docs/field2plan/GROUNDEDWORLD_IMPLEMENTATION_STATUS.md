# GroundedWorld-VLA implementation status

This implementation follows `GroundedWorld_VLA_Revised_Research_Plan.md` and
is isolated behind the new `QwenOFT_GroundedWorld` framework. Existing
`QwenOFT` and `QwenOFT_Field2Plan` defaults are unchanged.

## Implemented algorithm

1. **Physical world path**
   - current/history Qwen visual features are projected into calibrated ego BEV;
   - history fields are warped to the current ego frame with explicit SE(2)
     transforms;
   - geometry memory is multi-scale;
   - the current dynamics memory is aligned only to current/history external
     prior features;
   - because the external teacher cache does not declare a calibrated patch-to-
     BEV transform, the JEPA target is confidence-pooled to `[B,Cd]` and aligned
     to pooled current dynamics. It is not falsely resized into an ego-BEV.
2. **Predictive memory**
   - action-free future memory has shape `[B,8,Cd,Ny,Nx]`;
   - Stage I maintains a world-path EMA;
   - the offline future-target tool applies that EMA to future observations;
   - all teacher controls use the same `student_ema` target manifest.
3. **Planning path**
   - the repository's VLM+DiT produces one first-pass trajectory;
   - a multi-scale swept-tube reader queries geometry and current/future memory;
   - one bounded, zero-initialized refiner changes that same trajectory;
   - no high-noise flow state queries and no candidate reranking are used.
   - in Phase III-B, world losses reuse the differentiable planner forward so
     they reach the shared visual encoder; draft sampling remains detached.
4. **Planning consequence grounding**
   - deterministic GT-centered perturbations are training-only;
   - the head predicts clearance, TTC, collision, lane distance, progress, and
     comfort separately;
   - unavailable labels remain masked;
   - aggregate EPDMS is rejected by the cache schema and the head is absent
     from inference decisions.

## Tensor contracts

| Interface | Shape |
|---|---|
| tapped current/history visual field | `[B,Th,V,Cv,Hv,Wv]` logically, encoded one frame at a time |
| finest grounded geometry | `[B,Cg,Ny,Nx]` |
| geometry pyramid | `[B,Cg_i,Ny/s_i,Nx/s_i]`, `s=(1,2,4)` |
| current dynamics memory | `[B,Cd,Ny,Nx]` |
| external current/history prior cache | `[B,Th,V,Ct,Ht,Wt]` |
| confidence-pooled external target | `[B,Cd]` |
| action-free future memory | `[B,8,Cd,Ny,Nx]` |
| shared EMA future target | `[B,8,Cd,Ny,Nx]` |
| draft/final normalized action | `[B,1,8,4]` |
| physical trajectory | `[B,1,8,3]` |
| swept-tube context | `[B,1,8,Cr]` |
| consequence candidates | `[B,K,8,3]` |
| consequence components | `[B,K,6]` |

Formal defaults are `Cg=(128,192,256)`, `Cd=192`, `Ny=Nx=64`, `Cr=256`.
The debug config uses `(32,48,64)`, `Cd=48`, and `24x24` fields.

## Attribution signals

The trainer logs the following without running NAVSIM evaluation:

- `geometry_depth_mae_m`, occupancy accuracy, relative-geometry MAE, valid ratio;
- current-prior cosine similarity, scene-shuffled similarity, and their margin;
- future cosine/SmoothL1, temporal-shuffle margin, and uncertainty;
- per-source tube gates, tube valid ratio, and world delta norm;
- per-component consequence losses and consequence valid ratio.

Each run writes scalar records to `training_metrics.jsonl`. The strict
`06_inspect_training_signals.sh` report reads only that file and marks absent
signals `MISSING`. GroundedWorld inference can additionally save per-token
draft/final/delta/gates/tube arrays and supports a same-checkpoint
`disable_access` causal intervention.

Interpretation gates:

- real-vs-shuffled prior margin must be positive before claiming prior transfer;
- future temporal margin must be positive before Stage III;
- a nonzero reader gate alone is insufficient: compare B2-B1 and access/no-access;
- consequence labels with low valid ratio cannot support a physical-grounding claim;
- final trajectory gain with zero/unused world gates is not world-knowledge gain.

## Deliberate boundaries and unresolved external inputs

- No Driving-JEPA repository/checkpoint exists in this workspace. The cache
  runner therefore uses a lazy `module:function` adapter protocol and fails
  explicitly until local paths and an adapter are supplied. Generic V-JEPA is
  implemented only as a control, not silently relabeled as Driving-JEPA.
- The consequence cache runner defaults to the concrete local NAVSIM
  non-reactive provider in
  `starVLA/model/modules/grounded_world/navsim_consequence_provider.py`. It
  stores clearance/TTC/collision/lane-distance/progress/comfort separately and
  rejects aggregate EPDMS. The provider was exercised against the two-entry
  local `metric_cache_smoke`; a full train-split cache has not been generated.
- VGGT, Driving-JEPA, future-EMA cache generation, Stage I/II/III GPU smoke,
  full training, and NAVSIM evaluation have not been run in this turn.
- Stage III Phase A and Phase B are separate jobs. Phase B must load the Phase A
  checkpoint; it must not reload the pure baseline over Phase A.
- The current local evaluation wrapper now uses vendored NAVSIM-v2: navtest is
  one-stage and navhard is two-stage with explicit Stage-1/Stage-2/combined
  validation. Navhard processed metadata and full evaluator caches have not yet
  been generated.

Therefore, CPU algorithm/inference contracts and DLC launchers are ready.
**B1/B2 geometry jobs can be launched after the VGGT cache is validated. Formal
B3-B5 is not launch-ready until the local Driving-JEPA adapter/cache exists; B5
also requires a generated and validated full train-split consequence cache.**

Latest local verification: `pytest tests/grounded_world -q` passed 50 tests;
the combined `pytest tests/field2plan tests/grounded_world -q` passed 175 tests.
No full-checkpoint GPU forward/backward or formal training/evaluation was run.
