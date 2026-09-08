# PlanReg-WM-V2 review fixes taskbook

Source: user instructions in this conversation, 2026-09-08. No
`PlanReg_WM_V2_Review_Fixes_Package.zip`, attached Codex markdown, or nested
review-checks ZIP was found in the workspace/attachment search. Those attachments
have **not** been read. This document records the operative acceptance contract;
it does not substitute fabricated attachment contents.

## Scope and evidence

- Start: `6e1d9f8c6f91f5ddb2d04461554e0a7d27208084`.
- V1 reference: `d9ca73f3d61f059285fcbf12a5bc81177ee350d7`.
- Scorer: `valeoai/DrivoR@fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a`.
- Isolated branch `fix/planreg-wm-v2-review-20260908`; preserve original dirty
  worktrees. Local commits only; no push, PR, merge, full training or full Navtest.
- Preserve InternVL3-2B, single current front view, numerical ego/navigation,
  64 candidates, eight 0.5-second poses, language LoRA, rich memory, shared head,
  internal FP32 EMA, teacher forcing and differentiable rollout. No new external
  teacher, consequence heads, RL/CEM/ranking loss or deployment privileges.
- Read production V2 agent/action/targets/normalizers/motion/predictor/losses/
  backbone/language/memory/EMA/optimizer/runtime/checkpoint; parent backbone,
  register constructor, V1 targets, formal protocol, configuration and all
  initialization/cache/train/migration/export entries and scorer tests.
- Establish semantic red tests against actual production functions before fixes
  R1–R5. Missing imports/dependencies are not a red result. Reversion mutations
  must be detected; never loosen tolerances, xfail or change references to pass.

## R1: actual scorer integration

In `V2ActionDecoder.score`: detach physical proposals, flatten 8×3, embed,
decode against scene memory with optional padding mask, **then add ego**, then
call unchanged six-head scorer/aggregation. Preserve all labels, detach and
weights. Test the actual method against pinned upstream classes and ordering,
multiple seeds, nonzero ego, memory lengths 16/64 and unequal padding. Trim each
reference sample when upstream lacks masking. Compare six logits, log-PDM,
indices, scene/ego gradients and proposal/generator direct-gradient isolation.
Strict equality for identical paths, predeclared tolerances for different
kernels; pre-decoder ego mutation must fail. Distinguish component parity from
integration parity and hash the actual V2 action source.

## R2: long-2 labels and cache

Restore V1 `alpha=2*2/(8*9)`, `q_index=j+cumsum((j+1)*alpha)`;
nominal query times `(q_index+1)*0.5`:
0.52777778, 1.08333333, 1.66666667, 2.27777778, 2.91666667,
3.58333333, 4.27777778, 5.0. CubicSpline not-a-knot over **ten future
nodes**, no extra current node. Unwrap heading before interpolation, wrap after.
For timestamp jitter map q_index into actual ten future timestamps and spline
on those timestamps; final query is actual endpoint; extrapolate=False.
Check same log, eleven finite real poses, increasing times and declared
tolerance. Missing data gives long_valid=false, without extrapolation, log
crossing, duplicated endpoint or dropping scenes. Tests: all constant-speed
points, acceleration/nonlinear curve, jitter, angular wrap, incomplete/invalid
time/nonfinite; independent reference and uniform/linear mutation detection.
Version manifest AND per-scene long/cache; reject relabelled old cache, rebuild
in new root. Raw-GT statistics exclude long and require content/token/heading/
contract hashes for reuse.

## R3: learning-rate recipe

Reference GB32; actual GB=world×microbatch×accumulation. Peak is
min(reference_lr×sqrt(GB/32),cap), except language LoRA fixed 1e-5.
Planning/fusion/generator/scorer/predictor: 2e-4 / cap3e-4;
semantic queries: 1e-4 / cap1.5e-4; vision: 4e-5 / cap5e-5.
At GB128 peaks are 3e-4,1.5e-4,5e-5,1e-5 respectively.
Test actual load_config→optimizer groups for GB1/32/128, equivalent accumulation
layouts, caps, language exception, warmup/end and resume. AdamW .9/.999,
eps1e-8, matrix WD.01, bias/norm/query/register/gate/LoRA WD0, clip1,
5% warmup from .01×peak and cosine to .1×peak. No double scaling. YAML and
launchers must agree. Metadata records reference/actual/formula/cap/peak,
applied LR, recipe/schedule version. Old fixed-LR state is not new-recipe resume.

## R4: initialization

V2 actual register std=.02 and query std=.02; V1 default unchanged. FP32
conversion alone is not reinitialization. Order: VLM → new modules → versioned
shared init → teacher. No reinitialization after warm-start/resume. New schema
records architecture, std, module names/shapes/dtypes, recipe identity. Reject
old 1e-6 artifact even if shapes match; retain old files. Test tensor distribution,
bitwise Base/VQA common initial state and no load overwrite. Controls filter
the same named full parameter bank, not merely seed construction independently.

## R5: warm-start

Explicit attention←trajectory_decoder and trajectory_head←traj_head.4 mappings.
Output W_new=W_old/std; b_new=(b_old-mean)/std. Test physical head output on
same hidden, headings modulo2π; not whole-model parity. Coverage per module:
expected/copied/explicitly_new/excluded/unexpected_missing; required compatible
keys all hit. Correct hist_encoding columns for normalized ego input (or
explicitly exclude); test nonzero velocity/acceleration. New semantic/memory
remain initialized. No old EMA history/optimizer/scheduler; rebuild teacher.
Use real V1 checkpoint for real replay or mark BLOCKED, never synthetic as real.

## Boundary contracts

- Valid motion NaN/Inf, duplicate/backward/nonpositive time raises with
  batch/candidate location. Explicit invalid padding is safe before divisions
  and MLP; finite output/gradient. Causal physical kinematics before normalization,
  identical GT/ordinary candidates. One production interval coverage function
  shared by model and accumulation pre-count; independently calculated tests.
- Default statistics [8,3] training-GT stepwise_zscore, with explicit
  global_zscore and legacy_fixed_affine controls. Configurable serializable
  ego/motion physical scales with units/source, not mislabeled fitted Z-scores.
  Valid mask/std floor stored, padding NaN cannot corrupt unwrap/statistics.
- Audit real TTC label support before alleging errors. Preserve sentinel mask.
  Unequal-valid global reference test; if distributed/accumulated sentinel
  reduction unsupported, synchronize rejection across all ranks before backward.
  Generic support, if implemented, needs whole-optimizer-batch denominators,
  actual loss and gradient parity, not just all-reduced diagnostics.

## Framework regression

- TF/RO same predictor, matching first step; RO consumes predictions and late
  loss reaches early predictions/z0. No future target into RO, including
  nonconstant target perturbations. Third action cannot affect first two outputs.
  No indirect future path through semantic prefix. TF true-prefix validity and
  RO own-target validity separate. Detach/teacher-feedback/noncausal mutations
  must be detectable.
- Actual backbone/agent tests for query placement/masks, LoRA FP32/group/graph,
  tiles×16 memory, current/future identical layout, visual-only regression
  targets, padding perturbation. Registers are not a 4×4 image grid.
- One shared head, four .25 stage losses with gradients; GT/long independently
  matched. Only final proposals scored; scorer no direct generator-coordinate
  gradients; candidate permutation equivariance and joint scene/geometry
  permutation invariance.
- Same-batch L_traj, L_scorer, weighted L_WM gradients independently to vision,
  registers/readout, language, queries, with pairwise angles. At least two steps
  for LoRA zero-B initialization, not a total changed-parameter count surrogate.
- FP32 master sub-ULP accumulation vs FP64, dtype/save/resume, successful step
  only incl skipped AMP/accumulation. Normal teacher encodes only three future
  images; current teacher low-frequency drift. Establish unchanged target/loss
  before timing optimization.

## DDP, resume, deployment

Compare one big batch, microbatch accumulation, world2 unequal-valid, world2+
accum2, rank-empty and global-empty WM; compare loss AND gradients. Full
optimizer-batch denominator, no duplicate world/accum scaling. no_sync includes
forward/backward. Disable dropout for equivalent-split tests.
Real 8 continuous vs4+4 restores model, moments, master, scheduler, RNG,
Normalizer and sampler/progress. Separate run_step_limit from schedule_total_steps;
32-step stop must not shorten formal LR/EMA timeline. Explicit debug-short
schedule is not formal schedule validation.
Export eval/loaded-student current outputs equal; real AgentInput
compute_trajectory under traps on teacher/predictor creation, future read and
GT/PDM access. Declare remaining external VLM/tokenizer structural dependencies.

## Versions and evidence binding

Use architecture planreg_wm_v2.2 and separately version long/cache/shared init/
Normalizer/recipe/schedule/checkpoint. Reject old uniform-long caches, old 1e-6
init, undeclared old-LR resume, mismatched stats, GB4-as-GB128 permission and
smoke weights as formal initialization. Keep warm-start distinct from VLM-only.
Commit code/tests/config first as TESTED_CODE_COMMIT, validate that snapshot,
then report/docs-only DELIVERY_COMMIT, prove production/config/test hashes same.
Any later code change requires related tests/key replays rerun.

## Bounded real validation

Order: red/green+regression → GPU/DDP tests → new cache/init real32 optimizer
steps → 8vs4+4/export/AgentInput → resource-permitting GB128 profiles.
Use32–64 unique train scenes spanning stop/straight/turn/tile length, preferably
independent recorded drives (log segments ≠ independent drives). Fixed probe
does not update, preferably different drive; label training probe honestly.
Initial/final TF/RO per horizon, copy/no-action/mismatch controls, gradients,
representations, candidate mean/selected/oracle/regret, EP/TTC/calibration;
fixed candidate pool separates generator/scorer change. Declare training-PDM
reference vs official evaluation. No short-run improvement gate/WM causal claim.
Provide main/no-WM/TF-only/compact controls with common-init audit, no full runs.
At most3 profiles, each4 warmup+8timed optimizer steps, actual GB128. Keep all
modules/real PDM/future images; lower micro and increase accum instead of
disabling features. Record all-rank peak memory, throughput/median/p90, data/PDM/
teacher/VLM/backward times. Lock binds code/recipe/precision/hardware/max tiles/
logs. Unavailable resources → NOT_PROFILED, never fabricated formal-ready.

## Delivery and status

Required: REVIEW_FIX_MATRIX.md, ALGORITHM_CONTRACT_MATRIX.md,
SCORER_INTEGRATION_PARITY.json, LONG_TARGET_PARITY.json, LR_RESOLUTION.json,
INITIALIZATION_AUDIT.json, MIGRATION_COVERAGE.json, DDP_ACCUMULATION_PARITY.json,
GRADIENT_ROUTING.json, REAL_SMOKE_SUMMARY.json, RESUME_EXPORT_REPORT.json,
optional GB128_PROFILE.json, ARTIFACT_COMPATIBILITY.md. Update provenance,
implementation/validation matrices and commands. Each evidence item binds
production function, test ID, source commit/hash, command, input type,
PASS/FAIL/SKIPPED/BLOCKED and log.
ENGINEERING_READY needs fixes/semantics/real smoke/DDP/resume/export.
READY_FOR_FORMAL_TRAINING additionally needs full data/statistics/new shared
init/Base-VQA checks/real GB128 lock/fixed budget. PERFORMANCE=NOT_EVALUATED
without formal training/evaluation; no claim of exceeding91.3.
Deliver branch/base/tested/delivery commits, diffstat, tests, artifact identities
and remaining blockers. Review archive contains diff, complete changed source,
tests, taskbook, small logs and manifest/SHA; no weights/data/caches/secrets.
