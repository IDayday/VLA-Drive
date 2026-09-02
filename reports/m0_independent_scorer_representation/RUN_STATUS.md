# Independent scorer experiment status

Updated: 2026-09-02 UTC.

Unified evidence index:

`reports/m0_independent_scorer_representation/EXPERIMENT_EVIDENCE_INDEX.md`

The active No-VQA representation-learning route is locked at:

`reports/m0_independent_scorer_representation/NO_VQA_SCORER_REPRESENTATION_PLAN.md`

Canonical comparison of V8, the reconstructed Stage2 correction, and No-VQA
(checkpoint hashes, complete FP32 Navtest metrics, training settings, paired
log-bootstrap intervals, and evidence paths):

`reports/m0_independent_scorer_representation/M0_V8_CORRECTED_NOVQA_COMPARISON.md`

Do not reconstruct those three runs from this older rolling status section;
use the canonical comparison above.

## Resource contract

- The user subsequently confirmed that `vla-zt` and `vla-zt2` were free and
  explicitly authorized both for this scorer campaign; both 8-GPU hosts are
  now running the two primary No-VQA fixed-bank waves.
- `rl-zt3` was unavailable at the latest connectivity check. On `rl-zt4`,
  this campaign uses only GPUs 1/3/5/6/7, which were verified idle immediately
  before launch; the pre-existing work on GPUs 0/2/4 was not touched.
- No unrelated process was stopped or preempted.

## Complete Navtest results

Every row uses all 12,146 Navtest scenes, 136 segment logs, the same 64 frozen
M0 proposals, FP32 inference, and zero invalid scenes. Official candidate
scores are joined only after selection.

| Selector | Navtest PDMS | Delta from public M0 |
|---|---:|---:|
| Public M0 scorer | 0.909594 | 0.000000 |
| Factor-heavy, epoch 6 | 0.895126 | -0.014468 |
| Factor-heavy, final saved epoch 9 | 0.895823 | -0.013771 |
| Three-default-seed equal-score ensemble | 0.897006 | -0.012588 |
| Factor-only all-64, epoch 3 | **0.897539** | **-0.012055** |
| Factor-only all-64, validation-best epoch 9 | **0.900880** | **-0.008714** |

The current independent method has not improved public M0 on Navtest. Its
best change is a real improvement over earlier independent checkpoints, not a
new test-set best.

## Running on rl-zt4

| Run | GPU | State | Purpose |
|---|---:|---|---|
| low-res factor-only all-64 | 2 | epoch 4 complete; validation 0.923593 | isolate conflicting-loss effect |
| high-res 960-token factor-only (`v2_clean`) | 7 | training | test whether spatial visual detail is limiting |
| low-res factor-only final+epoch-3 replay | 5 | training | test fixed proposal-distribution overfitting |
| M0-native four-view cache, shards 0--3 | 0/1/3/4 | exporting 103,288 trainval scenes | replace DINO with released M0 vision features |
| Q-Former + current-actor auxiliary target | 6 | loading/training | test whether explicitly supervised dynamic queries improve ranking |

The high-resolution run is
`m0_independent_dino_highres960_factoronly_keep48_seed2_v2_clean`. Its manifest
has been checked against the live process: batch 20, evaluation batch 40,
48 sampled training candidates, top-16 fine configuration, and 10 epochs.

## Run-integrity incident

Two previously scheduled wait wrappers were initially mistaken for exited
processes because they later replaced their shell command with Python. A second
pair briefly launched against the same output names. The duplicate PIDs were
stopped before any epoch completed. The original multi-replay process and its
manifest agree and remain valid. The first high-resolution directory had a
manifest/process mismatch, so that process was stopped and the directory is
excluded from all results; it was not deleted. High resolution was restarted
in the separate `v2_clean` directory with a verified manifest.

## Current diagnosis

- The scorer uses current vision: cross-log scene shuffling lowers held-out
  PDMS by 0.029946 and zeroing scene tokens lowers it by 0.033757.
- Factor-only training transfers positively relative to the earlier
  independent scorer, but still loses mainly ego progress and DAC on Navtest.
- Equal-score ensembling improves safety factors but further lowers progress,
  so ordinary variance reduction does not solve the calibration problem.
- High-resolution perception and proposal-distribution replay are the two
  currently running causal checks.

## M0-native scorer-private representation

An 8-scene smoke export passed on `rl-zt4`. It uses the released M0 checkpoint
and its own frozen InternVL visual encoder on current `CAM_F0/L0/R0/B0` images.
The cache has 80 fixed camera-block slots of width 1536 (48 valid in the smoke
sample), contains no proposal, future, evaluator score, or factor field, and
records checkpoint SHA256
`7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d`.
Four full-data shards are running on `rl-zt4` GPUs 0/1/3/4 at approximately
3.6 scenes/s/GPU.

A pinned-code follow-up (`0c3f4b7`) is queued behind those exporters. After a
full 103,288-token cache validation, GPU 0/1 will train a parameter-matched
pair with and without current-actor auxiliary supervision using the released
six-head BCE, while GPU 2/6 will run the corresponding continuous-progress
regression pair. All four use the same factor-ranking auxiliary. GPU 3/4 will
export the complete 12,146-scene M0-native Navtest observation cache. This
queue does not reserve GPU memory while the current exporters are active. The
earlier idle watchers were replaced before they started any training or
created any output.

The previous dynamic/static/signal query banks had different parameters but
no semantic supervision. A new optional training-only auxiliary head now
teaches the dynamic queries to recover the 16 nearest current actors in the
current-ego frame (presence, type, position, velocity, heading, and size).
These labels never enter the model forward or deployment artifact. The target
store covers 45,377 balanced scenes; unmatched replay rows are explicitly
masked. A Q-Former control is running on GPU 6, and the same loss will be tested
on the M0-native four-view cache after export completion.

All repository tests pass: 194 passed. Warnings are dependency deprecations and
the pre-existing Shapely numerical warning; there are no test failures.

The low-resolution factor-only all-64 run reached a best held-out-log PDMS
of 0.931604 at epoch 9 (the public M0 selector on the same fold is 0.951612).
Its final validation-selected factor checkpoint reaches complete-Navtest PDMS
0.900880, an improvement of +0.003341 over the previously tested epoch 3 but
still -0.008714 below public M0. The physical-log bootstrap 95% interval for
the delta from public M0 is [-0.013556, -0.003615].

Source inspection identified an important training-semantic difference: the
released EpisodeDrive loss uses ordinary BCE for NOC/DAC/TTC, whereas the
independent campaign so far upweighted rare violations by 10. The resulting
Navtest signature (better safety, worse progress/DAC) is directionally
consistent with that change. A full 2x2 Q-Former diagnostic now tests
unweighted BCE with factor-ranking and current-actor supervision independently.
The inspection also found that the independent loss had excluded progress from
BCE and replaced it with `2 * SmoothL1`; released EpisodeDrive includes the
continuous progress target directly in the same six-head BCE. The new explicit
`episode_drive_bce` mode is unit-tested against the source-equivalent formula,
while the former behavior remains available only as a named ablation.

## Q-Former actor supervision and conservative-gate Navtest

With the frozen released 16-token Q-Former observation, the continuous-progress
factor scorer reached held-out-log PDMS `0.949755` at epoch 3 with current-actor
auxiliary supervision, versus `0.947291` for its no-actor control at the same
epoch.  The gain came mainly from ego progress and DAC.  The same actor model's
conservative reference gate reached `0.954641`, or `+0.003029` over Base on the
held-out fold with a positive physical-log bootstrap interval.  The no-actor
gate reached `0.955196`, or `+0.003585`.

Both validation-positive gates were immediately evaluated on all 12,146
Navtest scenes.  The result reversed sign: the actor gate obtained `0.894955`
(`-0.014639` from public M0, 95% CI `[-0.017602, -0.011590]`) and the no-actor
gate obtained `0.899366` (`-0.010228`, 95% CI
`[-0.012328, -0.008003]`).  Both audits pass the 64-candidate completeness,
oracle identity, and selected-score reconstruction checks.  These gates are
therefore rejected; the positive trainval result is not a deployable planning
improvement and provides direct evidence of a severe validation-to-Navtest
shift for post-hoc switching policies.

The no-actor continuous-progress factor ranker itself later became a strict
held-out promotion at epoch 6: validation PDMS `0.953213`, delta `+0.001601`,
with bootstrap lower bound `+0.000010`.  Its complete all-64 Navtest result is
`0.901797`, delta `-0.007797` with 95% CI
`[-0.010752, -0.004674]`.  This audit also passes all 12,146-scene completeness
and reconstruction checks.  Correcting factor weighting and adding a ranking
objective therefore improves the independent Q-Former control, but does not
close the public scorer gap or support a 0.93 claim.

An all-physical-log refit path is now available for validation-selected
architectures.  It accepts only an artifact whose held-out physical-log
bootstrap lower bound is strictly positive, then locks its model configuration,
loss weights, optimizer settings, seed, selected stop epoch, and original
scheduler horizon.  The refit itself reports no validation metric and cannot be
used as a new selection artifact.  This permits a pre-specified 103,288-scene
refit without using Navtest feedback for model or epoch selection.

The current-actor auxiliary ranker later reached a strict held-out promotion
at epoch 7: PDMS `0.954557`, delta `+0.002945`, with physical-log bootstrap
95% interval `[+0.000697, +0.004724]`. Its complete 12,146-scene Navtest audit
reversed sign again: PDMS `0.897208`, delta `-0.012386`, interval
`[-0.016207, -0.008677]`. The audit has 64 candidates per scene, zero invalid
scenes, exact selected-score reconstruction, and the unchanged M0 candidate
bank upper bound `0.984112`. Current-actor supervision therefore improves the
held-out trainval fold but worsens the present Navtest selection; it is not a
deployable improvement.

Two additional M0-native four-view experiments are queued on `rl-zt3` GPUs
5/6 behind completion of the immutable 103,288-scene observation cache. Both
score all 64 candidates and retain the released six-factor BCE auxiliary. One
optimizes regret-weighted pairwise ranking; the other adds listwise, top-set,
expected-regret, and top-regret objectives. These runs test direct candidate
ranking separately from the four parameter-matched factor-only controls on
`rl-zt4`.

A second queued design keeps the same M0-native scorer-private visual encoder
but makes the released M0 factor logits and aggregate score an explicit
calibration anchor. Its factor-correction and utility-correction heads are
zero initialized, so before training its 64-way output and selected candidate
are exactly identical to public M0. It then learns candidate-relative factor
and utility residuals from the current four-view representation. This path
uses M0 inference tensors only—no DrivOR representation or weight, future
annotation, MetricCache value, or official PDM input. Hybrid-residual and
factor-residual variants are queued behind the direct-ranking controls on
`rl-zt3` GPUs 5/6. The complete repository suite passes `198` tests, including
zero-init Base identity, candidate-permutation equivariance, inference-schema
checks, and finite end-to-end residual training loss.

`rl-zt4` is now explicitly available for this campaign. Its GPUs 0/1/3/4 are
finishing the immutable 103,288-scene released-M0 four-view cache, GPU 5 is
running the validation-locked no-actor all-log refit, and the existing watcher
will use GPUs 0/1/2/6 for the four controlled scorer runs plus GPUs 3/4 for the
12,146-scene current-only Navtest cache. The cache exporter remains unchanged
near completion rather than being repartitioned and duplicated.

The M0-native Navtest path now supports both independent rankers and the
Base-calibrated residual architecture. A strict promotion manifest reads the
artifact selected on held-out physical logs and requires both positive mean
delta and a bootstrap 95% lower bound above zero. Only promoted artifacts are
evaluated. Inference consumes current M0 visual tokens, current state, frozen
M0 proposals, and—only for the calibrated residual—M0's own deployable factor
logits and scorer values. The official `[12146,64,7]` candidate matrix is
joined only after all selected indices are fixed. Full audits are serialized
on an otherwise idle GPU, validated with the reusable Navtest skill, and
compared scene-by-scene with public M0. The complete repository suite now
passes `200` tests.

The validation-locked no-actor all-log refit has now completed and passed the
strict 12,146-scene audit. It reaches Navtest PDMS `0.897837`, delta
`-0.011757` from public M0 with log-bootstrap 95% interval
`[-0.015066, -0.008519]`. Its unchanged candidate-bank upper bound is
`0.984112`; the loss is selection-only. The refit increases progress by
`+0.003679` but degrades collision by `-0.005475`, DAC by `-0.008315`, TTC by
`-0.019595`, and DDC by `-0.005022`. Training on all physical logs therefore
does not repair the held-out-to-Navtest reversal and is rejected.

The separately validation-locked current-actor all-log refit also completed
and passed the same strict audit. It reaches Navtest PDMS `0.895842`, delta
`-0.013752` from public M0 with log-bootstrap 95% interval
`[-0.017315, -0.010182]`. Its factor deltas are collision `-0.006957`, DAC
`-0.009880`, TTC `-0.022641`, DDC `-0.003828`, and progress `+0.004004`.
Together, the two pre-specified all-log refits show that simply exposing a
post-hoc fixed-bank scorer to all 103,288 training scenes does not remove the
distribution shift: both trade measurable safety for small progress gains.

M0-native deployment artifacts now use the real
`M0NativePrivateScorerAgent`, not a cache-only surrogate class. The online
agent obtains F0/L0/R0/B0 from the current frame, extracts private tokens with
the released M0 vision encoder, and supports either the independent ranker or
the Base-calibrated residual scorer. The artifact pins the Base checkpoint,
source-ranker SHA256, camera order, crop policy, pooling grid, wrapper chain,
and exact scorer class. Every promoted architecture is scheduled for a
four-scene same-device online/cache parity test with `1e-6` tolerance in
addition to full Navtest. Remote validation now calls the repository's audited
validators directly instead of assuming `/root/.codex/skills` is mounted on
every worker. The complete suite passes `202` tests.

The immutable four-view trainval cache is now complete for all 103,288 scenes.
`rl-zt4` GPUs 0/1/2/6 are training the four factor-loss/current-actor controls,
GPU 5 is training the Base-calibrated hybrid residual, and GPUs 3/4 are
exporting the matching full Navtest observation cache. `rl-zt3` GPUs 5/6 run
the two direct-ranking controls and GPU 3 runs the Base-calibrated factor
residual. Strict full-Navtest promotion remains serialized on `rl-zt4` GPU 7;
no job uses `vla-zt` or `vla-zt2`.

A multi-stage proposal-replay residual is queued to take over `rl-zt4` GPU 0
when its present control finishes. It trains on both the frozen public-final
and epoch-3 M0 proposal banks while sharing the same M0-owned current-only
four-view observation. Checkpoint selection and promotion are computed only on
the predeclared `public_base` held-out physical logs; combined-source metrics
are diagnostic and cannot select an artifact. This directly tests whether the
joint scorer's exposure to changing proposal distributions is important,
without changing the final inference inputs. Multi-source selection semantics
are unit-tested, and the complete suite passes `203` tests.

The M0-native Navtest observation cache is complete and independently scanned:
96 chunks and two manifests contain exactly 12,146 unique scene tokens from
136 segment logs. Every row has `[80,1536]` FP16 visual slots and an 11-value
FP32 current-state vector with finite values. All four current cameras decoded
successfully (three crops per camera); the cache contains no future, proposal,
official-score, or factor key. The machine-readable audit is
`M0_NATIVE_NAVTEST_CACHE_AUDIT.json`.

## M0-native Base-calibrated residual result

Epoch-0 factor and hybrid residual snapshots were calibrated with a nested
physical-log protocol: 30 logs selected a deployment switch policy and 31
disjoint logs performed the promotion test. Factor residual improved the
promotion half by `+0.006047` (95% CI `[+0.003923,+0.007820]`); hybrid improved
it by `+0.005186` (`[+0.003069,+0.006791]`). Both therefore underwent the
mandatory complete Navtest evaluation without using Navtest for selection.

Both gains reversed decisively. Factor residual reaches Navtest PDMS
`0.897361`, delta `-0.012233` from public M0 with log-bootstrap interval
`[-0.014750,-0.009766]`. Hybrid residual reaches `0.894111`, delta `-0.015483`
with interval `[-0.018801,-0.012222]`. Both use 12,146 scenes, 136 segment
logs, all 64 unchanged M0 candidates, FP32 selection, and zero invalid scenes.
Repository audit, selected-score reconstruction, candidate-oracle identity,
and four-scene online/cache parity all pass; maximum online/cache score error
is `2.3842e-7` and selected indices match exactly.

The factor residual increases ego progress by `+0.004914` on Navtest but loses
collision `-0.006916`, DAC `-0.008315`, DDC `-0.007328`, and TTC `-0.021653`.
Hybrid has the same, slightly worse signature. Thus nested thresholding and a
Base anchor do not solve the trainval-to-Navtest safety shift. Both models are
rejected; the next scorer representation must receive explicit dynamic-future
or interaction-risk supervision rather than only aggregate/factor labels.

## Shared logged-future representation auxiliary

The next pre-specified experiment adds a training-only, candidate-independent
shared actor-future auxiliary to the M0-owned dynamic query bank. It predicts
eight future horizons for the 16 actors selected at the current frame in the
current-ego coordinate system. Its fields are presence, object type, position,
velocity, heading, length, and width. The future head receives no proposal and
therefore cannot encode candidate index or template. At inference the head is
discardable: scoring still receives only current F0/L0/R0/B0 M0 tokens,
current ego/navigation state, frozen M0 proposals, and the released M0
deployable logits/scores. No future tensor, evaluator value, or official score
is an inference input.

The immutable derived target table contains `45,378` trainval scenes,
`45,377` valid supervision rows, eight horizons, 16 fixed current-actor slots,
and eight actor fields. Its source and array SHA256 values are pinned in its
manifest; the one absent target is the same scene already unavailable in the
Gate-C preflight. The loader verifies provenance, hashes, coordinate frame,
shape, coverage, and the training-only contract before training. Candidate
permutation invariance, invalid-scene masking, target ordering, inference
signature, and gradient flow are covered by tests. The full repository suite
passes `207` tests. Two factor-residual runs with auxiliary weights `0.5` and
`1.0` are assigned to `rl-zt4` GPUs 3/4; selection remains confined to the
predeclared physical-log validation split and any promoted artifact must still
pass the complete 12,146-scene Navtest protocol.

The auxiliary-only runs are paired with a stricter
`SharedFutureFactorized` implementation. The M0-owned dynamic queries predict
one candidate-independent actor future; a differentiable relabeler then
transforms that same prediction against every proposal and computes signed
box clearance, soft collision probability, soft TTC, corridor occupancy, and
nearest-actor relative position/velocity at all eight horizons. A temporal
consequence token is injected through a zero-initialized scalar gate, keeping
the initial Base-selected trajectory exactly unchanged. The shared prediction
head is called once per scene and cannot receive candidate geometry; candidate
permutation equivariance and collision/clearance physical ordering are tested.
The deployable forward signature still contains no logged-future input. The
complete suite passes `211` tests.

Every shared-future artifact now has a held-out-log prediction audit independent
of the ranking metric. It compares actor presence and metric-space position,
velocity, heading, and size errors against both constant-position and
constant-velocity baselines on the 9,052 aligned validation scenes. The model
forward is completed before any future target is read for scoring, and the
report records the artifact, target, split, and current-actor table hashes.
These audits share the serialized evaluation-GPU lock with Navtest. The full
suite passes `212` tests.

A trainval-only prevalence audit covers all 103,288 replay scenes and
6,610,432 candidates. In the training split, exact/thresholded failure rates
are NOC `3.06%`, DAC `5.42%`, DDC `2.83%`, TTC `8.75%`, and comfort `5.65%`.
This confirms severe safety-class imbalance without consulting Navtest. A
predeclared weight of `5.0`—well below inverse-frequency weighting—is therefore
added as a separate factorized run while the source-equivalent `1.0` runs stay
unchanged. The weighted loss now covers DDC as well as NOC/DAC/TTC. Its input
evidence and split hash are recorded in `TRAIN_FACTOR_PREVALENCE.json`; the
complete suite passes `213` tests.

The auxiliary-only runs have now completed epoch 0. Weight `0.5` obtains
held-out-log PDMS `0.928984` (`-0.022628` from Base), while weight `1.0`
obtains `0.931785` (`-0.019827`). The corresponding no-future factor residual
at epoch 0 is `0.929651`; future supervision therefore changes early learning
but does not by itself produce a promotable selector. Both runs continue so
the prediction audit can measure whether their actor future improves across
epochs. The factorized and class-balanced factorized runs remain the decisive
tests because they use the predicted future in candidate scoring.

Review of the factorized geometry exposed a mask bug before launching its next
version: at an all-empty actor horizon, the old soft-min normalized across
zero-filled padding slots and could report a fictitious close obstacle. The
running immutable v1 jobs are retained as baselines. The corrected relabeler
includes actor presence in nearest-actor and TTC weights and explicitly falls
back to 40 m clearance, 10 s TTC, and zero relative state when no actor is
present. A new training-only candidate-relative loss directly supervises the
clearance, soft collision, TTC, corridor occupancy, and relative actor state
recomputed from the one shared logged future. The deployable forward signature
is unchanged and still receives no future tensor. Empty-mask behavior,
invalid-scene masking, gradient flow, and candidate equivariance are covered;
the complete suite passes `215` tests. This corrected direct-consequence run is
queued for the first available authorized GPU without terminating any existing
job.

The coordinate audit then confirmed that every M0/NAVSIM proposal pose is a
rear-axle pose, not a vehicle-center pose. The deployed local Pacifica has
`front_length=4.049 m`, `rear_length=1.127 m`, and `width=2.297 m`, so its
geometric center is `1.461 m` ahead of the rear axle with half-length
`2.588 m` and half-width `1.1485 m`. The initial factorized approximation had
treated the rear axle as the center and used `2.45/1.0 m` half extents. That is
a systematic collision/clearance/TTC alignment error, not a hyperparameter
effect. The corrected implementation transforms every proposal rear axle to
the exact footprint center and finite-differences center positions (including
heading-induced center motion) for relative velocity. Constants are asserted
against the installed official `get_pacifica_parameters()` result. The already
launched mask/direct-loss v2 is retained as an immutable ablation; the exact
rear-axle v3 will use a new output directory on the next naturally released
authorized GPU. The complete suite passes `216` tests.

An immutable epoch-0 auxiliary-only snapshot was evaluated on all 9,052
aligned held-out scenes before reading any future target. Its actor-presence
BCE is `0.3461` (better than the all-present baseline `1.1488`), but its
position MAE is `26.982 m`, versus `3.558 m` for constant position and
`0.804 m` for constant velocity from the logged current actor state. Velocity
MAE is `2.012 m/s` versus `0.468 m/s` for the baseline. Thus the early shared
head has learned occupancy/type signals but not a usable metric actor future;
injecting it into scoring cannot yet be interpreted as a successful world
model. The final eight-epoch audit remains required before rejecting the
target outright.

The next parameterization addresses that measured failure directly. The same
16 dynamic queries first predict current actor presence, type, position,
velocity, heading, and size from the current four-camera tokens. A fixed,
differentiable constant-velocity extrapolation is then formed in normalized
current-ego coordinates, and zero-initialized horizon heads predict only its
residual. Logged current actor annotations supervise this intermediate state
during training but never enter inference. The current and future stores have
exactly matching 45,378 scene-token order and 16 slots; across 4,404,241
jointly valid actor-steps, object type agrees `100%`, confirming slot identity.
This makes constant velocity an architectural prior rather than an oracle
inference input. Training-batch alignment, metric normalization, zero-residual
initialization, future candidate-independence, and inference-input boundaries
are covered by tests. The complete suite passes `219` tests.

The held-out future audit now also reports the current-actor head separately:
presence, type, and metric-space state errors at horizon zero are measured
before the constant-velocity rollout. This separates a current visual
perception failure from a future-dynamics failure without adding any input to
the deployable model or changing the immutable training runs.

With `rl-zt4` newly authorized, a second constant-velocity-residual run was
started on its otherwise idle GPU 7 without stopping the seven existing
experiments. Relative to the queued source-equivalent run, this predeclared
variant raises current-actor supervision from `1.0` to `5.0` and shared-future
supervision from `0.5` to `1.0`; candidate-relative and safety-negative weights
remain `1.0` and `5.0`. It directly tests whether the measured bottleneck is
insufficient current-scene actor localization, while preserving identical
proposals, split, seed, optimization steps, and strict Navtest promotion gate.

## Released M0 map/agent branch audit

The locally deployed public M0 artifact does not contain a pretrained map or
agent consequence branch that a new scorer can simply reuse. Its resolved
configuration sets `agent_pred`, `area_pred`, `bev_map`, and `bev_agent` to
`false`. The 4.0 GiB checkpoint contains 1,323 state-dict entries and zero keys
matching `pred_col_agent`, `pred_area`, `map_head`, or `_agent_head`.
Furthermore, the deployed `EpisodeDriveLoss.score_loss` initializes the agent
classification, agent regression, and area losses to literal zero and returns
them unchanged. The source tree therefore exposes dormant optional modules,
but the released weights do not provide their learned representation. Current
actor, shared-future, and candidate-relative heads in this campaign are newly
trained M0-owned modules rather than reuse of hidden public auxiliary weights.

## Masked-consequence conservative-policy Navtest

The immutable epoch-0 masked-consequence artifact has SHA256
`30b91c9444ada03d9b5ba4be0f481f9080190223af9e17f3b0c1f7c955b17df3`.
A nested physical-log calibration selected a predeclared conservative policy
with residual scale `0.75`, relative-factor safety gating, and safety floor
`0.85`. On the calibration half it improved PDMS by `+0.006234`; on the
disjoint promotion half it improved `0.953332 -> 0.958610`, with log-bootstrap
95% CI `[+0.002651, +0.007402]`. The derived ranker artifact has SHA256
`86d97ca0d95c3bef2f7dc1f0eba5101052f5f44dd4d04312c81791c28289e9dc`.

The required strict full Navtest audit then rejected this policy. Across all
`12,146` scenes, `136` physical logs, `64` candidates per scene, FP32, and zero
invalid scenes, selected PDMS is `0.893988`, versus the released M0 reference
`0.909594`: delta `-0.015606`, with log-bootstrap 95% CI
`[-0.018699, -0.012138]`. Best-of-64 remains exactly `0.984112`, proving that
the proposal bank did not change. Progress improves by `+0.002482`, but NOC,
DAC, DDC, and TTC change by `-0.007822`, `-0.009962`, `-0.006834`, and
`-0.024535`; those safety losses dominate the mean. There are 5,040 improved,
1,650 degraded, and 5,456 tied scenes, illustrating why win count alone is an
unsafe promotion criterion. The offline audit passes candidate-order,
selected-score reconstruction, oracle-identity, and public-reference parity.
Ray reached the configured 95% host-memory threshold after the four-scene
online replay had been written, but before the final check ran. A locked FP32
check on the same `rl-zt4` GPU then passed: proposals are bit exact, selected
indices match for all four scenes, and maximum score error is `2.38e-7`, below
the strict `1e-6` tolerance. The CPU diagnostic is retained separately; its
identical selections but `9.94e-5` score error demonstrate why cross-device
floating-point output is not accepted as the online/cache parity gate.

This sign flip rules out threshold calibration of the existing pooled feature
head as the main solution. The next representation test retains the same M0
InternVL checkpoint and current-only four-camera inputs but changes spatial
pooling from `2 x 2` to `4 x 4` per crop, increasing fixed visual tokens from
80 to 320. Four trainval cache shards are queued across authorized
`rl-zt3`/`rl-zt4` GPUs, with two matching Navtest shards and a frozen-generator
training/evaluation chain behind them. No `vla-zt` or `vla-zt2` GPU is used,
and no existing task is terminated.

## Released-context plus scorer-private dual stream

A source audit found a more fundamental representation confound in the first
M0-private experiments: when the four-camera raw-token cache was enabled, the
replay loader replaced the released M0 `scene_features` and `ego_features`
instead of preserving them as a second stream. The residual still received the
released factor logits and aggregate score, but it could no longer attend to
the 16 task-contextualized Q-Former scene tokens used by the public scorer.
Consequently, a failure of the replacement model could not establish that an
M0-owned private scorer representation is ineffective.

A predeclared single-variable dual-stream mode now retains the frozen released
M0 scene tensor `[B,16,256]` and ego tensor `[B,1,256]`, while continuing to
encode the current four-camera raw InternVL tokens independently. Candidate
features cross-attend to the released context through separate scene/ego
projections, modality embeddings, and a learned zero-initialized gate. The
proposal bank, Base factors/scores, labels, physical-log split, loss weights,
seed, and optimization schedule remain unchanged. Old artifacts default to the
original path, and the zero-initialized residual still reproduces the Base
selection exactly before training.

The same tensors are now wired through replay training, conservative
calibration, cached full-Navtest evaluation, the real online
`M0NativePrivateScorerAgent`, shared-future diagnostics, and same-device online
parity. No external-model feature, future annotation, or evaluator value was
added to inference. Candidate-permutation equivariance, token-aligned replay
joins, current-only forward signatures, and zero-initialized identity are
covered by tests. The complete suite passes `221` tests. `rl-zt4` is included
in the scheduler, but its current eight-task queue and approximately `942 GiB`
host-memory use are respected; this run starts only after a natural task exit
or sufficient memory release.

## Active No-VQA epoch-35 scene-token campaign (2026-09-02)

The current primary path no longer waits on the older public-M0/four-camera
experiments. It fixes the locally trained No-VQA epoch-35 generator and uses
the exact checkpoint/config pair to cache its own 16 current-scene tokens,
ego token, 64 proposals, factor logits, and Base scores over all `103,288`
legal trainval scenes. Eight A800 shards are active under launcher PID
`448227`; the first synchronized progress sample showed all GPUs at 99–100%
utilization and all shards at `11/101` chunks. Four 12-process CPU workers
attach labels in a physically separate tree.

The training watcher PID is `455360`; the validation-calibration/Navtest watcher
PID is `459362`, and the `training-vla-zt2` transfer/launch watcher is `459364`.
They first require a strict cache audit,
then starts eight log-disjoint full-data scorer-private runs and the matching
FP32 full-Navtest candidate-bank scoring job. The primary hybrid model is run
with seeds 2, 11, and 23; no-actor, direct-only, factor-only, deeper-refiner,
and shared-future auxiliary variants are fixed before validation results are
read. No DrivOR representation or weight enters any of these paths.

Each frozen learned ranker also receives a validation-only conservative-policy
calibration. The 61 held-out physical logs are balanced into disjoint halves:
one chooses residual scale, switch penalty, and predicted safety gate; the other
is the promotion set. Both raw and calibrated artifacts are eligible only under
a positive physical-log bootstrap lower bound, and every eligible artifact is
scheduled for complete Navtest rather than selected using Navtest outcomes.

An independent second wave is queued for the eight idle A800 GPUs on
`training-vla-zt2`. It compares M0-candidate-hidden-only residuals at top-64,
top-16 and top-8 against the full private representation at Base top-4/8/16.
This is a pre-Navtest representation/support ablation; the immutable cache is
transferred only after the local 103,288-scene verifier passes, and no existing
remote task is stopped.

The deployment adapter now supports the exact lightweight path used in
training: one frozen No-VQA forward produces `language_feature [B,16,256]`
and `ego_feature [B,1,256]`, plus the original scorer-attention
`scorer_candidate_features [B,64,256]`. Every active variant fuses this frozen,
current-only M0 candidate representation with its scorer-private representation;
no future tensor or external model enters inference. The new scorer consumes
these tensors without a second vision-tower pass. Packaging verifies source/base
checkpoint SHA identity. The cache verifier now requires both export flags and
the exact 64-by-256 candidate-feature shape. The scene-token packaging,
Base-preserving zero initialization, candidate-feature path, and candidate
permutation tests pass. Full commands, SHA values, variants, and cache evidence are recorded in
[NO_VQA_SCORER_REPRESENTATION_PLAN.md](NO_VQA_SCORER_REPRESENTATION_PLAN.md).

## No-VQA scene-token 首个完整 Navtest 与 wave-4（2026-09-02）

matching No-VQA candidate matrix 已完成 136/136 segment logs，并通过
12,146-scene、64-candidate、0-invalid validator。Base PDMS 为 `0.911493`，
Best-of-64 为 `0.983129`。Base-Top-4 oracle 仅 `0.929307`，因此不能满足
`>0.93`；Base-Top-8 oracle 为 `0.940399`。

wave-2 Top-8 frozen ranker 在 30-log calibration / 31-log promotion 的独立
后半集得到 `+0.002634`，95% CI `[+0.000059,+0.005095]`，因而进入完整
Navtest。严格测试结果反而为 `0.909580`，比 matching Base 低 `-0.001913`，
95% CI `[-0.003780,-0.000087]`。progress 增加 `+0.008555`，但 NOC、DAC、
DDC 和 TTC 分别下降 `-0.003046/-0.003540/-0.002676/-0.007410`。该 artifact
被明确判为失败，不用于后续调参或成绩声明。

源码层面的下一组固定对照已经在 rl-zt4 GPU 0/2/4 启动：给 shortlist 排序
增加显式 oracle-vs-rest Top-1 regret loss，并比较 factor/safety loss 在全部
64 候选和实际 Base-Top-8 shortlist 上的监督范围。旧默认保持不变，新增路径
有单元测试覆盖。完整证据见
[NO_VQA_E35_SCORER_CAMPAIGN_RESULTS.md](NO_VQA_E35_SCORER_CAMPAIGN_RESULTS.md)。

随后 wave-2 最终 campaign 已完整收口：11 个 validation-effective artifact
全部完成 Navtest 和在线/缓存 parity，coverage 为 11/11、0 invalid。最佳测试
值仅 `0.910824`，仍比 Base 低 `-0.000669`；其余结果介于 `0.906954` 与
`0.910437`。11 个 validation delta 全为正，但 11 个 Navtest 点估计全为负，
没有任何结果超过 `0.93`。这已排除“仅靠冻结 No-VQA candidate hidden、
Base-Top-K 和事后保守阈值即可泛化提升”的假设。

## 最新收口与在运行路径（2026-09-02 08:12 UTC）

- wave-3 三个 validation-selected 架构已按锁定 epoch 在全部 162 条物理日志、
  103,288 scenes 上 refit，并完成 3/3 严格完整 Navtest。最佳为 `0.910348`，
  相对 No-VQA Base `-0.001145`，95% CI `[-0.003195,+0.000996]`。全日志 refit
  缩小了 validation-to-Navtest 损失，但没有产生正增益。
- wave-4 的 5/5 promoted artifact 已收口；Top-regret calibrated 版本为
  `0.911414`，相对 Base `-0.000080`，是目前最接近 Base 的自研 scorer，但不构成
  改善，更不构成 `>0.93`。
- wave-4 Top-regret 已进一步按锁定 epoch 在全部 162 条物理日志上 refit，并
  完成严格 Navtest：`0.911192`，相对 Base `-0.000302`，95% CI
  `[-0.001954,+0.001243]`。12,146 scenes、136 logs、64 candidates、0 invalid
  与在线/cache parity（最大误差 `2.38e-7`）全部通过；全日志 refit 仍未翻正。
- wave-5 的 Base-relative conservative-reference head 已通过 152 项相关测试，
  正在 `training-vla-zt2` 八卡运行。它保留精确 Base fallback，只在相对收益、
  安全退化与可信度 gate 同时通过时切换。
- No-VQA epoch-35 原生四视角特征 4-scene smoke 已成功。全 103,288-scene
  trainval cache 正在本机八卡导出，随后自动导出完整 Navtest cache；预注册的
  wave-6 raw-multiview scorer 在 `training-vla-zt2` 等待 cache 和 GPU 自然空闲，
  不会中断 wave-5。
- DrivOR 始终只用于候选库差距分析；任何在训练、验证或部署中的 M0 新 scorer
  都没有读取 DrivOR 表征、权重或打分。

完整 consolidated 结果与当前命令边界见
[NO_VQA_E35_SCORER_CAMPAIGN_RESULTS.md](NO_VQA_E35_SCORER_CAMPAIGN_RESULTS.md)。

## wave-5 完整 Navtest 收口（2026-09-02 08:46 UTC）

- 8/8 validation-promoted conservative-reference artifact 已完成完整 FP32
  Navtest：12,146 scenes、136 logs、64 candidates、0 invalid，且 8/8 在线
  agent/cache parity 通过，proposal/score 最大误差均为 `0`。
- 最佳值为 `0.911499`，相对 matching No-VQA Base `0.911493` 仅
  `+0.000006`；它只改变并改善 1 个 scene，其余 12,145 scenes 与 Base 相同，
  因而不构成有意义的 scorer 改进。
- validation 提升最大的 q50 Top-16 模型为 `+0.006212`，但 Navtest 为
  `0.910079`、相对 Base `-0.001415`。balanced gate 在 Navtest 下降
  `-0.005313`。这再次证明单一 held-out split 的 scorer 增益不能外推到 Navtest。
- 三个按 validation 锁定的 wave-5 方案正在 `rl-zt4` 使用全部 162 条训练日志
  refit；完整 Navtest watcher 已就绪。该实验检验 split 方差，但不会修改已锁定
  架构、epoch 或 gate。
- 下一条主路径仍是 wave-6：直接使用同一 No-VQA/M0 vision encoder 的当前
  `CAM_F0/L0/R0/B0` 空间 token，训练 scorer-private perception；不读取 future、
  evaluator 或 DrivOR 表征。缓存完成后会自动启动 8 个预注册变体。
