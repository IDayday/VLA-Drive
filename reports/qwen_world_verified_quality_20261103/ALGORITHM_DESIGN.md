# S6 verified quality: fresh IL70, actual B1 improvement from epoch0

This is an independent scientific route, not an update to a running S1. The
population-level I2k audit found that its deficit relative to Control10k was
concentrated in the fixed IL-high cohort; many matched safe/high scores differed
by less than .02. These policy comparisons motivate a smaller quality margin.
They do not prove a sampled TRAIN group will supply a teacher or establish a
causal explanation. No NAVTEST scene IDs, labels or Control trajectories enter
the learner. The scorer and deployment policy are unchanged.

## Exact recipe

Same complete IL70, fresh Adam0/RNG42; world8,64 different fresh scenes, G16,
four source inner updates. Original W3 source PPO, SafetyEP/global-valid-rawAstd,
canonical rewards and beta50 physical IL mean anchor stay intact. Action/memory
peaks5e-5/5e-6, warmup .08 valid-scene epoch, hold .75, cosine to4, floor .05.
Initial6/minimum3/optional20epochs. First full TRAIN epoch is uniform, with the
same existing own-run TRAIN-only priority thereafter. Current memory/KV graphs
are rebuilt for every differentiable call.

Nominal supervision begins at **actual exposure0**, with no qualification gate
injection. Score the actual deployed single B1 ODE. A teacher must be an actual
same-scene/current-version primaryG16 (or original conditional extraG16) endpoint
with NC=DAC=TTC=Comfort=1 and canonical score>=.9.

| Current actual B1 | Required teacher improvement | Reallocated beta50 share |
|---|---:|---:|
| not strict-high | >=.02 | .05 |
| strict-high (four components1 and score>=.9) | >=.005 | .02 |

No valid verified gain means no auxiliary. Invalid padding makes no extra query.
The existing extraG16 rule is unchanged: only if B1 is not strict-high **and**
the primaryG16 has no strict-high witness. It is teacher-only, not PG/std/exposure
material. No extraG16 is silently added for strict-high B1 or a small gain.

For an eligible scene, the objective is

`PG + 50*((1-alpha)*IL_mean_MSE + alpha*(.75*scaled_CFM + .25*scaled_true_ODE_last_SmoothL1))`.

This replaces the original beta budget, not an extra beta or BC objective.

The actual v1 full-IL70 qualification exposed a cold-start calibration issue:
the current IL-anchor norm can be almost zero while PG and the two auxiliary
norms are nonzero. Therefore this independent **v2** changes only the S6
calibration target to `max(norm(gIL), .04*EMA(norm(gPG)))`. The PG-norm EMA uses
decay .9 and initializes at the first strictly positive measured PG norm.
Existing S1 V1 calibration is unchanged. These are rank-local gradients over
the actual full trainable closure for the same teacher microbatch, without
mutating `.grad`, adding a gradient collective or introducing another loss.

Component scales use the existing bounded .9 EMA and freeze after100 eligible
teacher-slot calls; diagnostics continue to500 eligible calls. When both
current PG and its historical EMA are zero, minimum scale is retained but no
eligible/calibration/measurement budget is consumed. Such calls continue to be
probed, so they cannot exhaust the measurement budget before PG appears. Once
initialized, the PG EMA can guide a PG0 teacher slot. Every counter, measured
norm, PG EMA and target is native-persisted and strictly validated at resume.
The unchanged beta50/alpha shares remain coefficients, not observed gradient
fractions. In an ideal same-target calibration, triangle bounds imply a
repair auxiliary norm of5–10% of PG and quality auxiliary2–4%; actual component
directions, moving EMA, clipping/floors and native measurements still matter.
No deployed-score gain or production stability is established by CPU tests.

The frozen-IL reference cache defaults on for this new fresh run and is recorded
only as an execution optimization. Its helper is byte-identical to c8a0440. It
does not cache current memory or change the original reference/PG mathematics.

## Native interfaces and evaluation

- `starVLA.rl.qwen_world_verified_quality.recipe.build_recipe(basecfg, arm='S6_VERIFIED_QUALITY')`
- `validate_recipe(cfg)`; `PROTOCOL` equals `cfg.research.verified_quality` exactly.
- `recovery.validate_boundary(..., legacy=False, expected_rank=None)` and `validate_ranks`.
- `tools.qwen_world_verified_quality.train`: the original native Trainer, thin runtime only.
- `tools.qwen_world_verified_quality.native_review export --parent-checkpoint ... --rl-checkpoint ... --output ...`.

The CPU exporter first verifies all8 actual native payloads, Adam/scheduler/
stream/calibration and quality protocol with the new recovery. It only exports
a complete-window READY/COMPLETE boundary through the original deployment
exporter. Its independent `world_verified_quality_export_native_validation_v1`
receipt has checkpoint string/files_sha256/identity_sha256/counts.updates and
`quality_protocol_sha256=fingerprint(cfg.research.verified_quality)`.
Qualification export requires explicit `--qualification-native` and remains
marked qualification, not a formal score. Pending review is never a deployment
export. Native pending material revalidates actual epoch, both gain/alpha gates,
same scene/version/condition, true B1 and teacher endpoints, seed and query counts.

Full single-B1 NAVTEST12146/seed2101 at .0625/.125/.25/.5/1/2/3/4/6/8/12/20epochs;
first two gates404/808actualAdam. These early observations do not reduce the
minimum training commitment. No best-of-G or TRAIN score is a deployment score.

## Qualification, not deployment

`tools.qwen_world_verified_quality.qualify_native prepare --root NEW_QUAL_DIR
--source FROZEN_SOURCE --config EXACT_CONFIG --job-prefix QUAL_S6_V2` writes three operator-dispatched
existing Trainer commands: continuous4, split2, same-native resume4. No tool
dispatches or allocates resources. After actual EXITED0 jobs, `compare` verifies
both true teacher classes activated at epoch0, strict8rank pending and final
native payloads, parameters/Adam/scheduler/RNG/stream/calibration and per-step
loss/gradnorm/LR equality. CPU mocks never count as real GPU qualification.

The helper generalization accepts explicit V2 config only. Default V1 material,
selection seed, loss and gradients are regression-tested bit-exact against the
old frozen S1 implementation. Existing formal/cache/runtime worktrees are not
edited by this development branch.
