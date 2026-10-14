# Frozen-pool Scorer objective and regularization experiment

Registered before any new cell has trained or produced validation predictions.
The runtime binding records the actual registration time and content hashes; this
document does not backdate the registration. Source: `b8bd2646042f45f46e30b59099b9008b22da0377`.

## Evidence and hypothesis

The original Scorers fit the 16,581-scene D_S population: S00's selected PDMS is
96.64444 versus its Base's 94.33102, while its selected stage_val endpoint is
91.23774 versus 91.39146. The frozen four-candidate stage_val Oracle is 96.52152.
These are different role populations; do not subtract their absolute means to
estimate a generalization effect. The diagnostic is a large difference in each
role's *paired* gain and significant costly replacement errors.

The deploy training bank already contains 4,606 zero-valued candidates among
66,324 candidates. Bad outcomes are not absent. Candidate-level NC and TTC
failure prevalence differs between D_S and stage_val. More epochs on the same
loss therefore cannot be assumed to solve the problem.

SmoothL1 elicits a robust location, not generally the conditional expected
reward needed by reward-maximizing argmax. In an elementary population with
90% reward 1 and 10% reward 0, its derivative at prediction 0.9 is negative;
soft-target binary cross entropy has zero expected derivative there. This is a
mathematical motivation, not proof that BCE will improve the actual model.

## Fixed experiment

All four new heads start from random initialization. Common parameters use the
same initial values, seed 20261014. No previous or external Scorer checkpoint is
an initializer. Only a checkpoint from the identical cell/experiment can resume.

| Cell | Total reward objective | Decoder dropout |
|---|---|---:|
| R00 | Soft-target BCE for expected score | 0.0 |
| R01 | Soft-target BCE for expected score | 0.1 |
| R10 | Joint nonzero event + conditional positive score | 0.0 |
| R11 | Joint nonzero event + conditional positive score | 0.1 |

The hurdle predicts `P(R>1e-8|context,trajectory) * E[R|R>1e-8,context,trajectory]`.
It does not multiply NC/DAC marginal probabilities or assume independence.
Its positive conditional loss is weighted by the event indicator under the
original scene/candidate distribution, not renormalized per scene's positives.

Fixed: own Base70k, existing Base + three-expert pool, W/A/current-ego inputs,
original inference precision, 2-layer 256-dimensional 4-head Scorer, immutable
D_S and stage_val banks. No new experts, visual encoder, rich-bank augmentation,
test labels, candidate identity feature, or change to the candidate generators.
The semantic-only choice isolates the new reward objective and regularization;
the original spatial matrix continues separately, unchanged.

Each cell: 6,000 updates, scene batch 128, microbatch 16, AdamW 1e-4 / wd 0.01,
100-step warmup then constant LR. First 3,000 steps: natural scenes and pointwise
objectives. Remaining 3,000: 64 natural scenes and 64 gap-weighted pair draws.
Components retain weight 0.2; ranking retains weight 0.2. The comparison sampler
is the existing `abs(gap)/number_of_valid_pairs`, never scene-gap normalized and
never weighted a second time after sampling. R10/R11 additionally use the joint
event BCE at weight 1.0. This extra supervision is part of the objective change.

Validation at exactly 1k, 2k, 3k, 4k, 5k, 6k. Select by stage_val raw-argmax true
PDMS, then fewer *new* NC+DAC failures, then earlier endpoint. Report every
endpoint and all four cells, including failed or regressing cells. No adaptive
extension or fifth configuration. At most 24,000 new optimizer updates.

After all four finish, lock the winning cell/endpoint using stage_val before
reading new dev labels. Optional margin uses selector_cal only and is reported
separately from argmax. Full dev compares Base, every fixed expert, Oracle and
all four Scorers on the identical pool; report gain over best fixed policy,
regret, zero counts, high-to-zero, new NC/DAC/TTC and log bootstrap intervals.
No new navtest, ACTIVE update or generalization claim based on training fit.

## Acceptance and execution boundaries

Unit checks cover mean estimation, hurdle masking, finite gradients, common
initialization, detached features, candidate independence, microbatch reduction,
dropout RNG/sampler exact resume across the phase switch, and terminal resume.
Current command: `python -m pytest tests/iqe/test_robust_scorer.py -q` — 10 passed.

Reuse the native trainer and checksummed checkpoints. Shared cached contexts are
read-only. Resource routing is recorded independently and can change without
altering this scientific identity. Preserve all existing Base/G/real jobs.
The new training uses permanently authorized hosts; no unguarded long optimizer
job is admitted to a temporary host. No full VLM is resident for head training.

## Primary sources and limits

- [Hydra-MDP](https://arxiv.org/html/2406.06978v2) learns environment metric
  predictions with BCE; its total-score ablation motivates inspecting the
  irregular aggregate objective. Its trained representation and candidate
  construction differ from our frozen-VLA setting.
- [DrivoR](https://arxiv.org/html/2601.05083v2) separately trains metric scoring
  with actual candidate geometry and BCE. Its visual adaptation, cameras and
  candidate interactions are not copied here. No weights or teacher outputs
  are imported.
- [GTRS](https://arxiv.org/html/2506.06664v1) studies candidate coverage and
  regularization. Its large vocabulary, consistent sensor augmentation and
  refinement are not equivalent to this four-candidate experiment.
- [DriveSuprim](https://arxiv.org/html/2506.06659v2) motivates attention to hard
  negatives and consistency. It does not establish that our present features
  can resolve these errors; no external teacher is used.

These papers motivate bounded hypotheses. They are not evidence that the new
heads are robust, nor permission to tune on existing test results.
