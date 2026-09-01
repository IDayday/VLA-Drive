# Public-Base scorer improvement toward Navtest PDMS 0.93

Status: active experiment. No Navtest improvement is claimed until a complete
12,146-scene FP32 audit passes all scorer-evaluation gates.

## Locked baseline and headroom

The released EpisodeDrive checkpoint was evaluated in FP32 on all 12,146
Navtest scenes with 64 proposals per scene:

| quantity | PDMS |
|---|---:|
| released scorer selection | 0.909594 |
| best of released-score top 2 | 0.919936 |
| best of released-score top 4 | 0.931367 |
| best of released-score top 8 | 0.942715 |
| best of released-score top 16 | 0.955316 |
| best of all 64, offline oracle only | 0.984112 |

The target of 0.93 therefore does not require changing the frozen proposal
generator, but it does require recovering a substantial fraction of the top-16
selection regret. The offline best-of-K values are analysis upper bounds and
are never available to inference.

## Design selected for the first strict experiment

The first production candidate freezes the released proposal generator and
released scorer, then adds a zero-output residual fine-ranker over the released
score top 16. Inputs are limited to:

- the released trajectory-conditioned scorer token;
- the 64 proposed trajectories and exact kinematics derived from them;
- released factor logits and released aggregate score;
- current-observation scene context already consumed by EpisodeDrive.

The model never reads a future image, future annotation, GT future trajectory,
MetricCache, or official PDM factor at inference. PDM factors are stored in a
physically separate label tree and are used only for offline training and
validation. The residual output layer is zero initialized, so an untrained
artifact selects exactly the public Base trajectory.

This coarse-to-fine design follows the useful scorer-side principle in
[SparseDriveV2](https://github.com/swc-17/SparseDriveV2): reduce the candidate
set with a cheap scorer, then apply trajectory-conditioned fine scoring. It
also retains the candidate-conditioned interaction principle of
[DTPP](https://github.com/MCZhi/DTPP), while deliberately avoiding a full joint
planner rewrite in the first attribution experiment.

The DriveVLA-M0 paper and released implementation also constrain what counts as
a genuinely new scorer feature. The Base Score Head already re-embeds every
final trajectory and cross-attends it to compressed language/scene tokens
before predicting six factor logits. The paper's separate map and agent
branches belong to the Retrieve Model: they produce structurally supervised
retrieval keys and affect planning through retrieved-case TTT, rather than being
concatenated directly into the released Base scorer. Since this study forbids
TTT, a dedicated-feature extension must demonstrate information beyond a second
attention pass over the same Base tokens.

## Active full-data pipeline

The source Navtrain cache contains 103,288 scenes across 1,192 log segments.
Three independent inference workers on rl-zt3 GPUs 1, 2 and 4 export disjoint
token shards using the released checkpoint. A 24-process CPU pipeline scores
completed chunks against the read-only Navtrain MetricCache. Both stages are
atomic and resumable.

Artifacts outside Git:

```text
/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/public_base_features_full_v1
/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/public_base_labels_full_v1
```

The inference cache contains proposals, public scores/factor logits, scorer
hidden states, scene tokens and ego tokens. It explicitly contains no official
score or future target. The parallel label cache contains the seven offline PDM
targets and cannot be opened by the deployable agent.

## Early diagnostic, not a result

A deliberately tiny 256-train/128-validation smoke increased pairwise accuracy
but reduced selected PDMS. This demonstrates that all-pairs RankNet accuracy is
not a sufficient model-selection criterion. The current loss therefore adds
near-tied Top-1 supervision, differentiable expected-regret supervision, a
conservative residual scale, and validation-selected shrinkage. Tiny-smoke
weights are rejected when held-out selected PDMS does not beat the exact
zero-residual baseline.

Subsequent partial-cache pilots use complete-log-disjoint Navtrain validation
but remain non-final because the watched cache was still growing. On a
2,000-scene snapshot, two layers of newly trained trajectory-to-scene
cross-attention produced only `+0.00270` selected PDMS after safety gating; the
set-aware variant produced `+0.00246`. Both were weaker than an earlier local
top-8 pilot (`+0.00631` on its 1,500-scene snapshot). Current evidence therefore
rejects generic repeated attention as the primary route. The next controlled
pilots compare a free utility residual against direct factor calibration with
the released PDMS aggregation formula and a hybrid of both paths.

The deployable selector always retains the released Base choice as an eligible
fallback. It can require refined collision, DAC and TTC probabilities to meet
both an absolute floor and a relative-to-Base tolerance. On the local top-8
partial pilot, one held-out setting yielded `+0.00631` selected PDMS with zero
mean change in collision, DAC and TTC; the gain came predominantly from better
progress selection. This is far below the roughly `+0.0204` Navtest improvement
needed to exceed 0.93, so no benchmark claim follows.

An oracle decomposition on the same 2,000-scene snapshot identifies safety
classification as the immediate bottleneck. Within the released-score top 16,
using the true safe set but the released progress prediction reaches
`+0.01419`; using the predicted safe set but true progress reaches only
`+0.00688`. The top-16 best-of-K gain is `+0.02053`. The label audit explains
why ordinary BCE is weak here: only about 2.4% of NOC labels (including partial
credit), 8.9% of DAC labels and 7.4% of TTC labels are violations. The new
controlled loss therefore (1) maps partial NOC/DDC credit to binary failure as
the official loss does and (2) compares unweighted BCE against a 10x
rare-safety-negative weight. This is training supervision only; inference still
uses current-observation features and proposals.

The controlled top-8 hybrid pilots further narrow the choice. The unweighted
factor loss improved selected PDMS by `+0.00467`; 10x violation weighting was
weaker at `+0.00404`. Adding a direct composite safety target for the
NOC/DAC/TTC conjunction at weight 1 reproduced `+0.00467` to within `0.000002`,
but held-out selection still chose the original per-factor gate. Composite
weight 5 fell to `+0.00319`. Thus a simple extra conjunction classifier is not
the missing improvement; it remains a negative control. The complete-cache
queue prioritizes top-16 unweighted residual and hybrid models across three
seeds, with top-8 runs retained as the candidate-headroom control.

A subsequent switch-aligned pilot found a stronger local objective. Raw
`torch.topk` can order a tied maximum differently from the released `argmax`;
the previous implementation trained on K top-ranked proposals but could add
the Base choice only during deployment, producing a K-versus-K+1 mismatch.
The retained set is now exactly the Base candidate plus K-1 alternatives. A
weighted loss over every Base-versus-alternative pair improved the 2,000-scene
held-out delta from `+0.00392` at weight 0 to `+0.00491` at weight 2, with a
log-bootstrap 95% interval of `[+0.00421, +0.00561]` and 14.5% regret
reduction. Weight 5 over-constrained the ranker (`+0.00329`). Combining the
weight-2 loss with 200x top-K safety weighting was also rejected
(`+0.00116`). Top-16 remained stronger than top-8 (`+0.00491` versus
`+0.00451`). These are partial-cache architecture pilots, not Navtest claims.

## Acceptance sequence

1. Train and choose hyperparameters only on complete-log-disjoint Navtrain
   train/validation logs.
2. Require positive held-out selected-PDMS delta, regret reduction, safety
   factor non-regression, and log-bootstrap support.
3. Run the exact four-scene public parity smoke for the custom agent class.
4. Export all 12,146 Navtest scenes in FP32 and score all 64 proposals.
5. Accept only if the selected PDMS exceeds 0.93 with 12,146 scenes, 136 logs,
   64 candidates per scene, no invalid rows, and no future/evaluator input.

## Navtest policy for every effective scorer

Navtrain validation is a promotion gate, not the reported endpoint. Every
scorer whose log-isolated validation mean delta is positive is entered into the
full Navtest campaign, including results whose bootstrap interval still crosses
zero. A positive lower confidence bound remains the stronger evidence tier,
not a way to omit weaker positive methods from test. Methods with non-regressing
NOC/DAC/TTC form the deployable tier; positive methods with a safety-factor
regression are still tested as a separately marked diagnostic tier and cannot
be selected as the final method without correction. Closely related
hyperparameters remain separate methods; a single winning validation
configuration is not used as a substitute for their test results.

To make this policy computationally practical, the released checkpoint is run
once over Navtest in FP32 while exporting proposals, Base factor logits, the
trajectory-conditioned scorer hidden state, scene tokens and the ego token.
The resulting cache is immutable and contains no future target or official
metric. Every residual scorer then selects from that identical 64-proposal
bank. Only after selection is fixed are the offline PDM candidate/factor
matrices joined for evaluation. A four-scene online-vs-cache check is still
required for each promoted artifact.

The four-scene cache smoke passed exactly on 2026-08-31: proposal and Base score
maximum absolute differences against the locked public FP32 cache were both
`0.0`. Full-cache export was then launched on rl-zt3 GPUs 1, 2 and 4. In
parallel, all-candidate factor scoring was split by complete log across three
hosts with 32 CPU workers per host. The full-data scorer queue runs separately
on rl-zt3 GPUs 3, 5, 6 and 7; no existing job was stopped.

## Complete Navtest outcome for every promoted scorer

The full campaign completed on 2026-09-01. The immutable feature cache contains
12,146 scenes from 136 logs and matches the locked released checkpoint at a
maximum absolute proposal and Base-score error of `0.0`. The candidate matrix
contains all 64 proposals for every scene, has zero invalid scenes, and traces
to the same locked proposal SHA256. Representative online checks for local,
factor-aggregate, set-aware, scene-cross-attention, and their set variants all
matched cached inference at `2.39e-7` or better with identical selected indices.

All 35 unique artifacts whose held-out-log validation bootstrap lower bound was
positive were then evaluated. This includes 27 partial-cache architecture and
loss pilots plus all eight full-data configurations across the planned three
seeds. No promoted artifact was omitted.

| quantity | result |
|---|---:|
| promoted artifacts tested | 35 / 35 |
| public Base PDMS | 0.909594 |
| best fine-ranker PDMS | 0.908851 |
| best delta | -0.000743 |
| best delta 95% log-bootstrap CI | [-0.002145, +0.000664] |
| methods with positive test delta | 0 / 35 |
| methods with positive test CI lower bound | 0 / 35 |
| validation-positive to test-negative sign flips | 35 / 35 |
| methods above 0.93 | 0 / 35 |

The best result is the full-data residual top-16 seed-1 run. Its interval crosses
zero, so it is statistically compatible with the released scorer but is not an
improvement. Most other methods are significantly worse. In particular, the
full-data validation gains of roughly `+0.0036` to `+0.0048` did not transfer to
Navtest. The current fine-ranker therefore fails the benchmark objective and
must not be presented as a new scorer result or SOTA.

This systematic sign reversal is stronger evidence than a single failed model:
the official Navtrain validation split is not a reliable selector for these
high-switch-rate residual heads. Further work must target cross-log/domain
robustness and reduce reliance on calibration patterns specific to Navtrain.
Test results will remain reporting-only; they will not be used to tune residual
scale, gates, or ensemble weights.

The complete per-method table, confidence intervals, factor changes, immutable
cache hashes, and promotion/test coverage assertion are recorded in
`NAVTEST_ALL_EFFECTIVE_SCORERS.md`, `.csv`, and `.json` in this directory.

## Domain-shift diagnosis and temporal-consequence scorer

The complete-cache distribution audit now covers all 103,288 Navtrain scenes
from 1,192 logs and all 12,146 Navtest scenes from 136 logs. Public-Base regret
is `0.022715` on Navtrain train, `0.037122` on the official Navtrain validation
logs, and `0.074518` on Navtest. Navtest therefore has 2.01 times the official
validation regret even though its best-of-64 ceiling remains `0.984112`.
Current-feature linear domain probes separate Navtrain train from validation
(AUROC `0.777`) but cannot separate validation from Navtest (AUROC `0.497`).
The failure is consequently better described as conditional safety/utility
shift than as an obvious current-feature covariate shift. NOC and TTC Brier
errors degrade most strongly on Navtest. Full measurements are in
`SCORER_DOMAIN_SHIFT_AUDIT.md` and `.json`.

Source inspection identified a directly relevant incomplete path in the
released model. `Scorer` defines `pred_col_agent` and `pred_area`, and
`compute_score(..., test=False)` constructs collision/TTC key-agent boxes and
ego-area targets. However, `EpisodeDriveLoss.score_loss` sets `pred_ce_loss`,
`pred_l1_loss`, and `pred_area_loss` to literal zero, while the released config
sets both heads false. The next experiment therefore restores the intended
candidate-consequence supervision without adding any future input at
inference. A new scorer-specific temporal hidden state predicts eight horizons
of collision/TTC occurrence, candidate-relative critical-actor geometry, and
non-drivable/oncoming occupancy before producing a zero-initialized residual
over Public Base.

The compact training-label cache completed for all 103,288 scenes and Base
top-16 candidates: 807/807 chunks, 1,192 logs, zero failures, and exact
(`0.0`) parity with the already locked candidate-factor cache. The labels are
stored separately under the experiment root and are marked training-only. A
2,048-scene one-epoch smoke produced `+0.004339` held-out PDMS with interval
`[+0.002787,+0.006924]`, while improving TTC; this is only an implementation
smoke, not a result claim. Five balanced log folds over the complete dataset
are now running on rl-zt3 GPUs 3, 5, 6, and 7. Every fold artifact with positive
validation mean will be evaluated on complete Navtest; Navtest will not be used
to select the epoch, residual scale, safety gate, or loss weights.
