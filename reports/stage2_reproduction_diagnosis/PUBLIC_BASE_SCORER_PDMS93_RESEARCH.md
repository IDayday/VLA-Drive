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

## Acceptance sequence

1. Train and choose hyperparameters only on complete-log-disjoint Navtrain
   train/validation logs.
2. Require positive held-out selected-PDMS delta, regret reduction, safety
   factor non-regression, and log-bootstrap support.
3. Run the exact four-scene public parity smoke for the custom agent class.
4. Export all 12,146 Navtest scenes in FP32 and score all 64 proposals.
5. Accept only if the selected PDMS exceeds 0.93 with 12,146 scenes, 136 logs,
   64 candidates per scene, no invalid rows, and no future/evaluator input.
