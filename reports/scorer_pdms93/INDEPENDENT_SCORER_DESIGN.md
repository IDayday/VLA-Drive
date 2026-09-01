# Base-score-independent scorer design

Updated: 2026-09-01

## Decision

The next scorer is not a residual over the released EpisodeDrive score.  It
must rank all 64 frozen proposals from current-observation inputs alone and
must remain functional when the released score and factor logits are absent.

The target experiment is a hybrid of the useful parts of joint and post-hoc
training:

- freeze a proposal generator whose best-of-64 ceiling is high;
- train a scorer-private current-scene representation;
- replay proposals from several generator checkpoints and continuous
  perturbations instead of training on one final 64-trajectory bank;
- attach deterministic, offline PDM factor labels to that immutable replay
  bank;
- optimize calibrated physical factors and same-scene ranking directly.

This keeps the broad, changing candidate support seen by the released joint
training while removing simultaneous proposal/representation drift.

## Evidence that constrains the design

Released public Base on complete FP32 Navtest:

| Quantity | Value |
|---|---:|
| Selected PDMS | 0.9095938782 |
| Top-4 oracle over Base-ranked candidates | 0.931367 |
| Top-8 oracle over Base-ranked candidates | 0.942715 |
| Best-of-64 offline oracle candidate-bank upper bound | 0.9841117300 |
| Candidate-selection regret | 0.0745178517 |

Candidate generation is therefore not the immediate ceiling.  Exceeding
0.93 requires recovering about 27.4% of the available all-64 selection regret.

The complete epoch-3 Navtest audit now rules out the stronger form of the
early-stopping hypothesis.  On all 12,146 scenes and 136 logs, with 64 unique
candidates and zero invalid rows, epoch 3 obtains selected PDMS `0.886904867`
and best-of-64 `0.983883012`.  The released final checkpoint obtains
`0.909593879` and `0.984111730`, respectively.  The paired best-of-64 delta is
only `-0.000228718` and its physical-log bootstrap interval
`[-0.002935, +0.002533]` crosses zero.  Thus late training did not measurably
destroy the maximum candidate-bank ceiling on Navtest.

It did materially improve the *distribution* of proposals: mean candidate
PDMS rises from `0.691376` at epoch 3 to `0.795276` in the released model,
while mean pairwise ADE contracts from `2.514 m` to `1.877 m`.  Epoch 3 is more
diverse but contains many more poor trajectories, making top-1 selection
harder rather than giving a better oracle bank.  The immutable artifacts are:

- epoch-3 checkpoint SHA256
  `2725d472620ef29e7082ddc65c21b9ee0c2db2c40cf0a8da9f6890f53afeb4b3`;
- epoch-3 proposal cache SHA256
  `1cca65ae9142837d5ae0e203a95604ab466f4d907db4a5aab7b54791ab484e63`;
- released checkpoint SHA256
  `7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d`.

The first fixed-feature independent-ranker control (`F0`) is also negative on
the complete held-out physical-log fold.  Its best validation PDMS is
`0.930073` for its direct utility and `0.941830` when selecting by its six
predicted factors, versus `0.962642` for the released Base scorer on the
identical 20,647 scenes.  Pairwise accuracy reaches about `0.87`, but top-1
selection remains inferior.  This is evidence that stable optimization over
frozen proposals is not sufficient when the scorer is restricted to the
released 16-token M0 representation.

That fold is held out from the new ranker, but it is not a symmetric comparison
to Base: 17,607 of its scenes come from official training logs on which the
released Base was already trained, and only 3,040 come from official validation
logs.  Base scores `0.965961` and `0.943421` on those two subsets,
respectively.  The random five-fold result remains a valid bank-overfitting
diagnostic, but its full `0.962642` baseline overstates the fair out-of-sample
target.  All promotion decisions therefore also use the original official
physical-log boundary: 85,109 training scenes from 101 physical logs and
18,179 validation scenes from 61 disjoint physical logs.  The split manifest
is `reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json`.

On the 3,040 official-validation scenes that happen to lie inside fold 0, the
same final F0 artifact obtains `0.899002` by direct utility and `0.914417` by
predicted-factor utility, versus Base `0.943421`; both deltas have strictly
negative physical-log bootstrap intervals.  Thus the corrected subset still
rejects frozen 16-token F0, while the separate full official-boundary run is
retained to avoid drawing a final representation conclusion from only 11
validation physical logs.

The full official-boundary F0 run now provides the stronger result.  On all
18,179 validation scenes from 61 disjoint physical logs, Base obtains
`0.951612`.  Its best direct utility obtains `0.908788` at epoch 6, while its
best predicted-six-factor selection obtains `0.927169` at epoch 7.  The latter
is still `-0.024443` below Base, with log-bootstrap interval
`[-0.027741, -0.021127]`.  The direct head's pairwise accuracy for pairs
separated by at least 0.05 PDMS is nevertheless `0.834105`.  This is direct evidence that
good average pair discrimination does not imply reliable top-1 selection and
that the frozen 16-token bottleneck, not merely loss capacity, is limiting.

A Base-rank-only coarse-to-fine control isolates catastrophic all-64 switches.
The independent model receives no numeric Base score; Base only defines the
eligible top-K set.  On the same complete held-out validation boundary, the
epoch-7 independent coarse head improves Base from `0.951612` to `0.956191`
inside Base top-4 (`+0.004579`, 95% log-bootstrap interval
`[+0.003491,+0.005562]`) and to `0.956349` inside Base top-8 (`+0.004737`,
interval `[+0.003311,+0.005996]`).  This validates coarse-to-fine as a useful
safety structure, but not yet as a deployable improvement.

Both validation-positive settings were promoted to complete FP32 Navtest.
The sign reverses: Base is `0.909594`, top-4 reranking is `0.903616`
(`-0.005978`, interval `[-0.008127,-0.003673]`), and top-8 reranking is
`0.896769` (`-0.012825`, interval `[-0.016299,-0.009232]`).  The online top-4
adapter matches cached inference on four scenes with identical selected
indices and maximum score error `4.77e-7`.  Therefore the reversal is a real
generalization failure, not a cache or adapter mismatch.  Frozen final M0
tokens plus a learned ranker are ruled out even when Base supplies a coarse
shortlist; the private-perception F1/F2 controls remain necessary.

A second conservative F0 policy makes the same point more sharply.  On the
complete 18,179-scene official held-out split it selected `0.955594` versus
Base `0.951612` (`+0.003982`, physical-log bootstrap interval
`[+0.002772,+0.005162]`).  Under the pre-registered complete 12,146-scene
FP32 Navtest audit, however, it selected only `0.896376` versus Base
`0.909594` (`-0.013218`, interval `[-0.017041,-0.009187]`).  The method passed
lineage and parity checks, so this is direct evidence that a stationary final
token bank can overfit even a complete held-out log split.  Stability must be
combined with scorer-owned perception and proposal-distribution coverage.

The first scorer-private InternVL control also fails this transfer gate.  A
validation-locked conservative reference head improved the complete official
held-out split from `0.951612` to `0.952124` (`+0.000512`, physical-log
bootstrap interval `[+0.000196,+0.000817]`).  The exact frozen epoch-1
artifact and thresholds were then evaluated on all 12,146 FP32 Navtest scenes,
64 candidates per scene and zero invalid rows.  It selected `0.905206` versus
Base `0.909594`, a delta of `-0.004388` with interval
`[-0.005926,-0.003095]`.  TTC fell from `0.942039` to `0.932817` and NC from
`0.982216` to `0.979582`, while progress changed only from `0.884715` to
`0.885085`.  Current-image private tokens alone therefore do not solve the
cross-log safety calibration problem; this artifact is rejected and the
multi-camera DrivOR-register replay control becomes the next representation
gate.

A second independently trained F1 conservative artifact now repeats the same
failure with a different locked epoch and policy.  It improves the complete
official held-out split from `0.951612` to `0.952289` (`+0.000677`, physical-log
bootstrap interval `[+0.000333,+0.001016]`).  Per the pre-registered promotion
rule, it was immediately evaluated on all 12,146 Navtest scenes.  It obtains
only `0.903778` versus Base `0.909594`, a delta of `-0.005816` with interval
`[-0.007487,-0.004277]`; NC is `0.978800` and TTC is `0.931912`.  The audit
passes 64-candidate, zero-invalid, regret-identity and artifact-lineage gates.
Two validation-positive/complete-Navtest-negative artifacts now reject this
F1 representation family rather than merely one unlucky checkpoint.  No
Navtest outcome is used to retune its thresholds.

The first post-hoc designs are ruled out as production candidates:

- a private copy of the released Q-Former was 0.002486 below the shared-feature
  scorer on the complete 18,179-scene continuation validation;
- a ranking-only continuation improved the same validation bank, but on
  complete Navtest it reached 0.899193 versus 0.897897 for its epoch-3
  factor-only control and remained below public Base;
- 16 temporal-consequence artifacts that were positive on continuation
  validation all reduced complete Navtest PDMS;
- their collision/TTC ranking AUROC on Navtest factor proxies was only about
  0.59--0.70, and each model introduced 39--120 safe-to-unsafe switches;
- a representative temporal artifact has passed real online-versus-cache
  parity on four identical scenes (maximum score error 2.38e-7 and identical
  selected indices), so the failure is not an offline evaluation mismatch.

The new design must therefore change representation support, replay support,
and ranking semantics together.  Merely adding capacity to frozen 16-token M0
features is not a sufficient hypothesis.

The prior full-log Gate-C audit also constrains consequence modeling.  On
45,377 valid log-balanced scenes, dynamic logged-future features add a
statistically positive but sub-threshold `+0.0271` pairwise increment over the
static baseline.  On frozen EpisodeDrive proposals, the original scorer
selects `0.9626` on the audited fold, whereas the logged-future full-dynamic
oracle ranker selects `0.9210` and the direct-risk oracle ranker selects
`0.9337`.  Most incremental signal is collision/TTC-adjacent risk rather than
raw actor state.  Predicted consequence is therefore retained only as an
auxiliary physical-risk representation; it cannot replace the independent
current-scene value scorer under the measured evidence.

## Training-label semantics in released M0

The per-scene MetricCache is precomputed, but proposal labels are not a fixed
table.  During each training batch, the current 64 proposals are detached,
transformed and simulated against that cache.  The local PDM scorer returns
NOC, DAC, DDC, TTC, progress, comfort and aggregate score.  Six factor heads
are trained with BCE-style losses.  Consequently the released scorer sees a
candidate distribution that changes as the generator changes, while the
shared scene representation also changes.

This is synchronous local logged-future relabeling, not benchmark-server
evaluation and not a reactive counterfactual future.

The word "scene representation" needs to be separated into levels.  In the
released M0 Stage-2 setup, the large InternVL backbone is frozen, so its raw
hidden states are stable.  The 16-token Q-Former compression inside the action
decoder is trainable, however, and is consumed by both the trajectory decoder
and the scoring decoder.  Those 16 scene tokens therefore move throughout
joint training even though the upstream VLM hidden state does not.

More precisely, the current proposal tensor is detached before scoring.  The
PDM simulator therefore supplies a correct label for the proposal that exists
in that batch, but the score loss cannot update its coordinates.  It can update
the scoring decoder and the shared Q-Former scene representation.  The loss
named `final_score_loss` in the implementation is the sum of six factor losses
(NC, DAC, DDC, TTC, progress and comfort); it is not a regression loss on the
final scalar PDMS.  The final scalar is reconstructed from the predicted
factors only for candidate selection and monitoring.

Over multiple epochs this gives the released scorer a useful implicit
curriculum: early checkpoints provide many easy, poor candidates and later
checkpoints provide increasingly close hard negatives.  The labels remain
valid as the candidates move because they are recomputed every batch.  The
cost is not label staleness; it is gradient conflict and non-stationarity in
the shared scene representation.

Thus the useful comparison is not simply joint versus frozen.  Freezing both
the generator and its final 16 scene tokens removes non-stationarity but also
locks the scorer behind a representation optimized partly for trajectory
regression.  The proposed controlled alternative freezes only the high-ceiling
generator and upstream VLM, while training scorer-owned current-observation
queries on a stationary, multi-checkpoint proposal replay distribution.

## Joint versus frozen training decision

Neither "joint" nor "frozen" is sufficient to describe the useful design.

| Setup | Candidate support | Scorer representation | Main benefit | Main failure mode |
|---|---|---|---|---|
| Released M0 / DrivOR | Changes during training | Shared perception, separate scoring decoder | Broad implicit candidate curriculum | Shared feature drift and task-gradient conflict |
| Frozen final tokens + one final bank | One immutable bank | Frozen 16-token M0 bottleneck | Stable and cheap optimization | Representation ceiling and bank overfitting |
| Chosen IPCR setup | Immutable multi-source replay | Trainable scorer-private current-observation branch | Stable labels plus broad support | Higher one-time replay/feature cost |

The selected experiment freezes proposal generation, not scorer perception.
It trains the scorer-private token compressor and ranker together while
replaying immutable candidates from several generator checkpoints.  This is
the useful hybrid: the target is stationary for every replay item, while the
candidate distribution remains broad enough that the scorer cannot memorize
one final proposal template.

### DrivOR classification

DrivOR is a joint-loss method, not a post-hoc frozen-generator scorer.  Its
trajectory and score losses are optimized in the same run.  However, it
re-embeds decoded trajectories, detaches them before scoring and uses a
separate scoring decoder.  Score gradients still update the shared perception
registers.  Its published navval ablation moves from 84.7 with a shared branch,
to 86.8 with separate decoders, to 90.0 with separate decoders plus
stop-gradient/re-embedding.  This supports a scorer-specific pathway; it does
not demonstrate that a fixed final 16-token representation is optimal.

The server also contains the released DrivOR NAVSIM-v1 checkpoint at
`/mnt/project/external/DrivoR/weights/releases/drivor_Nav1_25epochs.pth`
(SHA256 `e1a678f201e4f1ab93d117caad42782cd7ead293bdced2b5f80212bc92426ae3`)
and the matching source at commit
`f02665403df799c1b4ddd8b0d34e073f0555c13a`.  This enables a particularly
useful independent control: bypass DrivOR's trajectory decoder, feed the
frozen public-Base 64 proposals into DrivOR's detached trajectory embedding
and scorer decoder, and retain DrivOR's own current-camera register features.
The adapter must first verify the common eight-pose coordinate/time convention
and four-scene online/cache parity.  A positive result would isolate scorer
representation quality; a negative result would expose proposal-distribution
shift and motivate retraining the same separated architecture on Base replay.

That engineering equivalence is now established.  The external DrivOR full
forward and its immutable-proposal adapter agree exactly on 12 real scenes.
On four of those same tokens, using FP32 cached registers and the same batch
size/device, the custom `DrivORInitializedProposalRanker` differs from the
online adapter's FP32 aggregate score by at most `9.54e-7` and selects the
same candidate in 4/4 scenes.  Its FP32 factor logits, after applying the
external cache's FP16 archival quantization, match the archived values
exactly.  The checkpoint adapter maps 110 tensors containing 6,092,038 values.
The passing audit is
`reports/scorer_pdms93/DRIVOR_ONLINE_CACHE_PARITY_V2.json`; the earlier failed
attempt is retained separately and documents why batch-size alignment and
like-for-like FP16 archival comparison are required.

That direct cross-model control is negative on the complete random fold-0
bank: 20,647 scenes from 32 physical logs, 64 candidates per scene and zero
invalid rows.  Public Base selects `0.962642`, whereas the released DrivOR
scorer applied to the identical Base proposals selects `0.957946`, a delta of
`-0.004697` with physical-log bootstrap interval
`[-0.008314, -0.000867]`.  A stronger scorer architecture therefore does not
transfer automatically across proposal distributions.  The next control
keeps DrivOR's scorer-private current-image registers and released scorer
initialization, but re-fits it on immutable Base/epoch-3 proposal replay.

The cross-model result remains negative on the complete official held-out
split: 18,179 scenes from 61 physical logs, 64 Base candidates per scene and
zero invalid rows.  DrivOR's released scorer selects `0.949808` versus Base
`0.951612`, a delta of `-0.001804` with physical-log bootstrap interval
`[-0.004407,+0.000525]`.  It still obtains `0.7006` pairwise accuracy on
non-tied candidate pairs and `0.8857` for pairs separated by more than `0.10`,
yet switches candidates in `82.61%` of scenes.  This combination shows useful
ranking signal but incompatible top-of-list calibration under proposal shift;
re-fitting the private scorer representation and conservative decision rule
on Base replay is required.

The two policies are nevertheless strongly complementary.  An offline oracle
that may choose only between the Base-selected proposal and the released
DrivOR-selected proposal obtains `0.964267` on the same official held-out
split, versus Base `0.951612`, a gain of `+0.012655`.  This recovers `34.09%`
of the complete all-64 oracle headroom.  DrivOR is better on 5,288 scenes and
worse on 4,493; the signed mean is negative because its losses are somewhat
larger, not because its alternative choices carry no useful information.
This is an offline two-policy upper bound, not a deployable score or a Navtest
claim, but it establishes enough headroom for a learned conservative gate.

A factor-level audit localizes the calibration error.  Across the 5,288
scenes where the DrivOR choice beats Base, its mean target progress difference
is `+0.0804`; across the 4,493 losses it is `-0.0874`.  Safety regressions are
important but much rarer among losses: TTC regresses in `3.18%`, DAC in
`1.31%`, and NC in `0.65%`.  Thus a safety-only veto cannot recover the union
headroom.  The gate must estimate a conservative relative utility/progress
gain as well as rare safety regression, and the ranker experiments must compare
the released equal-weight factor BCE with heavier safety reweighting.

The one-alternative gate also leaves substantial candidate coverage unused.
On the same immutable held-out bank, the Base-plus-DrivOR-top-1 oracle is
`0.964267`, while Base plus the deployable DrivOR top 8/16/32 shortlists reaches
`0.973519`/`0.977614`/`0.982103`.  Their true 64-way oracle-candidate recall is
`19.62%`/`32.20%`/`54.47%`, respectively.  These are explicitly offline
shortlist upper bounds, not deployable selection results, but they justify an
independent `all` variant: frozen scorer-private DrivOR current-observation
features feed a reference-relative gain and safety head over all 64 proposals,
with no future or numeric Base score entering the model.  The reproducible
lineage is recorded in `DRIVOR_SHORTLIST_HEADROOM_AUDIT.json`.

Accordingly, the next independent method has two explicit stages:

1. a DrivOR-initialized, scorer-private ranker selects one proposal from all
   64 using current multi-camera registers, current ego status and proposal
   geometry;
2. a separately trained uncertainty-aware gate compares that proposal with
   the Base policy's proposal and switches only when its lower predicted gain
   bound is positive and learned NC/DAC/TTC regression probabilities are
   below validation-locked thresholds.

The gate receives the Base candidate **index**, never its numeric score.  Its
comparison features are the two scene-conditioned candidate hidden states,
their difference and interaction.  The final eligible set contains exactly
the Base and independent choices, so Base remains an exact fallback.  This is
not a residual correction to the Base score: the alternative proposal and its
representation come from an independently initialized scorer; Base is used
only as a conservative deployment policy.  Candidate order is randomized in
training and tested for permutation equivariance.

The implementation is
`navsim/agents/EpisodeDrive/score_module/drivor_ranker.py::DrivORReferenceGateRanker`
with training entry point
`local_stage2/train_drivor_reference_gate.py`.  Its forward signature has no
future, MetricCache, official score or numeric Base-score argument.  The
targeted scorer test suite currently passes 33 tests, including binary
eligibility, exact fallback and candidate-order equivariance.  All Navtest
promotion remains contingent on a positive held-out-log bootstrap lower
bound and complete 12,146-scene FP32 evaluation.

For strict cached evaluation, DrivOR scene registers now default to FP32.
FP16 register caches remain valid training-throughput artifacts but are not
accepted for benchmark claims.  The parity chain requires at least four real
scenes for (a) full online DrivOR versus its proposal adapter and (b) the
adapter versus the custom scorer on identical FP32 registers, proposals and
current ego status, with identical selected indices.

### Relevant scorer training regimes

- DriveSuprim is a fixed-vocabulary/offline-label design.  It scores a
  pre-defined 8,192-trajectory bank and uses coarse-to-fine filtering,
  refinement and self-distillation to focus learning on hard candidates.
- GTRS deliberately separates generator and scorer training.  Its scorer is
  trained on a dense 16,384-trajectory static vocabulary with trajectory
  dropout; dynamically generated proposals are appended only at inference.
  The paper explicitly motivates this choice as avoiding instability while
  improving cross-vocabulary generalization.
- SparseDriveV2 goes further and removes the learned dynamic generator from
  its main path.  It composes a super-dense fixed vocabulary from path and
  velocity anchors, then trains separate coarse path/velocity scores and a
  fine trajectory scorer with rule-teacher metric supervision.  This is
  additional evidence that scorer quality and candidate coverage can be
  optimized without a co-evolving proposal generator.
- CLOVER is an alternating generator--scorer method in its full system, but
  its Appendix F also defines the closest published control to this project:
  64 generated proposals are frozen, true PDMS sub-score labels are attached
  offline, and independent scorer backbones are trained and compared on the
  same bank.  Its reported fixed-bank Navtest scores range from `0.924` to
  `0.928`; the full CLOVER system reaches `0.945` only after evaluator-filtered
  proposal coverage training and conservative generator refinement.  Thus it
  supports fixed-proposal scorer research, while also showing that a larger
  backbone alone is unlikely to close the entire selection gap.  We borrow
  the fixed-bank diagnostic and value/sub-score formulation, but do not update
  the Base proposal generator in the current experiment.
- M0 and DrivOR are dynamic-proposal joint-loss designs with stop-gradient at
  the decoded trajectory boundary.
- Diffusion-style confidence heads are generally trained with their generator
  and imitation/denoising objectives; they are not equivalent to an
  independently trained PDM-factor ranker.

IPCR borrows disentanglement from DrivOR, vocabulary generalization from GTRS,
coarse-to-fine hard-negative refinement from DriveSuprim, and fixed-proposal
value-estimation diagnostics from CLOVER.  It does not use test-time
optimization, LoRA adaptation, the released Base score, or future inputs at
inference.

## Current-observation feature requirement

The existing Navtrain `internvl_feature.gz` cache contains image-path/current
state tensors, not cached VLM hidden states.  The replay export currently
contains the released 16 Q-Former tokens, which are sufficient for the frozen
representation control but not for the proposed private-perception model.
The private model must therefore run the frozen VLM on current images during
training (or create a separate large, non-Git current-observation token cache).
No future file is needed.  A token-source audit must distinguish language-model
sequence tokens from genuinely spatial vision tokens before corridor-aware
attention is claimed.

## Proposed model: Independent Proposal-Centric Ranker (IPCR)

### 1. Scorer-private perception

Inputs are current images, current ego state and navigation inputs already
available to EpisodeDrive.  The large VLM and frozen proposal generator remain
unchanged.  IPCR consumes spatial visual features before the released
16-token Q-Former bottleneck and learns three query banks:

- dynamic actor/occupancy tokens;
- static road/route tokens;
- traffic-control/global-context tokens.

Current-frame actor and map labels may be auxiliary training targets because
they are current-observation annotations.  Future annotations, official PDM
scores and future images are forbidden model inputs.

### 2. Proposal-centric interaction

Each proposal is encoded point-wise and temporally without a candidate-index
embedding.  Candidate queries attend to the private scene tokens and pool
features along their swept corridor.  The resulting representation has
explicit static, dynamic and kinematic streams.

### 3. Independent coarse-to-fine ranking

The coarse head scores all 64 proposals.  A permutation-equivariant fine head
reranks its own top 8--16 geometrically diverse proposals.  Neither stage uses
the released scorer score, released factor logits or released candidate
features.  Base-rank-only shortlisting is an ablation only; its complete
Navtest sign flip means it is not part of the selected independent method.

### 4. Outputs

- six calibrated PDM-factor logits;
- collision/TTC risk and uncertainty;
- direct utility for pairwise/listwise ranking;
- scorer confidence used to suppress unsafe low-confidence switches.

### 5. Objectives

- masked/weighted factor loss;
- PDMS-difference-weighted RankNet loss;
- low-weight top-heavy listwise loss;
- hard pairs that differ by at least 0.02/0.05/0.10 PDMS;
- explicit safety-before-progress lexicographic pairs;
- calibration loss and log-balanced rare-event sampling.

The direct utility is trained from scratch.  Candidate-set order is randomly
permuted during training and no candidate-source/type/index identifier enters
the model.

## Candidate replay bank

Planned immutable sources are epoch 2, epoch 3, epoch 6, epoch 9 and released
public Base proposals, plus deterministic continuous perturbations around
high-quality and hard-negative proposals.  Each source is hashed and scored
offline with the same read-only MetricCache/PDM path.

Sampling is balanced by complete physical log, proposal source, PDMS range and
safety event.  One checkpoint source and one perturbation family are held out
in turn to measure proposal-distribution generalization.

The complete epoch-3 FP32 all-64 Navtest audit has finished: its best-of-64
upper bound (`0.983883012`) is statistically indistinguishable from released
Base (`0.984111730`), while its selected score and mean-candidate quality are
lower.  Epoch 3 is therefore retained as replay diversity, not as the
deployment proposal generator.  The matching all-Navtrain epoch-3 replay
export and labels are complete in 807 immutable chunks with no zero-byte
artifacts.

The DrivOR-initialized independent path is implemented in
`navsim/agents/EpisodeDrive/score_module/drivor_ranker.py`, with current-only
register export in `local_stage2/export_drivor_scene_replay.py` and training in
`local_stage2/train_drivor_initialized_ranker.py`.  Four strict-log controls
are queued after the full register cache: factor-only and factor-plus-direct
ranking, each with Base-only and Base-plus-epoch-3 replay.  The forward
interface contains no released Base score, future field, MetricCache or PDM
evaluator input.  Candidate-permutation equivariance, proposal stop-gradient,
zero initialization of the direct head and exact released-checkpoint mapping
are covered by the targeted test suite.

The complete training path also passes a four-scene real-data smoke test:
current DrivOR register cache, Base proposals, offline six-factor labels,
physical-log split, one optimizer epoch, validation and checkpoint writing all
complete without future/Base-score model inputs.  The one-scene validation
number from this smoke is an engineering check only and is excluded from every
promotion decision.

## Required comparisons

| ID | Generator | Scorer representation | Candidate replay | Objective |
|---|---|---|---|---|
| J0 | jointly trained | released shared | online changing | released factor loss |
| F0 | frozen | frozen M0 16 tokens | one checkpoint | independent rank loss |
| F1 | frozen | private spatial | one checkpoint | factor + rank |
| F2 | frozen | private spatial | multi-checkpoint | factor + rank |
| F3 | frozen | private spatial | multi-checkpoint + perturbations | factor + rank + calibration |

The central contrasts are `F1-F0` (representation), `F2-F1` (checkpoint
replay), and `F3-F2` (continuous support/calibration).  Every validation-positive
method must receive a complete 12,146-scene Navtest evaluation.

## Promotion criteria

- exact 64-candidate proposal lineage and zero invalid Navtest scenes;
- no future/evaluator input in online inference;
- online/cache parity on at least four identical scenes per architecture;
- positive complete-log bootstrap interval on held-out logs;
- no material collision or TTC regression;
- positive held-out proposal-source result;
- complete Navtest PDMS above public Base before any SOTA claim.
