# Register64 PDMS-First Closed-Loop Planner

## 1. Objective and local evidence

This route is designed to maximize this repository's formal NAVSIM-v1.1 PDMS,
not to reproduce a leaderboard method in isolation.

The completed Register64 diagnostic already separates the two bottlenecks:

| selector | navtest PDMS |
|---|---:|
| random proposal | 71.364 |
| proposal 0 | 76.295 |
| trained DrivoR | 82.294 |
| Oracle@64 | 98.391 |

The generator ceiling is already high (only 1.609 points below 100), while the
selector leaves 16.097 points relative to Oracle@64. Therefore the route first
fixes value-label and ranking quality, then uses conservative generator
refinement to improve coverage without destroying the existing ceiling.

The previous Dynamic Top-32 DriveSuprim experiment is not the default best
route: its measured PDMS was 65.927, versus 82.294 for DrivoR OFF, and its
validation refinement gain was negative. The existing implementation and
configs remain available as an ablation.

## 2. Integrated architecture

```text
Qwen VLA (same prompt/history/navigation contract)
  -> GlobalSceneQFormer: 16 x 256 scene tokens
  -> 64 learned trajectory registers
  -> four donor-fidelity trajectory decoder blocks
  -> 64 x [8, 3] deterministic proposals
  -> detached proposal geometry
  -> four-layer cross-candidate DrivoR value decoder
       |- six structured safety/utility heads
       |- direct aggregate PDMS head
       `- validation-calibrated hybrid selector
  -> one [8, 3] trajectory
```

The scorer receives all 64 candidates together. Its self-attention models
competition among candidates, and its cross-attention reads the same scene
tokens as the generator. Proposal geometry is detached at the scorer boundary:
score loss may improve the shared Qwen/Q-Former perception in Stage 1, but it
cannot move waypoints through a shortcut gradient.

The final selector combines two complementary estimates:

1. the DrivoR structured formula, which preserves multiplicative safety gates
   and weighted TTC/progress/comfort utility; and
2. the DriveVLA-M0-style direct scalar head trained against actual aggregate
   PDMS.

Both scores are standardized inside each 64-candidate scene. A 21-point
validation grid chooses the fusion coefficient `alpha` and stores it as a
persistent scorer-checkpoint buffer. The grid includes direct-only (`alpha=0`)
and structured-only (`alpha=1`), so calibration can retain either endpoint.

## 3. What was migrated exactly

### 3.1 DrivoR proposal/scorer topology

The Register64 and scorer decoders retain the donor block order and dimensions:

- 256 model dimension and 1024 FFN dimension;
- one attention head;
- four blocks;
- register self-attention, scene cross-attention, then FFN;
- separate query and memory LayerNorm;
- projection dropout 0.1 and stochastic depth 0.2;
- one register emits one complete 8 x 3 trajectory;
- trajectory re-embedding is 24 -> 1024 -> 256;
- proposal geometry is detached before scorer embedding.

This is not the older simplified pre-LN attention implementation.

### 3.2 CLOVER coverage and closed-loop update

Stage 1 uses the official evaluator-filtered pseudo-expert package. Selection is
the released donor procedure:

- require `valid=true`;
- retain `pdm_score >= 0.8`;
- descending score seed;
- greedy farthest-point selection with 0.05 mean-XY separation;
- at most eight experts;
- append the logged expert when no selected target covers it within 0.5;
- invalid or absent scenes fall back to the logged expert.

Random/Gaussian perturbations are explicitly not substituted for pseudo
experts because they omit the privileged route, drivable-area, future-occupancy
and evaluator filtering used by the method.

Stage 2 follows the paper's conservative update, not the incomplete preview
loss:

```text
0.1 * (GT WTA + 0.02 * released inter-trajectory term)
+ 1.0 * scalar Top-8 set coverage
+ 1.0 * vector-Pareto set coverage
+ 0.05 * register-aligned teacher stability
```

The Pareto set has maximum size 8, minimum size 2 and reward threshold 0.4.
Teacher stability is averaged over corresponding registers. The public preview
uses a minimum over registers and omits the scalar Top-k term; those preview
differences are intentionally not copied because they do not match the paper's
stated objective.

The schedule is critic-first, then generator: 30 cycles of one critic epoch and
one generator epoch. Each cycle builds an immutable bank from the current
generator. A closing critic is fitted after the final generator update so
inference never pairs the last generator with a stale scorer.

Every critic is paired permanently with the exact generator and bank on which
it was fitted. Formal evaluation does not assume that the final alternating
cycle is best: it selects the distribution-matched pair with the highest true
validation PDMS across all cycles plus the closing critic. Generator and bank
SHA256 identities are checked before selection, so a late refinement regression
cannot silently replace a stronger earlier pair or combine incompatible
components.

### 3.3 DriveVLA-M0 value supervision

The scalar value head uses soft-label binary cross entropy against the actual
aggregate PDM score, matching the direct score-head objective described by
DriveVLA-M0. Six DrivoR submetric BCE heads remain active. Two transparent
ranking terms are added for this repository's measured selector bottleneck:

- listwise soft-target cross entropy over the full 64-candidate pool;
- hard-pair logistic ordering for candidates whose true PDMS differs by at
  least 0.05.

These additions optimize the actual argmax/regret behavior instead of assuming
that six independently calibrated BCE heads will rank candidates correctly.

## 4. Exact PDMS label protocol

PDMS and EPDMS are separate targets and require separate banks/scorers. The
formal route uses the manifest protocol:

```text
navsim_v1_1_pdms_two_way
```

Official NAVSIM-v1.1 normalizes progress for each candidate together with the
PDM expert. Scoring `[expert + 64 candidates]` and directly consuming the
pooled score is wrong because all candidates then share the best pool progress.

The implementation performs simulation and physical metric extraction once for
the complete pool, then reconstructs each candidate's exact two-way progress
normalizer and aggregate score from raw progress. A local real-cache parity
check against three independent official `pdm_score` calls produced maximum
absolute error `5.16e-09`.

All metric labels remain float32 under bf16 model autocast. The protocol name,
metric schema, generator hash and bank manifest hash are checked when a scorer
or refinement stage starts, preventing silent PDMS/EPDMS or stale-bank reuse.

## 5. Training stages

### Stage 1: coverage plus joint value initialization

- Qwen trainability follows the repository's visual-unfrozen VLA recipe;
- Q-Former, Register64 and scorer are trainable;
- global batch 32, 25 epochs;
- pseudo-expert coverage weight 0.5;
- full K=64 v1.1 evaluator labels once per scene/batch;
- direct, structured, listwise and pairwise scorer supervision;
- validation chooses and persists hybrid selector alpha;
- paired best generator/scorer checkpoint selected by true validation PDMS.

The DrivoR/CLOVER papers use DINOv2 LoRA rank 32. This Qwen route does not claim
that full visual unfreezing is an equivalent LoRA migration: it deliberately
uses the already-tested Qwen visual-unfreeze policy at 2e-6. A Qwen-specific
LoRA conversion should be evaluated as a separate controlled ablation.

### Stage 2: banked alternating improvement

For every cycle:

1. run the current Qwen/Q-Former/Register64 once and build train/val banks;
2. score every scene with exact v1.1 labels;
3. train only the scorer for one epoch (global batch 32, AdamW 3e-5);
4. calibrate structured/direct fusion on validation scenes;
5. require positive Top-8 and Pareto selected-set enrichment;
6. freeze Qwen/action adapter and scorer;
7. update Q-Former/Register64 for one epoch with Top-k, Pareto and stability.

The enrichment gate refuses a generator update when scorer-selected targets do
not improve true validation PDMS over the full pool. This implements the
condition behind conservative refinement rather than trusting an inaccurate
critic unconditionally.

After the closing critic, model selection compares only matched
`(generator, scorer, train-bank)` records using validation selected true PDMS.
The chosen identities and every alternative are written to
`model_selection.json`; navtest export and the final report consume that chosen
pair.

### Final evaluation

The launcher exports exactly one selected trajectory per navtest scene and runs
the official NAVSIM-v1.1 PDMS evaluator. The final inference model reads no
candidate bank, ground truth or metric cache.

## 6. Why other recent components are gated

### DriveSuprim

DriveSuprim's full method is an 8192-static-plus-dynamic hierarchy with coarse
Top-256 selection, three-layer fine refinement, shared metric heads,
intermediate losses and imitation supervision. Treating only Dynamic Top-32 as
the same method drops important topology and supervision. The current dynamic
port is retained but excluded from the best route because it reduced both
validation score and formal PDMS. It should return only after the complete
static/dynamic metric-parity gate passes and a holdout shows positive gain.

### TOAD/CEM search

TOAD demonstrates that a reliable internal scorer can guide iterative
control-space search. It also expands beyond the scorer's training proposal
distribution. With the current 16-point scorer-to-oracle gap, enabling CEM now
would amplify scorer extrapolation error. CEM is therefore a later inference
ablation, gated on low holdout regret and positive OOD perturbation ranking.

### DriveVLA-M0 structured map/agent branch

DriveVLA-M0 adds topology/agent-aware visual retrieval and structural
supervision because generic VLM features can miss road geometry and agent
layout. This is a valuable next extension, but it is not represented here by a
cosmetic extra MLP. A faithful migration requires explicit map/occupancy/agent
targets and matched sensor features; those dependencies are not fabricated in
this patch.

### Broader method assessment

The design is a synthesis for this repository's measured bottleneck, not a
copy of whichever paper reports the largest headline number:

| work | component worth transferring | decision in this route |
|---|---|---|
| DrivoR | learned proposals, donor decoder, detached-geometry structured critic | transferred with its dimensions, block order and score composition intact |
| CLOVER | evaluator-filtered pseudo experts, scalar Top-k plus vector Pareto targets, critic-first alternation | transferred with donor thresholds, weights and schedule; exact PDMS labels replace proxy labels |
| DriveVLA-M0 | direct aggregate-PDM value target and explicit structural reasoning | direct value target transferred; map/agent retrieval deferred until its real supervision exists |
| BeyondDrive | hard-negative generation and diversity-aware negative selection | motivates score-gap hard-pair ranking; no claim of exact migration without its generator and repulsive objective |
| SparseDriveV2 | factorized proposal vocabulary and coarse/fine selection | not promoted while this Register64 pool already has 98.391 Oracle PDMS; revisit only if coverage becomes limiting |
| GTRS | dense static vocabulary, candidate dropout and dynamic refinement | retained as a future proposal-OOD/scorer ablation, not mixed into the first closed-loop retrain |
| DriveSuprim | 8192-static-plus-dynamic coarse-to-fine selector | gated until the complete architecture and label-parity checks beat DrivoR on holdout |
| TOAD | scorer-guided iterative CEM search | gated until scorer regret and perturbation-OOD ranking are reliable |

This ordering preserves the high measured proposal ceiling and spends the first
retraining budget on the 16.097-point selection gap. Static vocabulary,
iterative search and structural memory remain explicit follow-up gates rather
than being partially approximated inside the promoted model.

## 7. One-command DLC run

Required environment:

```bash
export CLOVER_PSEUDO_EXPERT_PKL=/absolute/path/to/official/pseudo_experts.pkl
```

The official package is linked from the CLOVER repository README. Other data,
map and model paths are loaded from `env.local.sh`.

Dry run:

```bash
cd /mnt/zhangt_workspace/project/VLA-Drive-DDP-DRS
bash ./run_register64_clover_pdms_dlc.sh --dry-run
```

Formal non-interactive 16-PPU run:

```bash
cd /mnt/zhangt_workspace/project/VLA-Drive-DDP-DRS
export CLOVER_PSEUDO_EXPERT_PKL=/absolute/path/to/official/pseudo_experts.pkl
bash ./run_register64_clover_pdms_dlc.sh
```

Resume the same run identity:

```bash
cd /mnt/zhangt_workspace/project/VLA-Drive-DDP-DRS
export CLOVER_RUN_ID=<existing-run-id>
export CLOVER_PSEUDO_EXPERT_PKL=/absolute/path/to/official/pseudo_experts.pkl
bash ./run_register64_clover_pdms_dlc.sh --resume
```

Outputs are isolated under:

```text
navsim_exp/register64_clover_pdms/<run-id>/
  stage1/
  cycles/cycle_01..cycle_30/
  cycles/closing_critic/
  model_selection.json
  predictions/test/
  evaluation/pdms_v1_1/
  summary/summary.json
  summary/summary.md
```

The launcher uses 16 PPU processes and accounts for all 128 CPUs: 16 rank
processes, 48 data-loader workers and 64 persistent metric workers. Metric
caches are generated when absent, validated before reuse, and shared outside a
run-specific artifact directory.

## 8. References and adaptation boundary

- [CLOVER](https://arxiv.org/abs/2605.15120), *Closed-Loop Value
  Estimation and Ranking for End-to-End Driving*:
  pseudo-expert set coverage, vector-Pareto conservative refinement and
  critic-first alternating schedule.
- [DrivoR](https://arxiv.org/abs/2601.05083): learned trajectory queries,
  detached geometry scorer and structured PDM submetric composition.
- [DriveVLA-M0](https://arxiv.org/abs/2608.10413): direct soft-target aggregate
  PDM score head and the motivation for explicit structured spatial features.
- [DriveSuprim](https://arxiv.org/abs/2506.06659): complete static/dynamic
  coarse-to-fine selection requirements;
  retained as a gated optional branch, not approximated by Dynamic Top-32.
- [TOAD](https://arxiv.org/abs/2606.07170): scorer-guided CEM as a future
  inference extension after scorer generalization passes its gate.
- [BeyondDrive](https://arxiv.org/abs/2605.19771): hard-negative and
  diversity-aware training as motivation for scorer hard-pair supervision; its
  full negative generator is not claimed.
- [SparseDriveV2](https://arxiv.org/abs/2603.29163) and
  [GTRS](https://arxiv.org/abs/2506.06664): factorized/dense candidate designs
  evaluated as coverage alternatives, but not promoted over the measured
  Register64 pool.

Every donor-specific constant and topology is recorded in YAML/checkpoint
metadata. Repository-specific additions (direct/structured calibration,
listwise/pairwise losses, exact batched two-way labels, bank decomposition and
enrichment gate) are labeled as adaptations rather than attributed to a donor.
