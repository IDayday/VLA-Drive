Fresh IL70 RL research: mechanism, four live trials, and complete 1k evidence
Snapshot date: 2026-10-31 UTC

Live report (updated every 30 minutes and on milestones):
  /mnt/project/qwen-world-rl-optimization-20261030/report.html
The neighboring optimization report.html in Git is a historical publication.
This package is evidence, not authorization to launch or resume an experiment.

Actual full NAVTEST results (12146 scenes, single ODE K10/shift5/seed2101):
  IL70 parent                              88.792909907
  SafetyEP 1k                              90.707608180
  SuccessMemory 1k                          90.598877160
  FPO 1k                                   90.587598410
  Best completed model: SafetyEP 10k        91.349891737
  Original SafetyEP 30k                    90.999038371
The >=92 target has not been achieved. Neither new 1k result is a gain over
same-budget SafetyEP. Fulltime has no completed score at this snapshot.

What has been learned
  * SafetyEP versus original PPO changes structured credit and its scaling,
    while initialization, data, learning rates, anchor, and trainable scope
    match. At 10k the gain is +0.3304 PDMS; at 20k +0.0681 is uncertain.
    Descriptive score decomposition attributes most of the 10k gain to EP,
    not uniform improvement in safety metrics.
  * Centered batch normalization alone improves original PPO at 10k by
    +0.1344. SafetyEP minus that ablation is not statistically established.
    All SafetyEP gains must not be attributed exclusively to its gate.
  * Actual same-theta credit replay changes Memory gradients as well as
    action gradients. Sparse probes do not show the physical anchor
    overpowering PG in the measured window. BF16 component-additivity
    residuals and the small observation window are explicitly disclosed.
  * Original SafetyEP can assign zero PG to safe EP=1 groups with remaining
    Comfort differences. This is distinct from all-fail equal-cost groups.
  * FPO 1k improves fixed-IL70 high-score scenes by +0.321 PDMS and repairs
    16 of their 22 zeros without new zeros in that cohort. Its remaining
    NAVTEST low cohort loses 1.244 points, so the whole score is lower.
    With the same baseline, TRAIN high and low cohorts both improve; the
    NAVTEST low-cohort regression is not reproduced on TRAIN1024.

Registered live trials
  PPO_SAFETY_SUCCESS_MEMORY: fresh IL70, current-run verified-success bank,
    independent historical scene CFM every four collections. Cap 10k,
    complete evaluation gates 1k/5k/10k. Source 5ac3ae8.
  PPO_SAFETY_FPO: fresh IL70, ordinary per-MC PPO using endpoint CFM energy
    differences; M2, sigma [.2,.8), clip .01. Cap 10k, gates 1k/5k/10k.
    The energy ratio is a surrogate, not exact endpoint likelihood or IS.
  PPO_FPO_FULLTIME: fresh IL70, only sigma upper bound .8 -> 1.0. Cap 5k,
    gates 1k/5k. Source 128a843. Common-region sampling density and expected
    inverse-sigma-squared weight change too; not a pure causal coverage test
    or a reproduction of full [0,1] paper training.
  PPO_SAFETY_FEASIBLE_SCORE: fresh IL70, feasible quality changes from EP to
    (5EP+2Comfort)/7. Source6dc988d, cap5k, gates1k/5k, actual8rank updates.
    Safety feasibility and all-fail fallback remain. Bounded quality keeps
    mixed feasible/unsafe raw-credit ordering. EP weight drops to5/7 and
    Comfort receives2/7; global normalization also changes. Actual reused
    TRAIN groups:111 zero-PG groups become15, activating96 Comfort-only
    groups. This is not measured World gradient or performance improvement.
All use one training seed42, full103288 eligible scenes, the same IL70 hash,
fresh Adam/update0/cursor0, trainable action LoRA plus scene/predictive memory,
and 5e-5/5e-6 learning rates with 100-step warmup then constant. No RL10k forks.

Continuation and allocation
  Native training does not wait for intermediate evaluation. Independent CPU
  owners advance registered gates and end at finite caps without a budget-
  review limbo. Pair futility guard needs BOTH 1k and 5k severe whole-score and
  fixed-IL-high deficits; a small single 1k deficit cannot trigger that rule.
  Root ended the original SafetyEP continuation after complete 20k and 30k
  evidence: 30k-minus10k -0.3509, CI[-0.5946,-0.1401], with no clear NC/DAC/TTC
  gain. It exited0 at31318/inner2 with native recovery retained. This is a
  manual post-hoc allocation decision, not satisfaction of an unobserved
  predeclared35k gate. Closed old routes stay closed. IL work is unchanged.

Evidence navigation
  EXPERIMENT_*_REGISTRATION_20261031.json: exact experiments and finite caps.
  configs/: sealed scientific configurations.
  analysis/SAFETY_EP_VS_PPO_MECHANISM_20261031.json: matched controls/credit.
  analysis/SAFETY_NORMALIZATION_EXISTING_ABLATION_20261031.json: scale control.
  analysis/SAFETY_CREDIT_GRADIENT_REPLAY_INTERPRETATION_20261031.json:
    actual same-theta gradients on already-scored native TRAIN rollouts.
  analysis/SAFETY_FRESH_PAIR_MATCHED_REVIEW_20261031.json: complete 1k scores,
    fixed IL cohorts, zero repair/new failures, and component paired intervals.
  analysis/FPO_TIME_COVERAGE_DESIGN_20261031.json: third-trial motivation/limits.
  analysis/SAFETY_TRAIN_ZERO_SUPPORT_AUDIT_20261031.json: TRAIN zero exposure,
    native witnesses, and strictly separated same-theta ODE/SDE diagnosis.
  analysis/SAFETY_DAC_TIE_COUNTERFACTUAL_20261031.json: CPU-only occupancy
    tie-breaking in two actually observed DAC-only failed groups; no GPU trial.
  owners/: historical launch/state/recovery receipts, never live-owner copies.
  SNAPSHOT_MANIFEST.json: source paths, byte counts, hashes, and capture time.

Uncertainty and data use
  Paired intervals resample physical logs with 2000 draws/seed42, not training
  seeds. TRAIN1024 is part of eligible TRAIN, not independent heldout data.
  NAVTEST informs method/model review; its tokens are not training selectors.
  Existing native checkpoints, canonical scorer, and live runtime snapshots
  were not rewritten. Code checks are bounded CPU checks; they are not new
  driving performance evidence.
