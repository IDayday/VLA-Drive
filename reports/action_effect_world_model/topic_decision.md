# Topic decision at Gate 3

## Decision

**Stop the current AEE-WM formulation and select Direction B as the next
probe-only research direction: Partially Identified World Supervision from
Single-Future Driving Logs.** This is a concrete next-study choice, not a claim
that Direction B has already passed a planning benchmark.

## Evidence

- Gate 1 passes and rules out a data-starvation explanation: 5,300 scenes yield
  84,800 fixed-count candidates, 78,688 valid replay-grounded targets, 558,423
  pairs, and 18,947 safety-boundary pairs. Pair thresholds use only 4,200 train
  scenes and exclude the held-out perturbation family.
- Gate 2.5 rules out action-branch engineering failure. The production action
  encoder/fusion rapidly fits a synthetic trajectory target, candidate
  consequences can be overfit, and gradients/Jacobians/embedding variance are
  non-zero.
- Factual-only collapse is a statistical shortcut. Multi-candidate absolute
  supervision raises Effect Alignment by 0.0595 and lowers false-safe rate by
  0.3121 relative to factual-only, showing that candidate supervision has real
  value.
- AEE-specific evidence fails. Relative to absolute supervision, AEE alignment
  changes by -0.00621 with 95% scene-bootstrap CI [-0.01263, 0.00004],
  false-safe rate worsens by +0.10346 [0.06270, 0.14537], and held-out-family
  alignment worsens by -0.02469 [-0.03653, -0.01264]. AEE lowers global
  separation's Equivalence Leakage by only 12.3%, short of the 20% gate.
- Structured action effects are learnable: drivable-area and lane SDF pass the
  mean, zero, scene-only, and within-scene shuffle controls. The failure cannot
  be attributed to an action-invariant target.
- LR/IDM disagreement is concentrated rather than dominant: 218 / 10,879
  critical pairs (2.00%) conflict. Confidence weighting significantly reduces
  false-safe errors versus unweighted AEE, but does not significantly improve
  alignment or safety AUPRC. All 128 reactive-label scenes fall in the training
  partition, so LR-to-IDM test transfer remains unidentifiable in this run.

## Why this direction

Direction A is rejected at Gate 3. Direction C cannot be diagnosed because the
authorized scope deliberately stops before shared-backbone training and thus has
no FM/world gradient cosine. Direction D is too strong because fixed logs do
support useful multi-candidate absolute supervision and action-dependent
structured channels.

Direction B is therefore the most defensible next hypothesis, but it must be
tested with reactive labels that are scene-disjoint across train/validation/test
and with set-valued or interval consequences. The next experiment should focus
on disagreement calibration and selective supervision, keep multi-candidate
absolute as the primary baseline, and require held-out-family improvement before
any Qwen+DiT integration.

Development stops at Gate 3: no Qwen+DiT world loss, planning training, PDMS, or
EPDMS result is included.
