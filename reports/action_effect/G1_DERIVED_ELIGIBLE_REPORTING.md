# G1 full requested population and eligible-intervention reporting

This is a reporting-only correction starting at `b1a5438f`. It does not alter
trainers, existing evaluators, their original COMPLETE/METRICS files, checkpoints,
model predictions, simulations or dataset contents. No re-training or forward
pass is needed. The original full233-scene evidence remains immutable.

## Why the original comparison refused

The complete native TRAIN1024 bank has20678 pairs across983 scenes. Its source
roles are791 fit/233 validation scenes. Streaming native compact metadata gives:

| Role | Native pairs | Scenes with pairs |
|---|---:|---:|
| fit |16063|758|
| validation |4615|225|

The8 remaining requested validation scenes have zero valid native intervention
pairs in this locked bank. The original per-pair evaluator correctly predicts
all4615 available validation pairs but retains requested233, covered225 and
missing8 in its metrics. The existing bootstrap refuses incomplete populations.
That guard is retained verbatim. It must not be weakened to intersect whichever
model predictions happen to exist.

## New source-defined admission

`compare_eligible_effect_models` first uses the original strict receipt loader:
original checkpoint/run identity, pinned dataset/context, all native pair labels,
all locked prediction IDs/order and original recomputed metrics must match.
Only then does a new versioned report:

1. Check the complete TRAIN native dataset and exact original full validation
   scene/log inventory. Any native invalid-scene evidence requires separate audit
   and is refused, not reclassified as structural no-action.
2. Define eligibility solely by source scenes having native intervention pairs.
   Require every native validation pair exactly once, with its own scene/log,
   candidate and kind. A missing pair is rejected even if that scene has other
   predictions. A missing entire eligible scene is likewise rejected.
3. Require original missing IDs to equal exactly the source's no-pair IDs. No
   true missing prediction can be hidden among those8 structural scenes.
4. Recompute descriptive statistics from existing predictions on the complete
   **225-scene /4615-pair task**, including all physical labels, errors, events,
   uncertainty, risks, recommendation deltas and paired-log intervals. Report
   full233 and structural8 separately. This is not full233 policy PDMS.
5. Optionally define forced identity on structural scenes with incremental delta0.
   Its requested-population incremental mean is the eligible sum divided by233.
   This does not invent their Base/selected absolute score or call no-action a
   model recommendation, repair success or generalization success.

All models must share exactly the same full requested and admitted populations,
Base/data/context, fixed endpoint, label schema, capacity, seed and optimizer
recipe. Main-six comparisons still reject the no-temporal recipe. The two-run
temporal comparison uses the same admission and the existing exact weight/
sampler/runtime checks; only its `.2→0` temporal-grid objective may differ.
Independent review additionally required actual six-arm consumption proof,
not merely matching declarations: the tool now fingerprints the already-loaded
checkpoint scene sampler, pair RNG and consumed-scene count and requires exact
equality. These small proofs are retained in the derived JSON. No extra native
decode or model forward is needed.

## Executable read-only entry

For the fixed six actual evaluation locations, use the explicit mapping in
[G1_ORIGINAL_EVALUATION_PATHS.json](G1_ORIGINAL_EVALUATION_PATHS.json):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tools.action_effect.compare_eligible_effect_models \
  --comparison six \
  --evaluations-json reports/action_effect/G1_ORIGINAL_EVALUATION_PATHS.json \
  --step 1000 --bootstrap 2000 --seed 42 \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/g1_evaluation/derived_eligible_comparison
```

This does not launch the missing evaluations. Any absent COMPLETE stays PENDING
with exit2. Actual result values will be produced only after the original CPU
queue has written all full evaluation receipts. The output is immutable under
`OUTPUT/<content-identity>/DERIVED_COMPARISON.json`, with CSV and Markdown beside
it. Different inputs produce another version, not an overwrite.

For the independent temporal ablation, an explicit JSON maps `full` and
`no_temporal` to their actual original COMPLETE receipts. Run the same tool with
`--comparison temporal --evaluations-json ACTUAL_MAPPING.json`, preserving the
same endpoint/data/runtime budget. There is no automatic checkpoint refresh or
new scientific control hidden inside this reporting change.

## CPU evidence and conclusion boundaries

`python -m pytest -q tests/action_effect/test_derived_eligible_report.py tests/action_effect/test_g1_comparison.py tests/action_effect/test_temporal_ablation.py`

**42 passed in6.54s**. New cases cover source-only eligibility, missing whole-scene
and within-scene predictions, duplicate/foreign pair/scene/log, omitted requested
scenes, native invalid evidence, wrong original denominator, retained original
bytes, forged missing prediction in the real receipt-loader chain, versioned
PENDING output, six-model matching including changed actual sampler/exposure,
and the complete temporal comparison path.
These use small explicitly synthetic artifact/tensor fixtures and prove reporting
behavior, not learned effectiveness. CLI `--help` and whitespace checks pass.

Scientific interpretation remains held-out TRAIN **intervention-bank** prediction
and recommendation performance. It is not DEV policy performance, NAVTEST, a
simulator-free absolute233-scene score, temporal-credit proof or a safety guarantee.
