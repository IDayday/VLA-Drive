# G1 dense temporal-grid supervision: pre-registered single-factor ablation

This source change starts at `b7cc81fb`. It does not alter that immutable running
runtime, the six primary G1 arms, their main teacher, G0 samples or policy arms.
The additional run is **NOT_RUN** until Lead separately amends the finite budget
and dispatches it. It is one `delta_feedback`, seed42, 1000-update experiment.

The full model uses the original loss weights:

```
value=1, ranking=.2, events=.2, temporal=.2, uncertainty=.05, repair=.2
```

The ablation changes **only** `temporal=.0`. This is the dense `[41,3]` native
simulator event-grid auxiliary objective. All eleven physical mean targets,
including observed first-collision/first-road/first-TTC times, remain present
with the same masks and fixed physical scales. Event transitions, repair labels,
gap-sensitive ranking, uncertainty, model capacity, initialization, observations,
samplers, microbatch reduction, scheduler horizon and checkpoint semantics stay
fixed. All heads remain allocated, including the unsupervised temporal head.
AdamW still sees the same parameter inventory and its existing weight-decay rule;
zeroing this loss does not silently remove parameters or change optimizer groups.

`train_effect_model --temporal-loss-weight` accepts exactly `0` or `.2`, default
`.2`. The zero variant is restricted to `delta_feedback`. Each invocation creates
its own weight dictionary; the existing default constant is not mutated. The
actual dictionary enters immutable run identity, so resuming one output namespace
with different weights fails before reading a new batch. The automatic zero-run
namespace is `g1_ablation/no_temporal/seed_42`; it never defaults to the main G1
teacher output. Explicit outputs must also use a separate experiment directory.

## CPU evidence

`python -m pytest -q tests/action_effect/test_temporal_ablation.py tests/action_effect/test_model_effect.py tests/action_effect/test_g1_comparison.py tests/action_effect/test_small_training.py`

Final affected-scope recheck: **64 passed in 11.44 s**, including the strict
scalar-type and primary-matrix isolation negative cases. These use small declared
tensor fixtures, not real model-fitting or simulator evidence. A final focused
empty-output compatibility recheck passed all 8 ablation tests in 3.78 s. The
tests establish:

- Default `.2` total loss, every term and all parameter gradients match the old
  recipe exactly on the same deterministic model and batch.
- Zeroing temporal retains all term values/denominators; its total gradient is
  exactly the sum of the unchanged non-temporal objectives. The temporal-head
  loss gradient is zero while first-event-time mean-head gradients remain real.
- Altering the weight during resume is rejected before the next batch.
- Reporting marks the zero variant's dense time-grid head as unsupervised while
  retaining the same first-event metrics, all other labels and scene population.
- The two explicit-snapshot comparison rejects changed ranking weights, optimizer
  recipe, sampler states or population; missing outputs remain PENDING.

## Actual commands (preparation only; no execution authorization implied)

These paths are the registered budget-v2 campaign and its planned 1024-scene
population. They are inputs, not a claim that all artifacts already exist.
Use the immutable reviewed source snapshot, not the running b7 runtime worktree.
The training uses its unchanged 1000-step timeline; do not shorten the schedule.

```bash
python -m tools.action_effect.train_effect_model \
  --config configs/action_effect/shared_budget_v2.yaml \
  --context-manifest /mnt/project/action-effect-repair-20261020-budget-v2/contexts/MANIFEST_001024.json \
  --pair-dataset /mnt/project/action-effect-repair-20261020-budget-v2/g0/population_001024/PAIR_DATASET.json \
  --arm delta_feedback --temporal-loss-weight 0 \
  --device cuda:0 --seed 42 --max-updates 1000 \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/g1_ablation/no_temporal/seed_42
```

For interruption, rerun the same command with `--resume`. A bounded segment may
add `--stop-after N`; that preserves the original full scheduler horizon.
The existing main full-temporal snapshot remains the explicit comparator. Once
both fixed `step_001000.pt` receipts exist, run the existing CPU-only
`compare_effect_models evaluate` separately on both snapshots, using the same
campaign binding, frozen context/pairs and a `--max-pairs` budget covering **all**
held-out pairs. A smaller pair budget rejects rather than truncates the cohort.

Then compare their explicit evaluation `COMPLETE.json` receipts:

```bash
python -m tools.action_effect.compare_temporal_ablation \
  --full-evaluation /ABSOLUTE/FULL_EVALUATION/COMPLETE.json \
  --no-temporal-evaluation /ABSOLUTE/NO_TEMPORAL_EVALUATION/COMPLETE.json \
  --step 1000 --bootstrap 2000 --seed 42 \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/g1_ablation/comparison
```

The two `/ABSOLUTE/...` arguments are deliberately explicit receipts to resolve
after evaluation, not invented existing experiment assets. The comparison checks
actual weights and checkpoint identities, equal private capacity/initialization
recipe, same physical labels, full held-out population, sample exposures, consumed
scene/pair RNG streams and numerical backend. Different source revisions are
disclosed; the tests only establish the affected default-objective equivalence.

This comparison reports complete physical errors, native event transitions,
uncertainty diagnostics, time-grid prediction, within-bank recommendation gain,
regret and introduced risk, plus paired log bootstrap. It does not select a model
or tune a DEV threshold. It cannot prove a policy-level independent contribution
of temporal credit assignment: that requires matched route-A/B interventions and
actual downstream policy evaluations. Current scientific status: **UNTESTED**.
