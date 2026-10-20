# G1 complete-task reporting entry

This adds CPU-only evaluation of an explicit trained snapshot and a read-only
six-control comparison. It does not start training, GPU work, simulation, DEV,
NAVTEST, checkpoint selection, teacher activation, or deployment. Scientific
results remain pending the real matched six-arm experiment.

`python -m tools.action_effect.compare_effect_models --help` exposes two phases.
`evaluate --help` and `compare --help` list the actual accepted arguments.

For the already existing 64-scene engineering qualification, the following is an
actual-path example; it is not a final G1 study or a recommendation to select
this 32-update snapshot:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tools.action_effect.compare_effect_models evaluate \
  --checkpoint /mnt/project/action-effect-repair-20261020/qualification/g1_delta_seed42_32/step_000032.pt \
  --context-manifest /mnt/project/action-effect-repair-20261020/contexts/MANIFEST_000064.json \
  --pair-dataset /mnt/project/action-effect-repair-20261020/g0/population_000064/PAIR_DATASET.json \
  --campaign-binding /mnt/project/action-effect-repair-20261020/CAMPAIGN_BINDING.json \
  --max-pairs 405 --microbatch 8 --threads 2 \
  --output /mnt/project/action-effect-repair-20261020/analysis/g1_qualification32_cpu_v1
```

The implementation agent did not execute that real evaluation. It is supplied
to document a real existing checkpoint and data interface. Historical private
G1 checkpoints can have earlier forward contracts; source versions remain
explicit, and a successful strict tensor load alone does not establish that
changed private-module semantics match historical training. Formal controls
should use a common audited source contract and the predeclared fixed endpoint.

Evaluation requires the entire held-out TRAIN pair list; `--max-pairs` is an
upper cost bound, not permission to truncate a favorable subset. It verifies the
original training checkpoint, exact context/data pins, independent native compact
admission and log roles. Frozen head inference is FP32 on CPU. It saves full
pair-level predictions, per-scene metrics, complete-task statistics, an immutable
training-history snapshot and a completion receipt. `--resume` only reuses a
complete identical evaluation; interruption before completion can repeat CPU
inference, never optimizer updates or simulations.

For comparison, create a JSON mapping all six actual arm names to their explicit
`COMPLETE.json` evaluation paths, using `null` for not-yet-produced controls.
Invoke `compare --evaluations-json FILE --step FIXED_STEP --bootstrap 2000
--seed 42 --output OUTPUT`. Missing controls produce PENDING/NOT_RUN and exit2;
corrupt or mismatched inputs exit1. A complete descriptive table exits0 but still
keeps `scientific_conclusion=UNTESTED`. The report does not pick the best arm or
refresh to a later checkpoint. Its default paired-effect teacher identity is a
method declaration, not a choice based on DEV.

The table checks identical Base, frozen context, physical pair population,
complete held-out scene/pair IDs, label units/scales, seed, allocated parameter
count, model configuration apart from mode, optimizer recipe and fixed step.
Training source commits and runtime identities are disclosed. Same parameter
count does not prove same source semantics or same information supervision.

Metrics include all eleven native channels with valid pair/scene counts,
scene-weighted physical-unit RMSE/MAE/bias, separate pooled-pair RMSE, native
selection and Oracle opportunity, regret, random interventions, all component
changes, high-score-to-zero transitions and introduced collision/road/TTC
violations. Joint intervention combinations retain their own held-out subset.
Every covered scene is retained. Missing scenes prevent a full-population PDMS
claim and prevent silent common-subset bootstrap.

Event heads report true-by-predicted transition confusion, class support,
rare-class recall, balanced accuracy and fixed-bin calibration diagnostics.
Untrained terminal-control auxiliary heads are explicitly marked untrained.
Uncertainty coverage uses physical standard deviations derived from the declared
normalization scales. No output is declared calibrated.

Native absent first-event times remain masked; zero placeholders never enter time
error. Before-only/after-only/both/neither observation patterns are reported.
Censored-likelihood regression/calibration is UNAVAILABLE because such a target
was not trained. Top-k true gain is an offline upper bound among recommendations,
not a simulator-free deployed selector.

Training cost sums immutable checkpoint history update times, retains all
validation snapshots and separately identifies the last RESULT invocation.
Checkpoint/validation amortized cost is UNAVAILABLE when the original trainer did
not measure it. Evaluation records actual CPU wall/process seconds and peak RSS;
no projected speed is substituted.

Current automated evidence: **19 CPU tests passed**, including native-label join
rejection after rehashing an evaluation, complete-six-control artifact reporting,
fixed-step rejection, different scene/pair weighting, first-event missingness,
rare risk, stable tie selection, group bootstrap, pending states and CLI exit
codes. Fixtures are synthetic reporting data, not learned-model or simulator
results. The implementation agent ran no GPU, optimizer or new simulator call.
