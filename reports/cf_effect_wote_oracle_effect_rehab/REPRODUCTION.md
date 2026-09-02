# Reproduction

Run `research/cf_effect_gate_wote/scripts/run_gate2o_matched_rehab.sh` from the isolated Gate2O worktree.  The matched launcher takes
the stage as its positional argument and resolves machine-local inputs through
the following explicit environment variables:

```bash
ORACLE_SOURCE_RUN=/path/to/oracle-effect-v2-run \
DIRECT_CHECKPOINT_ROOT=/path/to/direct-rehab-confirmation \
MATCHED_ORACLE_OUTPUT_ROOT=/path/to/new-experiment-output \
MATCHED_ORACLE_REPORT_ROOT=/path/to/new-report-output \
bash research/cf_effect_gate_wote/scripts/run_gate2o_matched_rehab.sh all
```

Each output destination must be new; the launcher refuses to overwrite prior
evaluation or report artifacts.

The launcher enforces: preflight, fixed split, deterministic 16-scene relabel,
full six-factor relabel, label-free frozen features, deterministic primitive
effects, overfit smoke, shared pilot, locked A--L training, full-256 evaluation,
interventions, automatic verdict, and stop.

No test label is used for hyperparameter selection.  EP is always evaluated on
the complete 256-candidate set.  The published `formatted_pdm_score_256.npy`
is not loaded by this pipeline.

NOT_RUN by design: forward effect prediction, inverse dynamics, VLA/WoTE
training, trajectory refinement, and policy distillation.
