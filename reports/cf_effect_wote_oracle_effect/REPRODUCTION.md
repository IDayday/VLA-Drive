# Reproduction

Run `research/cf_effect_gate_wote/scripts/run_gate2o_all.sh` from the isolated
Gate2O worktree with explicit `--data-root`, `--map-root`, `--wote-root`, and
`--release-root` arguments.  Paths follow CLI > one-shot environment > shared
repository defaults; no personal mount is committed.

The launcher enforces: preflight, fixed split, deterministic 16-scene relabel,
full six-factor relabel, label-free frozen features, deterministic primitive
effects, overfit smoke, shared pilot, locked A--L training, full-256 evaluation,
interventions, automatic verdict, and stop.

No test label is used for hyperparameter selection.  EP is always evaluated on
the complete 256-candidate set.  The published `formatted_pdm_score_256.npy`
is not loaded by this pipeline.

NOT_RUN by design: forward effect prediction, inverse dynamics, VLA/WoTE
training, trajectory refinement, and policy distillation.
