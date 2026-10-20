# Independent affected-scope review: dense temporal-grid ablation

Reviewed source: `d80f658e12485e82879b794b0240b9c4df730443`, parent
`b7cc81fb`. Review was performed in the isolated worktree
`VLA-Drive-effect-temporal-review-20261020`; the running runtime and author
worktree were not edited. Conclusion: **PASS for the affected CPU/code scope**.
No GPU, simulator, real VLA forward or new training job was dispatched.

## Default six controls

The original six-term weights remain value=1, ranking=.2, events=.2,
temporal=.2, uncertainty=.05 and repair=.2. Each invocation copies this mapping.
The default CLI remains .2, retains the existing main output namespace and adds
no extra identity field. The same dictionary enters immutable training identity.
Independent tests loaded the actual parent's source and, for all six modes,
checked exact total loss, every parameter gradient, initial private state,
pair-stream RNG, output namespace and training identity against the new default.

The identity comparison fixes the declared source revision to isolate this
change. Real runs continue to disclose their actual Git source commit; this
review does not claim identities with different source SHA are equal or permit
silently resuming a running old source with a new source identity.

## Meaning of zero

`--temporal-loss-weight 0` is accepted only for `delta_feedback`, with a separate
automatic `g1_ablation/no_temporal` namespace. It removes only the dense [41,3]
time-grid objective's contribution. The temporal head remains allocated, its
forward and diagnostics remain present, and the optimizer parameter inventory
and existing weight-decay behavior stay unchanged.

All eleven physical mean targets remain supervised using their original masks,
units and scales, including observed first-collision, first-road and first-TTC
times when valid. Event transitions, ranking, uncertainty and repair supervision
remain intact. Zeroing the grid loss is not removal of all temporal information,
and cannot alone establish a policy-level causal-credit contribution.

## Restore and reporting isolation

The actual loss dictionary is part of immutable run identity. Tests cover both
.2→0 and 0→.2 restore attempts and require rejection before another batch is
consumed. The unchanged default does not reuse the ablation output implicitly.

The primary six-control comparator rejects a current checkpoint with ablated
weights. The separate explicit-snapshot comparator checks equal model capacity,
data, labels, evaluation population, seeds, optimizer/scheduler recipe, hardware
and numerical runtime, exposures, consumed scene sampler and pair RNG. Only the
dense temporal weight may differ; device ordinals and disclosed source commits
are separately handled. Independent negative tests reject changed data, event
weight, batch, exposure and hardware. Existing tests additionally reject changed
ranking weight, microbatch and pair sampler. Missing results remain NOT_RUN /
PENDING rather than scientific PASS.

The reporting flag correctly marks the zero-grid arm's dense head unsupervised
while retaining first-event mean-channel metrics. The comparator remains an
effect-prediction and within-bank ranking analysis, not a policy evaluation.

## Actual tests

CPU environment: CUDA hidden, BLAS limits 1, PyTorch intra-op threads at most 2,
existing `ddp` conda environment. Affected test command:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/action_effect/test_temporal_ablation.py \
  tests/action_effect/test_temporal_ablation_independent_review.py \
  tests/action_effect/test_model_effect.py \
  tests/action_effect/test_g1_comparison.py \
  tests/action_effect/test_small_training.py
```

Final result: **76 passed in 12.36 seconds**, one pre-existing pynvml deprecation
warning. Initial run was 75 passed / 1 failed because this review's new shuffled
context fixture omitted its required recorded permutation. The fixture was
corrected to exercise the original interface; production source was unchanged.

This is small CPU behavioral evidence, not actual G1 fitting, unobserved-log
scientific success, repair benefit or permission to infer causal temporal credit.
