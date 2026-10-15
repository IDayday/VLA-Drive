# Cost-sensitive objective

The only optimization target is `-sum_i (r_i-min_valid(r))*log(p_i)`, averaged
over naturally sampled scenes and divided by a fit-only fixed `Z_train`.
It is not per-scene normalized, not one-hot winner classification, and not
absolute PDMS regression. Official PDMS still supplies its numerical costs.

For95 indistinguishable scenes with rewards(.90,.92) and5 with(.90,0), expected
Base and expert weights are.045 and.019. The analytic optimal Base probability
is.045/(.045+.019)=.703125. Tests verify the loss and gradient favor Base even
though expert wins95% of scenes. Per-scene normalization would reverse this
decision and is explicitly excluded.

`tests/iqe/test_decision_selection.py` checks this counterexample, score-shift
invariance, masks, ties, K1, nonnegative costs, fixed normalization, microbatch
gradient equality, context detachment, model initialization, permutation and
duplicate candidate behavior. `test_decision_selection_ddp.py` compares actual
two-process Gloo gradients with uneven valid-scene counts, including an all-tie
rank. All ranks retain a differentiable zero term and participate in backward.

The two natural samplers use the same seed and scene list. There is no second
gap-based sampling stream. Fit normalization is recorded in each RUN_IDENTITY
and checkpoint dependency. Validation/calibration/development rewards never
compute it. The real internal-fit value was.7317371446843723. Full-D_S refits
recompute Z over the entire legal training role and record it separately.

`test_decision_margin_reference.py` checks that a feasible but lower-mean or
exactly tied margin cannot displace always-Base, and that the predeclared mean,
switch-count and gamma tie order is respected. This was corrected before any
final-model external evaluation; generator or training objective did not change.

Actual full-suite and real-GPU results are in REGRESSION_AND_RESUME.md and its
archived logs; tensor tests are not substitutes for the real-image integration.
