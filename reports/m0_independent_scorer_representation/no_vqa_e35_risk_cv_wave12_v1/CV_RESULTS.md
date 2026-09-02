# Wave-12 common conservative-reference policy

The final epoch and neural weights are fixed independently in each of five disjoint Navtrain physical-log folds. Navtest is not read.

- Validation coverage: `103288` scenes / `162` logs
- Policies evaluated: `192`
- Robust eligible policies: `0`
- All-log refit gate: `FAIL`

No policy passed the all-fold point, all-fold clustered-CI and safety-factor gate.
The diagnostic-only best worst-fold delta was `+0.00379070`.

Policy priority is fixed as worst-fold robustness, combined clustered lower bound, weighted gain, then lower switch rate. Per-fold tuning is forbidden.
