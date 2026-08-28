# Gate C Test Audit

## Result

- Command: `pytest -q tests`
- Result: **42 passed, 0 failed**
- Gate C-specific file: `tests/test_shared_future_candidate_consequence.py`
- Gate C-specific tests: **16 passed**
- Existing NAVSIM candidate-relative audit tests remained green.
- Python sources passed `compileall`; all new shell entrypoints passed `bash -n`.

The 15 emitted warnings are non-failing deprecation/runtime warnings from
Matplotlib/PyParsing and Shapely in the existing NAVSIM scoring smoke test. No
vendored nuPlan upstream test tree was collected: the requested repository
scope was `tests/`, so unrelated upstream collection failures were neither
hidden nor counted as failures of this change.

## Covered before Gate C1

- v3 static/dynamic/risk field separation and inference metadata;
- rejection of official score/factor and future-field inputs;
- legal `train`/`trainval` split enforcement;
- randomized, unique, deterministic candidates and randomized GT index;
- current-only EpisodeDrive feature payload;
- log-disjoint five-fold assignment and all-log per-log cap;
- preservation of the complete 103,288-scene inventory separately from the
  45,378-scene log-balanced manifest;
- within-scene and cross-scene future controls;
- recomputed TTC numerical behavior and calibration-bin accounting;
- mmap store trailing-failure handling;
- deterministic model-candidate diversity selection;
- cache-superset filtering/reuse.

The exact EpisodeDrive checkpoint was also loaded in the real export path and
repeated proposal export had max absolute error 0. This is recorded as an
integration result rather than mocked as a unit test.

## Conditional model tests

The protocol requires `SharedFutureHead`, direct consequence injection,
GT-only visual loss, zero-gate checkpoint compatibility, permutation
equivariance and consistency-verifier tests only after Gate C1 passes. Gate C1
failed its predeclared +0.03 fold-level dynamic-gain threshold, so those modules
were intentionally not implemented or tested. They are **NOT RUN**, not
silently treated as passing or failing.
