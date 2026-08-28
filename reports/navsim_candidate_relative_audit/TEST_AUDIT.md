# Test Audit

- `pytest -q tests`: **26 passed**, 15 warnings, 2.12 s. This includes the 12 candidate-relative audit tests and the repository's two existing top-level test modules.
- `pytest -q tests/test_navsim_candidate_relative_audit.py`: **12 passed**.
- A root-level `pytest -q` was also attempted. Pytest recursively collected the vendored `nuplan-devkit` test suite and stopped during collection with 121 errors because optional upstream development dependencies/fixtures are not installed (`mock`, `cachetools`, `moto`, `testbook`, and `nuplan_test` fixture configuration among them).

No dependency was installed or changed for the audit. The vendored-suite collection blocker predates and is outside the added audit code; the scoped repository tests complete successfully.
