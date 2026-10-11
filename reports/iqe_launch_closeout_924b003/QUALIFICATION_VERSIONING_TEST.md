# C1 — qualification attempts and immutable Base

PASS on current source. `iqe/base_qualification.py::qualify_base` stores each identity under `base_qualification/<qualification_id>/QUALIFICATION.json`. Identity includes checkpoint, framework/input/normalizer, fixed validation manifest, scoring protocol, rules and evaluation artifact hashes. Repeated identical attempts reuse the same bytes.

`LATEST_QUALIFICATION_ATTEMPT.json` follows attempts. `SELECTED_BASE_QUALIFICATION.json` changes only with explicit `qualify-base --select` and only for QUALIFIED results. A later failed attempt cannot displace a selected version. `freeze_base` and `use_locked_base` revalidate the selected evidence and immutable experiment lock. Replacing a formally locked Base requires another output namespace. Legacy reports are preserved with read-only migration receipts; failure is never promoted.

Regression: `tests/iqe/test_closeout_qualification.py` covers failed→qualified in one root, idempotence, later failure, locked replacement, explicit paths, legacy failure, changed rules/evaluation/checkpoint and a forged successful report whose assessment still fails. The last case recomputes the rules instead of trusting report flags.

Real evidence: step_022000.pt, SHA256 `42320ea3e6fb09c6cd2b955dde7398a4033338a98a68bc4bac111fe8e5e01e93`, selected qualification `175b61961441ccd3005a6b794ed5443198e470ff7f7cb88b612774255772ddb6`. Formal lock is in `outputs/iqe/formal_round1_22k/LOCKED_S0.json`. This is a predeclared engineering qualification for an incremental experiment, not a convergence or deployment certificate. Old weights and historical qualification files are unchanged.
