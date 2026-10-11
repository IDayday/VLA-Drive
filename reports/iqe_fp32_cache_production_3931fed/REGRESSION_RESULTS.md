# Current regression evidence

Initial final suite: `PYTHONPATH=. OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m pytest -q tests/iqe`
=>204 passed,2 skipped,4 warnings,39.39s (`evidence/cpu_regression_final.log`).
After adding explicit fallback metadata conversion, its6 negative/behavior tests
passed (`evidence/fallback_migration_unit.log`). Delivery suite: **211 passed,2 skipped,4 warnings in39.66s**
(`evidence/cpu_regression_delivery.log`); it supersedes the earlier count.

Behavior coverage: real payload schema and all3 DeepStack; per-record bitwise storage;
bounded FD and no whole-shard torch.load; atomic INTENT recovery/partial commit;
corrupt/wrong identity/role/augmentation rejection; left padding/regroup/tail;
missing rank failure propagated through2-process Gloo; hardlink reuse/idempotence;
JSON plan-order recovery; diagnostic checkpoint cannot become formal S0; original
C1/C2/C3 andR2/R4/R5/R6 tests; strict fallback source/backend/precision and full-state
fingerprint. Real8GPU200-step and real8GPU exact resume are separate integration
runs. Two opt-in generic GPU tests remain skipped, not falsely counted as passed.

No protected Base forward/loss/hash definition changed. `Pipeline.freeze_base`
adds an explicit diagnostic option while preserving normal strict qualification.
No `strict=False`, source hash deletion, old checkpoint rewrite or22k relock.
