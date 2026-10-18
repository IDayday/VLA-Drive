# Actual warmup execution check

This read-only audit checks the first 105 completed updates of the running
rollout-standard-deviation arm against the original C1 control. All four ranks
consumed the same 3,360 unique TRAIN observations, with identical learning-rate
and exposure sequences. All 13,440 new-arm official feedback records are valid;
the trainer verified actual FP32 master updates at every checked step. The
original 4,096-step schedule passed warmup at step 100 without restarting.

This verifies this prefix of the execution contract, not final convergence,
full-population feedback quality, a planning improvement or actual GPU resume.
No model, image, feature cache or DEV input was opened and no scoring was run.

The first invocation's result is preserved as
`RESULT_RUSAGE_HIGH_WATER_OBSERVATION.json`. Its `peak_process_RSS_bytes` was
incorrectly labeled: Linux `getrusage.ru_maxrss` retained the executor ancestor's
2,095,808,512-byte high-water mark. This is not an isolated peak for this small
stdlib audit. The measurement correction reads `/proc/self/status` and is
explicit in `audit.py`; the second invocation's `RESULT.json` records a real
24,461,312-byte executable RSS high water and labels the inherited observation
separately. The data/LR prefix hashes match in both attempts. Both attempts are
retained, with 2,132,698 bytes total log input, 0.0690 seconds wall and 0.0352
seconds CPU (rounded). Neither invocation changed training or consumed a batch.

Reproduction (exclusive result creation intentionally refuses silent overwrite):

```bash
python reports/st_world_policy/20261018/rollout_scale_warmup_execution_audit_v1/audit.py
```

The source logs are append-only running artifacts; the result binds the exact
read prefix bytes and SHA256, rather than claiming that a still-growing full
file has an immutable checksum.
