# Exact C3-only retry resource review

PASS for owner `ba7a70093feb86e29173eca4323e8417b533b878`, `run_c3_once.py` SHA `0a21f2286b45361572f2c55ed866f66c276e268b3273593d2d72ba44eac3223e`. This is a narrow additional invocation review, inheriting the previous fixed-wrapper review `586395381679eb63cfc50eb156ba5cda58725f78`. No scientific settings, evaluator implementation, GPU operation or actual payload analysis was rerun here.

The immutable original command list is reverified and only element 1 (`C3_minus_C0`, `DIAGNOSTIC_1024`, FORMAL) is selected. An independent fixture exercised the actual borrowed `run_pair`: exactly one call to the original `run_fixed_pairs.py --child C3`, with unchanged monitor core SHA `8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644`. C2 is neither analyzed nor scored again.

The old failed session's **1,103,776,571 B** remains pinned and is never overwritten or reset. This includes its observed sampled overshoot. The new explicitly authorized attempt gets **1,073,741,824 B** of whole-process parent/child read allowance; declared cumulative ceiling is **2,177,518,395 B**. Its new ReadMeter first charges absolute parent rchar, including setup/import overhead, and the inherited post-reap accounting avoids counting child reads twice. Terminal receipts add old plus new costs; an uncertain failure charges at least the entire new allowance. Limits remain sampled, so small detection overshoot is possible and must be reported, not hidden.

Before consuming the new session, the wrapper validates the three prior receipts, C2 COMPLETE, C3 exact read-limit failure, and both zero-survivor cleanup receipts. It freshly acquires the original flock and a new control-root flock; C3's original output must be absent both before and under those locks. The new single-use session and attempt markers reject a repeated execution. No historical PID is signaled. Actual small-control reads during this review confirmed the expected receipt hashes and absence of both new control root and C3 analysis output; launcher checks remain authoritative at execution time.

Inherited resource scope: CPU 0/1, aggregate RSS 2 GiB, 1800 s per invocation plus 40 s owned cleanup, CUDA disabled, one invocation only. Output remains **32 MiB of logical serialized bytes including controls**, using original analysis/control reserves; filesystem block/directory rounding is additional and is not represented as an enforced physical 32 MiB cap. A failed/partial result remains evidence and is not automatically retried.

Owner 13 tests plus 5 independent routing/accounting/lineage tests: **18 passed in 0.30 s**. Fixtures cover original child C3 routing and inherited limits, unexpected prior child accounting, C2 survivors, wrong C3 charge, exact new read-cap exhaustion, lock conflict, output absence, and refusal to execute a consumed session. The first independent test collection had a reviewer fixture relative-path error; its log is preserved, the fixture path was corrected, and no production source changed. No evaluation payload, image, model, tensor, GPU or live ledger was accessed.

The Lead may execute this exact already-authorized single retry using:

```bash
taskset -c 0,1 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C3_1024_RETRY_PREPARATION_v1/run_c3_once.py --execute
```

This review does not assert completed C3 scientific results or authorize another attempt, new scoring, C2 rerun, changes to statistical plans, or any training. Exact source changes invalidate this narrow conclusion.
