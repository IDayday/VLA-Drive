Prepared one C3-only retry for independent resource review. No retry or payload analysis has been dispatched. Parent commit: `20ab82fb505fa9ce44ca168e8177b5a09c96532a`; owned path: this new preparation directory only.

The delta imports the exact reviewed wrapper SHA `2a47c91254707d3390bb74f92516b9cb5f48bf64d2365d2e874c81a6064a0c5f`, verifies the original two commands, and selects only the original second command: C3−C0 at DIAGNOSTIC_1024. It reuses that wrapper's `run_pair`, original `--child C3`, output/read guards, cold-import sequence, counters and core SHA `8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644`. The imported module's control-directory variable changes in memory; no existing source file is modified. The child still runs the same frozen 6714 evaluator with identical plans, inputs, output location and statistics.

The new control namespace is `/mnt/project/st-world-policy-20261016/audits/evaluation/FULL_PAYLOAD_C3_1024_RETRY_v1`. It is created only by a reviewed Lead execution. The original failed namespace stays unchanged. Before consuming the new marker the adapter requires:

- Exact prior SESSION_TERMINAL and C3 RESULT hashes, failure type, return code and the recorded 1,103,776,571-byte charge; the exact C2 COMPLETE receipt is also pinned.
- Both prior owned-cleanup receipts report zero live processes. A fresh nonblocking exclusive acquisition of the original inherited RUN.lock must succeed and remains held through the new invocation. This uses current lock ownership and reviewed cleanup evidence; it never signals or assumes identity from a historical numeric PID.
- The original C3 analysis output directory is absent. Existing complete or partial C3 output is rejected; it is not overwritten or used to spend a second attempt.
- The new namespace has no consumed marker/attempt. Every completed, failed or interrupted new session rejects another `--execute`.

The separately authorized allowance is **1 GiB whole-process `rchar` for this one C3 CLI**, including this adapter, imports, the unchanged C0/C3 input validation and the child. It does not rerun C2. The old 1,103,776,571 bytes remain declared separately and are added to the new measured/charged cost in the terminal receipt. The declared campaign ceiling for these two accounting boundaries is 2,177,518,395 bytes. This is an explicit new allowance following the underestimated earlier two-pair budget, not a reset or an automatic retry of the failed session.

All other limits are inherited: affinity CPUs 0–1, empty CUDA visibility, aggregate RSS 2 GiB, 1800-second wall limit plus the existing 40-second owned cleanup grace, and 32 MiB logical serialized output including control records. The 64 KiB small-control reservation contains ADMITTED at most 32 KiB plus SESSION_STARTED/TERMINAL at most 16 KiB each. RUN.lock is empty. The existing one-second resource sampler can overshoot between checks; failures and their actual final costs remain recorded. Physical `read_bytes`, shared-library mmap traffic and filesystem allocation rounding are not relabeled as logical payload bytes.

The unchanged comparison still covers all 1696 dev scenes / 16 logs, 2000 paired-log bootstrap draws, seed 20261017, and the original progress/stopping definitions. It does not score, simulate, access navtest, change a threshold, select a checkpoint or stop training. Raw payloads are read only if Lead later runs the original CLI under this reviewed allowance.

After independent resource review, the Lead command is:

```bash
taskset -c 0,1 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C3_1024_RETRY_PREPARATION_v1/run_c3_once.py --execute
```

Preparation validation: 13 CPU tests passed in 0.26 seconds; `--help` passed. Tests use small controls and temporary lock/receipt fixtures, including single-command routing, success/failure cost lineage, consumed-session rejection, old-lock contention, wrong prior status/charge/cleanup/return code, output-presence rejection and original 32 MiB accounting. The mocked execution tests never spawn an analysis child. Original evaluator/cold-import correctness is inherited from the reviewed exact wrapper; it was not retested against payloads.
