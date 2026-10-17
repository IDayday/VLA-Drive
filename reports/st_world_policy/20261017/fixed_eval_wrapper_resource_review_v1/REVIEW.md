# Independent fixed C2/C3 analysis wrapper review

**PASS for the exact bounded execution source**, owner commit `04ef69768bdd49d4a39733ba08ae18ef99165eb5` (parent `41826094045964bf7ca5b70552a923a1a6e7267c`). Wrapper SHA256 `2a47c91254707d3390bb74f92516b9cb5f48bf64d2365d2e874c81a6064a0c5f`. Reviewed core remains `8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644`; frozen evaluator remains `3ff851dcabd131d94fc22dc700802a68b4a6efd6fa0f345850ed2d9a8b8628a3`.

Independent agent `/root/st_metadata_resource_review` used an isolated worktree and modified only its report/test namespace. No actual analysis was launched, no real payload was read, and no ledger/policy/runtime file was modified. This review inherits the previous frozen-plan and payload-budget review; it does not repeat or change scientific definitions. Both comparisons are C2/C3 versus C0 at1024,1696 dev scenes/16 logs, not C0@4096. Full-run metrics/time/RSS remain unmeasured.

## Narrow findings closed

The original18 owner tests passed, but the first source had three integration gaps retained in `INITIAL_REVIEW.json`: Torch/dill cold import was denied access to `/dev/null`; control output was additional to the32 MiB analysis allowance; and a new invocation could resume after a completed prefix with a new read allowance. The final source closes these without changing the evaluator:

1. Fixed dependency imports occur inside the owned monitored child with binary payload read protection active, before enabling the restrictive analysis-output write guard. Ordinary import-time `/dev/null`/standard-library temporary operations are permitted. Afterwards only the original immutable analysis outputs can be written. Imports remain charged and monitored; this is a restriction around a fixed trusted CLI, not an OS sandbox.
2. The32 MiB serialized output envelope per pair now includes15.9375 MiB analysis writes,16 MiB escaped RESULT and64 KiB other controls. C2's latter reserve covers ADMITTED≤32 KiB plus shared START/TERMINAL≤16 KiB each; RUN.lock is empty. Temporary analysis JSON is charged at write time; hardlink publication is not charged twice. These are logical serialized bytes; ordinary directory/allocation rounding and ephemeral library import files are stated separately, not mislabeled as zero cost.
3. A campaign-local lock covers the two sequential comparisons. One immutable session marker precedes them; any consumed session, successful or interrupted, refuses another `--execute`. No incomplete-prefix estimate is used to renew the1 GiB allowance. At the first invocation, already-complete original output can still be identity-validated and reused without a child. Failed/partial artifacts are retained and not retried.

## Tests and observed resource evidence

Final **29 tests passed** (21 owner +8 independent),4.20s pytest time /4.33s harness. New independent checks cover consumed-session files/directories, the inclusive32 MiB arithmetic, payload rejection while import writes are enabled, cumulative second-pair remaining allowance, complete-output reuse with zero child launches, and actual guarded cold imports. Test files total31,733 B. Parent affinity0/1 and a4 GiB RSS guard/300-second alarm were used. The parent's sampled RSS was36,921,344 B; the recorded Linux self ru_maxrss is an inherited/high-water diagnostic rather than this short parent sampling measurement.

The independent cold subprocess executed the final wrapper's fixed preload, the original evaluator `--help`, and the original immutable JSON writer for a49-byte fixture. Empty payload allowlist remained active; no evaluation data/model/image was opened. Observed: exit0, no monitor failure,3.925s, aggregate parent/child RSS650,993,664 B, child lifetime rchar136,622,435 B, correctly charged post-wait parent+child136,997,763 B, zero owned survivors. `INDEPENDENT_COLD_IMPORT.json` preserves the pre/post-wait evidence. This checks real import/write compatibility, not full1696-scene execution. The original owner's separate cold-import evidence is also source-pinned in its manifest.

The unchanged reviewed monitor uses owned pidfd/session identities, live RSS/read checks and bounded pipes. It signals only verified child processes. Its terminal parent rchar already includes the reaped child; `charge_after_reap` adds later parent activity without adding that child again. The first comparison and parent setup consume the same ReadMeter used by the second; only the remaining1 GiB is passed to it. Missing terminal accounting consumes the remaining budget and stops the sequence. A new session cannot reset it.

RSS acceptance is2 GiB aggregate, wall1800s per pair plus existing40s cleanup, CPU0/1, empty CUDA; monitor sampling is1s, so this is sampled fail-closed enforcement with possible between-sample overshoot. Parent final output verification is included in accepted wall/read totals. Logical rchar does not equal physical read_bytes/mmap traffic. The former512 MiB payload and128 MiB metadata estimates remain explanatory components of the larger1 GiB whole-process cap, not independently measured runtime counters or additive renewed allowances per pair. No future cache reservation is involved.

## Exact Lead command

Only Lead may execute once after its normal resource scheduling. At this review's small stat snapshot, both analysis roots and the wrapper control root were absent; the program itself rechecks consumed state under its lock. The command uses the owner clean source path with the SHA above:

```bash
taskset -c 0,1 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2/run_fixed_pairs.py --execute
```

It runs the two unchanged pinned commands sequentially, writes bounded receipts, and stops on any failure. Do not delete a failed session or rerun `--execute` to recover it; a later explicitly reviewed attempt must retain all old evidence. Actual complete analyses must provide their own observed limits and paired results. No automatic training, scoring, navtest or deployment permission follows from this review.
