This adapter is ready for Lead review. No population analysis, scoring, GPU work, model/image read or cache-ledger operation was executed in this preparation.

Parent commit: `63d7bd557fbca9172a461eba74b55e0d86ac2fb4`. Owned path: this new directory only. The prior budget, commands and all scientific definitions remain unchanged. The implementation invokes the two original frozen CLI commands in order: C2−C0 and C3−C0 at DIAGNOSTIC_1024, complete dev 1696 scenes / 16 logs, 2000 paired-log bootstrap draws, seed 20261017. These diagnostics do not select checkpoints or stop the 4096-step comparisons.

Lead execution command after reviewing the exact manifest hashes:

```bash
taskset -c 0,1 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2/run_fixed_pairs.py --execute
```

The original plans, arguments and output directories are read from the pinned `FULL_PAYLOAD_C2_C3_1024_BUDGET_v1/COMMANDS.json`. The analysis child runs unchanged `tools.st_world_policy.evaluate_campaign` using `runpy`, from frozen `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`. This preserves `load_run`, `paired_statistics`, every reference/execution-array verification and the original stopping definitions. No new science or source approval is added by this adapter.

Resource envelope:

- Two sequential invocations; CPUs 0–1, empty CUDA visibility, aggregate parent/child RSS acceptance limit 2 GiB and wall limit 1800 seconds per pair. The reviewed monitor samples once per second; these sampled limits can overshoot between samples. Its existing owned-session cleanup allows 40 additional seconds and only signals this invocation's verified children.
- **1 GiB total whole-process `rchar`**, including taskset/Python/Torch imports, parent preparation, both child processes and output verification. The prior estimate of payload read-opens is 395,043,712 bytes (376.743 MiB); 128 MiB was the metadata/source estimate. The remaining allowance covers imports and supervision. These are estimates, not observed full-run costs, and `rchar` is logical I/O rather than physical disk traffic or mmap traffic.
- The existing SHA `8674aea1…` core supplies `monitor`, `terminal_read_accounting` and `charge_after_reap`. Each pair gets a new wall/read baseline and only the remaining total read allowance. The parent's post-wait `/proc/self/io` already includes its child; it is charged once, and the second pair does not reset the total. Incomplete accounting consumes the remaining allowance conservatively.
- Stdout and stderr each retain at most 1 MiB per pair. The inner CLI stdout is capped at 64 KiB to fit the final envelope. Analysis files have a 32 MiB combined serialized-write cap checked before each write, including the temporary JSON; a hardlinked publication is not counted twice. Control receipts are separate, serialized before publication, at most 16 MiB each to accommodate escaping of retained pipe prefixes.
- The exact frozen RSS helper is inherited from the reviewed metadata runtime; only `process_tree_rss` is called. No storage, admission, policy, ledger or migration API is invoked.

Invocation records will be created only when Lead executes, under `/mnt/project/st-world-policy-20261016/audits/evaluation/FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2/{C2_minus_C0,C3_minus_C0}/`. A campaign-local flock serializes this adapter. Each pair consumes one marker before spawning. Failure preserves its marker, receipt, bounded diagnostics and partial output, stops the sequence, and never retries or overwrites. A subsequent call can validate already-complete small output files and return without rerunning the CLI; a failed or incomplete consumed attempt is refused.

The source-specific read guard allows only the registered dev ego labels, the two fixed prediction roots and existing scorer `RESULTS.pt` bundles as binary payloads. It rejects model, teacher, image and GPU-device reads. This guard is a restriction around the fixed trusted CLI, not a replacement for the CLI's provenance validation or an OS sandbox.

Validation: 18 CPU tests passed in 0.16 seconds; `--help` passed. Tests cover fixed command/pin changes, sequential wait accounting, incomplete accounting, exact serialized output limits, retained partials, immutable JSON publication, forbidden payload paths, complete-output validation, failure/no-retry behavior, and two tiny owned CPU children with exit codes 0 and 3. All generated data were small fixtures under `/tmp`. No full-run time, RSS, import cost or scientific result has been measured here.
