# Fixed COMMON4096 CPU analysis handoff

Prepared by `/root/spatial20k_evidence`; no dispatch was performed. This is a small
adapter for the four existing preregistered commands, not a new analysis method.
The source-only / metadata-fixture tests passed (9 tests). No new CSV, trajectory,
score ledger, model, GT payload or simulator was opened/run in this preparation.

## Absolute reviewed sources

- Original bounded pair runner:
  `/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2/run_fixed_pairs.py`
  SHA256 `2a47c91254707d3390bb74f92516b9cb5f48bf64d2365d2e874c81a6064a0c5f`.
- Successful C3-only retry frontend:
  `/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C3_1024_RETRY_PREPARATION_v1/run_c3_once.py`
  SHA256 `0a21f2286b45361572f2c55ed866f66c276e268b3273593d2d72ba44eac3223e`.
- The monitor implementation it used is `monitor()` in
  `/mnt/project/VLA-Drive-st-c4-cpu6-execution-20261017/reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py`
  SHA256 `8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644`.
  This retained copy is byte-identical to the original reviewed monitor. The
  wrapper's historical path under Lead now has SHA256 `4b2f0fcc...`; using that
  moving path would fail its existing checksum check. The new frontend only
  relocates this dependency; it does not update the accepted checksum or code.
- CPU child body reused unchanged from
  `/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C4_1024_PREPARATION_v1/run_c4_once.py`
  SHA256 `c4806201c46caee4a770545fa6e00764a4b5da5ba65ea97b389268b08be14c5f`.
- Frozen native analyzer:
  `/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017/tools/st_world_policy/evaluate_campaign.py`
  SHA256 `3ff851dcabd131d94fc22dc700802a68b4a6efd6fa0f345850ed2d9a8b8628a3`,
  runtime commit `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`.

The historical successful C3 run's receipts remain at
`/mnt/project/st-world-policy-20261016/audits/evaluation/FULL_PAYLOAD_C3_1024_RETRY_v1/SESSION_TERMINAL.json`
and `C3_minus_C0/RESULT.json` below that directory. Their current small-file hashes
are recorded in `DEPENDENCIES.json`. Its previously inspected result was 36.698 s,
802,971,648 bytes peak aggregate RSS, 583,617,585 charged whole-process read bytes
and 898,697 analysis output bytes. Those are historical 1024 measurements, **not**
new 4096 timings. The preceding failed shared allowance remains separate evidence;
this preparation does not retry either historical analysis.

## Exact adaptation and retained limits

`run_common_once.py` selects only C1, C2, C3 or C4 versus C0 at COMMON_ENDPOINT.
`FIXED_COMMANDS.json` pins the original command files and their unchanged argv,
environment, cwd, 1800-second limit, plans and population. C1's old command lacked
the wrapper's later `analysis_plan` metadata key; the adapter adds its already
registered plan reference without changing argv. All original statistics remain:
1696 dev scenes, 16 logs, paired log bootstrap 2000 with original seed20261017,
executed-motion/stopping diagnostics, all arms at4096.

The original `run_pair`, output guard, read accounting, observer, process cleanup
and aggregate-RSS monitor are reused. Completion validation is the original
function with only its stage literal changed from DIAGNOSTIC_1024 to
COMMON_ENDPOINT; a source-AST equality test checks this. Inputs require complete,
valid1696 scoring and matching actual export/checkpoint/source identities.

Each pair gets at most one analysis invocation: 1 GiB logical reads (including its
binding process), 2 GiB aggregate RSS, 1800 seconds plus the original40-second
cleanup allowance, and32 MiB total output including controls. Up to four pairs
therefore allow128 MiB total outputs and4 GiB reads. The native paired outputs
retain the original ~15.94 MiB cap; controls reserve16 MiB for RESULT and64 KiB
for actual bindings, admission and session receipts. A shared nonblocking serial
lock prevents these four adapters running concurrently on the fixed CPUs0/1.
No automatic retry or fresh allowance for a consumed session is supported.

The preparation does **not** claim 4096 has been analyzed or the new wrapper has
passed a live workload. Source/fixture checks are current; native analysis remains
pending the actual scoring dependencies and Lead execution/review.

## Commands after the corresponding official score completes

Use this absolute script (or its identical cherry-picked copy):

```bash
/root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-spatial20k-evidence-20261018/reports/st_world_policy/20261018/consequence4096_bounded_preparation_v1/run_common_once.py --help

# Example: C2 can run first; no need to wait for C1/C3/C4.
# --bind reads only small actual receipts, pins them immutably, and never dispatches.
/root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-spatial20k-evidence-20261018/reports/st_world_policy/20261018/consequence4096_bounded_preparation_v1/run_common_once.py --arm C2 --bind

# Lead executes only after checking the generated actual binding and resource availability.
/root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-spatial20k-evidence-20261018/reports/st_world_policy/20261018/consequence4096_bounded_preparation_v1/run_common_once.py --arm C2 --execute
```

For C1/C3/C4 change only `--arm`. Each invocation keeps that arm's original plan,
source identities, output and endpoint. No native evaluator overrides are exposed.
Bindings/session receipts go under
`/mnt/project/st-world-policy-20261016/audits/evaluation/COMMON4096_FIXED_PAIRS_v1/C2/`
(respectively C1/C3/C4). Analysis outputs are the predeclared
`formal/evaluation/analyses/C2_minus_C0_COMMON_ENDPOINT_dev1696_v1` directory under
the campaign root, with only `ANALYSIS.json` and `PAIRED_SCENES.csv`.

No actual binding was generated here because C1–C4 scoring completion receipts
were absent at02:54:25 UTC. C0 scoring COMPLETE exists; C2 export identity exists.
`DEPENDENCIES.json` records presence only, not a scientific completion judgment.
The original analyzer checks all rank export receipts and complete payload joins
when invoked. Missing native `export/COMPLETE.json` is not a dependency: exports
use `COMPLETE_rankN.json` by schema.

Tests (already run, no workload):

```bash
python reports/st_world_policy/20261018/consequence4096_bounded_preparation_v1/test_preparation.py
```

`TEST_RESULTS.txt` retains the full 9-test output; `CLI_HELP.txt` records the actual
CLI. No training, export, score job, CSV/bootstrap computation or model test was
started. No scientific endpoint/metric/seed/budget was added or altered.
