Full-population payload analysis budget preparation: C2/C3 versus C0 at 1024

The payload scale fits a bounded CPU-only follow-up using the existing frozen CLI. This is budget preparation, not execution approval or a completed population-level stopping analysis. C2−C0 and C3−C0 remain the existing diagnostic comparisons at 1024, each with all 1696 fixed dev scenes / 16 logs. The 4096 endpoint, thresholds and stopping/selection rules remain unchanged.

| Existing run | Ledger RESULTS.pt files | Ledger bytes | Prediction NPZ files | Prediction bytes |
|---|---:|---:|---:|---:|
| C0@1024 | 424 | 44,996,032 | 1696 | 1,248,256 |
| C2@1024 | 424 | 44,996,288 | 1696 | 1,248,256 |
| C3@1024 | 424 | 44,996,864 | 1696 | 1,248,256 |

All three ledgers have 424 complete groups of four requests, 1696 admitted/completed requests, exact scene/log coverage and no additional retry admissions in the inspected metadata. Counts are receipt metadata; the complete result contents have not yet been loaded. The shared dev GT adds 1696 ego files totaling 2,511,776 bytes (1481 bytes each). No GT or prediction array was read in this preparation.

| Read measure | Bytes | MiB |
|---|---:|---:|
| Unique existing payload content, three runs plus one shared GT population | 141,245,728 | 134.702 |
| Three unique load_run + trajectory_diagnostics calls, no extra GT cache | 296,283,328 | 282.558 |
| Original CLI: C2_minus_C0 | 197,521,280 | 188.371 |
| Original CLI: C3_minus_C0 | 197,522,432 | 188.372 |
| Both original CLI commands, sequential | 395,043,712 | 376.743 |

The unique-content line is a lower bound on distinct existing files. The CLI estimates conservatively charge each full file size per source read-open: prediction hash plus two np.load openings (×3), result hash plus torch.load (×2), and ego label hash plus torch.load per run (×2). The second CLI rereads C0 and the shared GT. These are logical read estimates, not measured physical filesystem I/O. The smaller three-run function estimate would require reusing C0 in a thin analysis driver; no such driver was implemented, and the unchanged two-command CLI is the proposed path.

load_run first checks the full formal registration/approval/checkpoint/input/protocol identities and CSV. It then verifies every prediction, every admitted scoring receipt, complete per-scene ledger results and reference-v2 arrays. trajectory_diagnostics additionally verifies every dev ego label for ADE/FDE. Images, teacher tensors, metric-cache payloads and native weights are outside this read chain. A full analysis must retain all 1696 scenes and reject unresolved or changed records rather than drop them.

The registered stopping proxy uses official executed motion: reference raw progress ≥5 m, candidate raw progress ≤1 m and mean executed speed ≤0.5 m/s. Speed is mean(hypot(executed_states[1:,3], executed_states[1:,4])) over the 40 timestamps 0.1–4.0s. Those columns are velocity_x_mps and velocity_y_mps. Candidate raw progress is progress.raw_meters; reference progress is progress.reference_raw_meters. Planned trajectory displacement is not a speed substitute. The frozen implementation also requires exact paired reference-progress equality; this preparation does not weaken that gate or introduce a tolerance. Bootstrap stays 2000 paired-log replicates, seed 20261017.

The optional schema probe read exactly one 105,824-byte original RESULTS.pt bundle from the first lexicographic C0 ledger group. The receipt SHA was checked before CPU deserialization. That original bundle contains four records, so four were materialized; only index 0 was inspected and validated. The other three were not examined. Exactly one record passed frozen validated_reference_input, requested-trajectory hash, finite float64 executed_states 41×11, state-column and 0.1 s time-grid checks, and finite raw/reference-progress checks. No sample metric, stopping classification or population estimate was reported. The full saved execution arrays and raw progress remain unverified for the remaining population.

Proposed Lead registration: two CPU threads, two CLI commands run sequentially, RSS ceiling 2 GiB, existing 1800-second wall ceiling per pair (3600 seconds total), payload-read ceiling 512 MiB total, metadata/source-read ceiling 128 MiB total, and output reservation 32 MiB per pair. The original CLI does not itself enforce these outer resource ceilings; Lead must enforce the registered finite limits. Full-run CPU time, wall latency and peak RSS are NOT_MEASURED. These ceilings are a proposal, not measured performance or a resource PASS. The earlier full CSV audit used 627.85 MiB peak RSS and 8.872 seconds; this metadata/schema preflight used the measurements below. Those observations support reserving 2 GiB but do not certify the future full-payload peak. No new cache is needed.

COMMANDS.json preserves the exact already registered C2 and C3 CLI entries and their original plan pins. Their C0/treatment bindings, full population, protocol and statistics match the current immutable C4 execution binding used for this inventory. Both output directories were absent at preparation. The only executable entry is frozen 6714 tools.st_world_policy.evaluate_campaign; launch_authorized and execute_now remain false. After Lead registers the budget, the existing commands are:

```bash
# cwd: /mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017; run only after Lead budget registration
env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES= MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.evaluate_campaign --plan /mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation/full_dev_C2_C3_execution_v1/EXECUTION_PLAN.json --population /mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/dev/SELECTED_SCENES.json --control-score /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C0_step001024_dev1696_seed42_v1/official_score --control-export /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C0_step001024_dev1696_seed42_v1/export --treatment-score /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C2_step001024_dev1696_seed42_v1/official_score --treatment-export /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C2_step001024_dev1696_seed42_v1/export --comparison C2_minus_C0 --stage DIAGNOSTIC_1024 --scope FORMAL --output /mnt/project/st-world-policy-20261016/formal/evaluation/analyses/C2_minus_C0_DIAGNOSTIC_1024_dev1696_v1
```

```bash
# cwd: /mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017; run only after Lead budget registration
env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES= MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.evaluate_campaign --plan /mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation/full_dev_C3_execution_v1/EXECUTION_PLAN.json --population /mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/dev/SELECTED_SCENES.json --control-score /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C0_step001024_dev1696_seed42_v1/official_score --control-export /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C0_step001024_dev1696_seed42_v1/export --treatment-score /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C3_step001024_dev1696_seed42_v1/official_score --treatment-export /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C3_step001024_dev1696_seed42_v1/export --comparison C3_minus_C0 --stage DIAGNOSTIC_1024 --scope FORMAL --output /mnt/project/st-world-policy-20261016/formal/evaluation/analyses/C3_minus_C0_DIAGNOSTIC_1024_dev1696_v1
```

Preparation core cost: 17,515,917 metadata/source bytes, 105,824 payload bytes in one file opening, one validated record, two CPU-affinity slots, 615.04MiB observed peak RSS and 15.663s wall. Additional command/source inspection reads were small metadata. The current 128 MiB metadata and 16 MiB sample-payload limits were respected; the full-population phase has not run.

No simulation, scorer call, GPU, model/image/teacher/native-weight read, navtest, training change, new threshold, new comparison or dispatch occurred. Handoff parent: 3469890b0377ed73c7f41baed8da3ce5b6671765. Owned path: reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C2_C3_1024_BUDGET_v1/ only. Existing CSV-only reports retain their original verification boundary.
