# C1–C4 common4096 endpoint: existing full-payload analysis handoff

The analysis is already implemented and all four exact commands are already registered. **No new statistical framework, endpoint or scientific gate is needed.** Run `tools.st_world_policy.evaluate_campaign` from `/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017` at frozen source `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`. Its file SHA is `3ff851dcabd131d94fc22dc700802a68b4a6efd6fa0f345850ed2d9a8b8628a3`. `COMMANDS.sh.txt` contains all four original command lines copied without changing argv or environment; `HANDOFF.json` pins their existing source COMMANDS/EXECUTION_PLAN files and verifies each C0/treatment registration, approval and run identity.

`--stage COMMON_ENDPOINT --scope FORMAL` already means4096 for consequence cells. All plans use the same full1696 dev scenes/16 logs, original canonical score protocol, paired-log2000 bootstrap replicates and seed20261017. Control and treatment must both be4096; neither C0@1024 nor an unfinished treatment may substitute. Existing plans explicitly bind C2/C3/C4, so do not use the older generic v4 plan for those cells: its old NOT_REGISTERED metadata would be the wrong dependency.

## Actual input state,2026-10-18T02:42:40Z

| Arm | Observed training step |4096 export completion |4096 official score completion | Analysis dependency |
|---|---:|---|---|---|
| C0 |4096 COMPLETE |present |COMPLETE,1696 valid |Control ready; no rerun |
| C1 |3846 RUNNING receipt |absent |absent |Wait its4096 checkpoint/export/score |
| C2 |4096 COMPLETE |IDENTITY present, COMPLETE absent |absent |Existing exporter progressing; wait export and score |
| C3 |4055 RUNNING receipt |absent |absent |Wait its4096 checkpoint/export/score |
| C4 |3772 RUNNING receipt |absent |absent |Wait its4096 checkpoint/export/score |

These are timestamped small-receipt observations, not fresh PID health claims. The four predeclared COMMON_ENDPOINT analysis output directories were absent when checked. No duplicate analysis was launched. C2 can be analyzed first as soon as its original scorer writes a valid complete receipt; it need not wait for C1/C3/C4.

## Full-payload boundary, not CSV-only evidence

The unchanged analyzer's `_formal_binding` verifies each exact registered run, config, optimizer horizon, source and native checkpoint identity. `load_run` validates full CSV population and all prediction/score-ledger receipts and reads their saved prediction/executed-state arrays. `trajectory_diagnostics` reads the fixed dev ego targets for ADE/FDE and raw/reference progress. It does not load model weights, teacher cache, images or metric-cache payloads, and it does not invoke simulation/scoring.

The analysis reports canonical metrics, ADE/FDE, paired log intervals, zero repair/newzero/high-score degradation and progress/stopping diagnostics. The stopping proxy remains: reference raw progress≥5m, candidate raw progress≤1m, mean executed speed≤0.5m/s over0.1–4.0s. No threshold changes or candidate selection are introduced. Full unresolved/invalid rows cause explicit failure rather than a favorable valid-only subset.

## Existing bounded execution and the1024 wrapper limitation

The native analysis CLI is generic and needs no rewrite. Its existing per-pair command budget is1800seconds, with CPU-only operation and one numerical-library thread. Existing monitored full-payload execution used affinity0–1,2GiB aggregate RSS,32MiB total output including controls and40seconds cleanup allowance.

**Do not execute `FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2/run_fixed_pairs.py` for4096.** It intentionally hardcodes two C2/C3 DIAGNOSTIC_1024 commands, old hashes, old output paths and consumed-attempt namespaces. Likewise its one-off C3 retry is not a general4096 runner. The minimal reuse is the already-registered native CLI with the corresponding COMMON_ENDPOINT plan/score/export/output arguments; an outer dispatcher can reuse the existing supervisor and counters with those new command identities. Do not edit or relabel old1024 receipts or rerun their analysis.

Do not reuse the erroneous1GiB allowance for multiple cold CLI processes. The old two-pair session exhausted it and retained a failure. Its separately bounded successful C3-only retry used583,617,585bytes whole-process logical `rchar`,36.698seconds invocation wall and802,971,648bytes peak aggregate RSS, producing898,697bytes analysis output. Those are **historical1024 measurements**, not4096 measurements. The original one-CLI1GiB allowance and2GiB RSS/1800second/32MiB ceilings are a reasonable reusable execution envelope; allocate/measure it per new pair, not across all four. Four independent sequential pair invocations imply up to4GiB logical read allowance and128MiB reserved output, without rerunning any prior pair. This handoff does not itself allocate those resources or claim the native CLI enforces outer limits.

Retain one existing Lead-controlled bounded CPU session at a time when sharing resources. Once the actual full score inputs exist, use the existing command directly under the original resource monitor; no additional model test, bootstrap trial or scientific review is needed merely because the stage advances from1024 to its preregistered4096 endpoint. Failure keeps partial evidence and does not auto-retry.

## What was executed here

`inspect_commands.py` read only source and JSON metadata, checked the pre-existing command/plan lineage and actual file existence, and wrote this small handoff. `--help` passed on the identical analyzer source. No CSV, GT, prediction array, scoring ledger payload or model was loaded; no bootstrap, simulation, dispatch or process signal occurred. This is preparation for real full-payload analysis, not an analysis result.
