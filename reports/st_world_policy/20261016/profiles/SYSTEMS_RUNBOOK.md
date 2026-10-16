# Bounded profile evidence and commands

This runbook documents actual completed diagnostic runs. It does not authorize a fresh formal launch, further topology search or resuming PROFILE weights into a formal model. No dynamic cache, IQE expert/scorer or old MOPD task was started.

Timing runtime: `/mnt/project/VLA-Drive-st-profile-runtime-20261016`, source58f85044. Trace/short-screen runtime: `/mnt/project/VLA-Drive-st-profile-trace-20261016`, sourceab2184bf. Analysis tool revision0c54ed2c is read-only and does not require modifying either runtime.

The recorded complete commands/environment/PIDs are the canonical replay evidence:

- `profiles/sp2_pca384_8gpu_timing_v1/REGISTRATION.json` and original launch evidence.
- `profiles/sp2_pca384_4gpu_a_timing_v1/SYSTEMS_LAUNCH.json`, GPUs0–3, port29641.
- `profiles/sp2_pca384_4gpu_b_timing_v1/SYSTEMS_LAUNCH.json`, GPUs4–7, port29642; launched concurrently with A.
- `profiles/sp2_pca384_8gpu_trace_v2/SYSTEMS_LAUNCH.json`, all8GPUs, port29643,4updates/trace2+3.
- `profiles/sp2_pca384_4gpu_micro4_screen_v1/SYSTEMS_LAUNCH.json`, GPUs0–3, all8locks for isolation, port29644,32updates.

All paths above are relative to `/mnt/project/st-world-policy-20261016`. Completed run directories must not be overwritten or blindly rerun. The trainer rejects existing outputs without explicit native resume. For interrupted diagnostics the recorded immutable source, registration, environment and global batch remain mandatory; native `--resume` uses this run's own complete receipt. An acknowledged operator stop additionally needs `--acknowledge-stop`. Completed budget-limited runs do not acquire a larger budget by resuming.

## Offline summaries

From the analysis worktree, the actual commands are:

```bash
python -m tools.st_world_policy.profile_pipeline --output /absolute/new_summary.json summarize --ranks /absolute/rank0.json /absolute/rank1.json /absolute/rank2.json /absolute/rank3.json /absolute/rank4.json /absolute/rank5.json /absolute/rank6.json /absolute/rank7.json
python -m tools.st_world_policy.profile_pipeline --output /absolute/new_aggregate.json aggregate-two-by-four --first /absolute/A_summary.json --second /absolute/B_summary.json
python -m tools.st_world_policy.profile_trace --trace /absolute/existing_trace.json --output /absolute/new_trace_summary.json
```

Use explicit existing per-rank files from the above run's `runs/<run_id>/profile/`; these examples intentionally require a new output path, because immutable old analyses remain evidence. `summarize` checks the full240 updates and shared identity; aggregation requires actual simultaneous windows, not multiplied solo rates. The existing completed summaries are listed with hashes in `MEASUREMENT_EVIDENCE.json`.

## Actual nonrecycled I/O command

Each rank0…7 was started simultaneously with the following command, substituting its explicit rank and unique output. Runtime: `/mnt/project/VLA-Drive-st-loader-runtime-20261016`, sourcec4c3e3f. Process receipts and exact commands are in `audits/systems/loader256_8proc_v1/SYSTEMS_LAUNCH.json`.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false timeout --signal=TERM --kill-after=10 180 /root/miniconda3/envs/ddp/bin/python -m reports.st_world_policy.history_data.loader_cost --data-root /mnt/project/ddp-foresight-artifacts/20260928/student_train_v1 --manifest /mnt/project/st-world-policy-20261016/audits/history_data/train256_current_cost_v1/preparation/SELECTED_SCENES.json --teacher-root /mnt/project/st-world-policy-20261016/audits/history_data/train256_current_cost_v1/native384 --pca-root /mnt/project/st-world-policy-20261016/audits/history_data/train256_current_cost_v1/pca_fixed384_scope_v1 --output /absolute/new_rank0_output --max-scenes 256 --workers 2 --prefetch-records 4 --torch-threads 2 --shard-count 8 --shard-index 0 --stage full
```

The completed measurement has exact disjoint256 coverage; rerunning it is unnecessary for this result. Do not label it physically cold or convert its scenes/s into model-training throughput.

## S-P3 separate placement

Prepared remote registration `sp3_pca384_8gpu_timing_v1` was **not launched** by this reviewer: Tsinghua-zt became occupied by protected real FM-transfer evaluation. Original registration remains unchanged.

Lead requested the separate local registration `sp3_pca384_8gpu_local_timing_v1`, SHA256 `27fcf3da96653724eebdcac3c0d7c475c8d62f1d96c21742c5091df87d034966`, runtimeab2184bf, W96/PCA384, global32/micro4/world8, original20k LR,240update/3600s invocation budget. `COMMAND.json` contains exact cwd/argv/environment and port29656. It uses all eight `/tmp/foresight_gpu_N.lock` files and an outer3900s timeout with a120s cleanup allowance. Lead alone may dispatch after a fresh allowed-host, process, UUID, lease and output check. Systems did not launch it or sign its unobserved result. Lead subsequently launched this exact local registration: PID286276, `LEAD_LAUNCH.json`, sourceab2184bf. All240 updates and the complete native checkpoint subsequently finished; the process exited at BUDGET_PAUSED. Actual whole-update mean2.060298s, p502.045717s, p952.188342s,15.524013scenes/s. Final save46.381613s. See `SP3_W96_ACTUAL_PROFILE.md` and `SP3_MEASUREMENTS.json`. Do not launch a duplicate or initialize formal training from PROFILE weights.

Final FORMAL approval must be created by the independent reviewers against the final clean source/config/data/resource fingerprint. Current reports deliberately contain no fabricated PASS approval or final-population teacher coverage claim.
