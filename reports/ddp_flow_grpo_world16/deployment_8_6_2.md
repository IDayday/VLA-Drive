# GPU-cap deployment change, 2026-09-18

Requested caps: training-rl-zt4 uses at most6 GPUs and leaves devices6/7 free; training-rl-zt2 uses at most4 GPUs. Other users' workloads are untouched.

Target F allocation: local training-vla-zt devices0–7, training-rl-zt4 devices0–5, training-rl-zt2 devices5/6. The last two slots were idle at selection; every launch rechecks memory/utilization. rl-zt2 devices0–4 had unrelated allocations, so they were not selected. U retains training-vla-zt2 devices0–7 plus training-vla-zt3 devices0–7. Each variant remains world16, global scene batch16, accumulation1, group8, steps10, inner_epochs2,2000 actual optimizer updates.

CPU allocation: F local128 logical CPUs; rl4 cores0–47 and siblings64–111 (96 logical); rl2 cores0–15 and siblings64–79 (32 logical). U keeps128 logical CPUs per node. OMP/BLAS/MKL remain1 thread; reward workers1/rank (32 for both variants total); data workers0. No background GPU waiter or unrelated process termination.

The old controller was paused with SIGSTOP only, allowing both existing native jobs to reach their already scheduled complete update100. This prevents it from launching another old-placement segment. Old checkpoints and logs are retained. Before restarting, its job-specific supervisors must all have exited0, then its controller receives SIGTERM and SIGCONT. Administrative cancellation evidence is retained separately from numerical failures.

The source CUDA RNG inventory has8 states per process. Global ranks0–7 keep8; ranks8–13 project states0–5; ranks14–15 project states6–7 onto local0–1. Both native rank RNG and Accelerate RNG files are handled. Model, optimizer, scheduler, CPU/Python/NumPy RNG, stream cursors and pending behavior chains are unchanged. Only subdivision of original contiguous rank groups is supported. The new snapshot has its own seal and mapping receipt; original checkpoint files are immutable. No source checkpoint is overwritten.

`deployment.project_checkpoint` uses the real native checkpoint validator. CPU fixtures test active RNG correspondence, byte-identical model/optimizer, integrity, idempotence and interrupted publication. Actual installed torchrun is tested with heterogeneous2+1 CPU workers and Gloo. `paired.validate_plan` enforces caps and rejects duplicate slots. Descriptor migration requires unchanged native executable and recipe, matching prior layout and cutoff, and the source-bound GPU release. Ordinary `--resume` still rejects a changed deployment.

CPU:71 tests PASS, exit0 (11 deployment tests,8 explicit asset-reuse tests,24 previous cluster tests,28 affected native control/evaluation tests). Executed from the isolated deployment worktree; actor executable remains `854dbc2ecb67fb28a235ccc7238729f7560a7ccafead08d9f5307a626152153c`. These are not GPU acceptance or performance evidence. The historical BF16 candidate chunk1/2 failure remains unchanged.

GPU migration evidence is independently recorded under `runs/resource_reallocation_v3`. The actual16-rank NCCL probe passed. The original saved update1 was projected to8+6+2 and a bounded real official-reward update1→2 completed. The exact zero-tolerance comparison against the original8+8 update2 **FAILED** in17 of50 files (16 optimizer shards and forward model); all RNG/rank/cursor files and scheduler matched. This FAIL is preserved, with no relaxed historical tolerance. Equal pre-update losses and slightly different global clipping norms are consistent with FP32 collective summation-order sensitivity. A fixed-layout scope therefore requires a separate independent8+6+2 repeat and same-layout interrupted/resumed run. Its exact comparisons retain zero tolerance; they do not certify cross-layout bitwise equality. Source GPU qualification remains under resource_reallocation_v2; its raw tensor archive location is recorded in the prior report.

Commands (repository root, existing ddp environment and PYTHONPATH):

```bash
python -m scripts.cluster_flow_grpo.test_release
python -m scripts.cluster_flow_grpo.cluster run runs/resource_reallocation_v3/f16_split_nccl_spec.json
python -m scripts.cluster_flow_grpo.cluster run runs/resource_reallocation_v3/f16_split_resume_spec.json
python -m scripts.cluster_flow_grpo.boundary_evidence --continuous runs/resource_reallocation_v3/projected_expected/update_000002 --resumed runs/resource_reallocation_v3/f16_split_resume/checkpoints/update_000002 --output runs/resource_reallocation_v3/f16_split_resume_comparison.json
python -m scripts.cluster_flow_grpo.release --variant f
python -m scripts.cluster_flow_grpo.release --variant u
python -m scripts.cluster_flow_grpo.paired --plan configs/cluster_flow_grpo/paired_world16.json --resume --migrate-deployment
```

Diagnostic spec/control/output directories are single-use and refuse overwrite. A repeated diagnostic requires new attempt paths. The completed formal update100 is projected once on demand; update200 onward already has the new inventory. Release publication must pass before the migration command can launch formal work.

After the user's instruction to stop repeated file checks, `scripts.cluster_flow_grpo.assets` explicitly reuses the completed105GiB verification from the successful actual GPU diagnostic. The source execution context and successful control receipt are SHA-bound in the deployment plan. There is **no fresh full-corpus byte check** on these restarts. Processor/numerical/dependency/token and native checkpoint checks still execute. The wrapper changes only `verify_asset_manifest`, not model forwards, backward, sampling or optimizer code; its source and8 CPU tests are bound by the cluster release. Data changes require a new full verification receipt. This is a user-authorized deployment optimization, not proof that post-verification same-path data replacements would be detected.

The capped F fixed-layout verification runs are: `f16_split_resume` (original1→2), `f16_split_cont` (same original1→3, exposing another update2 for repeat comparison), and `f16_split_resume3` (new-layout2→3). Each is bounded to no more than2 actual optimizer updates. The original cross-layout FAIL remains a bound historical artifact. Final deployment acceptance requires both new exact within-layout comparisons, all original native GPU gates, and the current-source71-case CPU receipt.

Final qualification: **READY_FOR_THIS_PROFILE**, source-bound cluster identity `b8ec9631c70116e3f851b071f774df8ed4ad4d2a325c0faa0b8bf14ed1333cd9`. Both native release records contain17 satisfied semantic gates. Fixed8+6+2 repeat and resume each passed all50 native state files at zero tolerance. All3 comparison reports, including the cross-layout FAIL, are mirrored in `deployment_evidence/`. The production dispatcher is PID1331113, log `runs/resource_reallocation_v3/formal_pipeline.log`; the native CLI resumes F from its separate projected update100 and U from its original update100. The migration archive is `runs/paired_full_navtrain_world16/deployment_history/1789708115095684488/`.

Development results at update100,1696 scenarios, seed42, official v2 one-stage aggregation: F-SFT0.9362947686→F1000.9303079390; U-SFT0.9418673185→U1000.9348736715. Both are declines on this evaluation. They are not final navtest results and did not trigger parameter/seed/budget changes. Full per-scene paired tables are committed alongside the report. The scheduled2000-update experiment and fixed5-seed final evaluation remain in progress.

For future restarts of this same accepted deployment, use `python -m scripts.cluster_flow_grpo.paired --plan configs/cluster_flow_grpo/paired_world16.json --resume`; the migration flag is only needed for the first descriptor handover. Both variants are configured to continue to2000 updates, save every100 and evaluate every200. The original references remain their respective SFT models.
