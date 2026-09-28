# DDP vehicle-only from-scratch campaign — FORMAL TRAINING RUNNING

Branch:feature/ddpolicy-vehicle-joint-from-scratch-20260928
Reference base:632cf74c4c7269228d66d569e0c54846b07c95f4
Development:/mnt/project/VLA-Drive-ddpolicy-vehicle-joint-20260928
Artifacts:/mnt/project/ddpolicy-vehicle-joint-artifacts/20260928
Frozen FORMAL source:4f27cbb5b83806331325aa473a01da4fd1738d60
Frozen FORMAL worktree:/mnt/project/VLA-Drive-ddpolicy-campaign-20260928
Small-fit source:4843e4ddf8340ecc8b44a47fc5fc9b688a58e3a8
Small-fit worktree:/mnt/project/VLA-Drive-ddpolicy-optimizerfix2-20260928
Report/viz commits are separate from training provenance; do not change either running worktree.

## Active controller — DO NOT DUPLICATE

PID1328493 on training-vla-zt-worker-0. Read the live status before any action.
Plan:formal_campaign_v1/plan.json
Plan SHA256:422e5cbaf777ac03bd59eee397263545b71cbaf8202dad3d9f7613fa6b1f57d9
Controller:formal_campaign_controller_v1/status.json
Controller log:formal_campaign_controller_v1.stdout.log

The complete100000-update campaign has started. formal_C_seed42_001 is performing actual full-manifest updates on vla-zt2GPU4/5/6/7. A42 waits for local0/1/2/3; B42 waits for local4/5/6/7. The controller starts each automatically when its assigned devices are idle. No small-fit weights are reused. The512-update small fits finish concurrently and independently.

Primary A/B/C42 all have fresh generic/random initialization, full101592-scene training, globalbatch32/microbatch4,100000updates, warmup5000/cosine100000, AdamW1e-5, final10000all-hidden for B/C. B/C43 follow as a complete pair, subject to the campaign cap; A43 is not planned. Every full pass3175updates,24-scene tail. Per-run exposure3199752. Head xy scale1 remains fixed. No old driving foundation, Reader, graph, scorer or hidden cache is loaded.

Existing local/vla-zt2 resources only. Remote0/3 belong to other tasks and are untouched. Registered total cap8000GPUh (optional cap preference received no reply; user changes override it);300GPUh held for final evaluation. No rental/expansion. Formal training cap7698GPUh includes all prior campaign cost. Actual4GPU startup measured12.4sec/update and41.56GiB peak; five-run training extrapolation6889.6GPUh, with initial ego-only graphs/cached data. This is not a completion-time guarantee.

Controller behavior:24h allocations safely pause/save and resume the same identity; immutable milestones0/1000/5000/10000/25000/50000/75000/90000/100000, periodic1000. Explicit STOP_REQUESTED is preserved. It will evaluate full dev at the common25000/50000/75000/100000grid, select max mean PDMS over42..46 (later checkpoint on ties), freeze the models/protocol, then run complete Navtest and paired analyses. Temporary evaluation checkpoint copies are task-owned and released after complete scoring; original checkpoints stay intact. No Navtest model predictions exist yet.

## Concurrent diagnostic runs

small_fit_fixed_A_seed42_001 local0/1; B local4/5; C vla-zt2GPU1/2. Each same64training scenes/globalbatch16/microbatch2/512updates; final64all-hidden, fresh initializations. Actual statuses and progress are in training/<run>/status.json. Diagnostic cap40GPUh includes failures, tests, loading and extraction; old20GPUh registration is retained historically. The cap uses the shared ledger, so formal spend also contributes while diagnostics finish.

Real fixed-seed camera inference: at64updates egoADE A7.317/B10.037/C10.707m, all64scenes each,0failures. At128updates B ADE6.276/FDE10.252m; detection1/333 under the fixed2m evaluation gate, joint-selected matched0. These are training diagnostics and poor early detection coverage, not planning or method-gain evidence. Later logs have nonzero joint vehicle supervision and actual C role tasks. A/B256 and C128 exports are running; evaluate their saved banks after completion. Keep full denominator/miss rows.

Real-output audit:256camera predictions have ego decoded from the SAME joint slot0, max rounding difference2.38e-7; all unmodeled channels exact zero. Private first8scene figures exist at private_figures/B_step128_first8, never push them.

## Critical historical correction — DO NOT RESUME INVALID RUNS

small_fit_A/B/C_seed42_001 stopped at179calls each; small_fit_scale_B_seed42_001 at34; startup_B_2gpu_001 at4. These575calls were optimizer no-ops. Installed DeepSpeed FusedAdam uses int32 tensor-size metadata; one2GPU ZeRO partition exceeded2^31. Real GPU counterexample and exact kernel hashes are recorded. All raw logs/costs/exposures remain intact with optimizer_stasis_correction_v1/validity_overlay.json. The corrected trainer uses same-hyperparameter groups bounded to500M elements and verifies actual FP32-master changes before counting EVERY update. No shared environment was patched. Provisional head scale20 was NOT selected.

Corrected C continuous4 and independent2+2: all RNG states equal, full FP32-master max difference4.95e-6, predeclared tolerance FAILED. Same-parent repeat (startup_fixed_C_sameparent_001) performs only2new updates after2inherited updates; sampled100000FP32 masters max1.18e-6 and all RNG states match. This narrower check is NOT uninterrupted or bitwise equivalence. Preserve the full failed comparison. Current CPU suite30passed/1CUDA-marked skipped; separate real CUDA RNG test previously passed.

## Data and evidence

Trainval103288/1192logs; train101592/1176logs,dev1696/16logs; zero token/log overlap. Navtest12146/136logs is disjoint from trainval. Vehicle targets:vehicle_targets_v1_complete, identity59a0f36f22f33cc0e10bc82328191d126a9acfdc89b8c59b3931873ec212a423,0failures. Raw log population available; vehicle name filter precedes capacity32. Generic depth covers all103288scenes. Current-only caches:current_dev_v1,current_navtest_v1,current_smallfit64_v1. No annotations/futures enter camera prediction.

Offline evaluation labels: evaluation_vehicle_targets_smallfit_v2 (64/0fail) and evaluation_vehicle_targets_dev_v1 (1696/0fail,8624ROI/FOVvehicles). Core box/track/future labels equal the training cache on64scenes; training-only supervision_grid is not rebuilt. One raw log has a temporal gap; relative ego/vehicle metrics explicitly mask mismatched timestamps. Original-framework ego labels/normalization remain unchanged. The failed earlier eval-label attempt is retained. Navtest label-side generation is deferred until final model lock.

Public evidence:reports/ddpolicy_vehicle_from_scratch/corrected_learning_preflight/RESULTS.md plus JSON records. The registration snapshot says NOT_STARTED because it predates the real launch; live controller/status files are authoritative. Full dev/Navtest scores remain NOT_RUN. Published scores and historical weights do not replace this experiment.

## Resume only if controller actually stopped

```bash
cd /mnt/project/VLA-Drive-ddpolicy-campaign-20260928
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m tools.ddpolicy_vehicle.campaign --plan /mnt/project/ddpolicy-vehicle-joint-artifacts/20260928/formal_campaign_v1/plan.json --directory /mnt/project/ddpolicy-vehicle-joint-artifacts/20260928/formal_campaign_controller_v1 --resume --acknowledge-stop
```

If FAILED, inspect the saved error/log before restarting. Do not resume invalid/no-op runs or start duplicate formal trainers. Formal model files, caches, raw data and private images stay outside git. The public official-token partition is now provided as NAVTRAIN_PARTITION.json; derived paired score CSVs may be published as requested results, without trajectories or annotations. Push only this task branch; never merge/force-push.
