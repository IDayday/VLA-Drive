# Live Base audit — read-only

Capture: **Sun Oct 11 03:09:34 UTC 2026**, actual SSH host `training-vla-zt2` / hostname `training-vla-zt2-worker-0`. State: **RUNNING / KEEP_RUNNING**. This is a live inspection, not the old FORMAL_TRAINING_ACTIVE snapshot.

Launcher PID2124247 started2026-10-09 23:57:22UTC; eight rank PIDs2124255–2124262. Cwd `/mnt/project/DriveVLA-M0-iqe-20261009`, source HEAD1e876da, original `configs/iqe/main.yaml`. Command: `/root/miniconda3/envs/ddp/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=8 -m iqe.cli train-base --config configs/iqe/main.yaml --mode full --max-samples 101592 --device cuda`.

Live optimizer step **22796**, global batch32. Losses {"interaction": 0.009354081004858017, "current_dino": 0.01857159286737442, "future_clip": 0.33356934785842896, "ego_il": 0.16662904620170593}; gradient norm 4.785349. All finite. Latest fully validated checkpoint is step_022000.pt with its complete receipt and SHA25642320ea3e6fb09c6cd2b955dde7398a4033338a98a68bc4bac111fe8e5e01e93. It strictly loaded for real22k qualification. A newer filename without a verified receipt is never called complete.

Eight A80080GB GPUs showed83–100% instantaneous utilization, about43–45GB memory each. All eight GPU compute PIDs belong to this Base. **No competing task was found on the Base host.** Output disk and latest-step JSON are retained in `evidence/LIVE_BASE_FINAL.txt`. There is no observed numerical/data-denominator fault requiring discarded weights or a fresh Base. Checkpoint22k is22.8GB; saves continue on the original1000-step cadence.

The qualified execution profile used another free host, `training-vlawm-zt`, and finished before an independent structured-world task began there at roughly02:57UTC. Our already-running short single-GPU qualification completed afterward; no additional GPU dispatch was made onto those occupied devices. Existing other research jobs were preserved. User permits retiring inefficient training when needed, but no old Base was stopped and no new full Base was launched in this closeout. Prefer saved-checkpoint continuation if an optimized handoff is later executed; new-from-zero training discards22k+ completed updates.

Existing navtest observer PID1181750 was verified alive; start2026-10-10 17:45:57UTC, unchanged `SCHEDULE_PLAN.json`. Its existing20k/50k/100k plan was not expanded, stopped or redirected. Observed20k full navtest PDMS87.1802704/100 over12,146 scenes is historical monitoring evidence, separate from the32-scene22k engineering qualification. No test failure cases entered training or calibration.

No live training worktree, checkpoint, optimizer definition or monitoring plan was hot-modified. CPU diagnostic process1298470 was the only owned task stopped, after cmdline/cwd verification, to move our inefficient qualification forward to a separate GPU diagnostic root; its partial artifacts and migration receipt were preserved. This was not the ongoing Base.
