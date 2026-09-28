# DDP vehicle joint from-scratch campaign — IMPLEMENTING

Branch: feature/ddpolicy-vehicle-joint-from-scratch-20260928
Reference base: 632cf74c4c7269228d66d569e0c54846b07c95f4
Worktree: /mnt/project/VLA-Drive-ddpolicy-vehicle-joint-20260928
Artifacts: /mnt/project/ddpolicy-vehicle-joint-artifacts/20260928
Official framework reference: youngzhou1999/DriveDreamer-Policy @ 8cefcac46e5944add529e1be19cba78bc06cc2bd

New authorization: full camera driving A/Base, B/vehicle Joint, C/Joint+role auxiliary from generic public modules and RANDOM driving modules. No old driving checkpoints, controllers, scorer, RL, second ego execution head, nonvehicle tasks or new BEV. Old V3 campaign and ledgers remain sealed; its unused 48 GPU-hour quota does not apply.

Actual progress: clean isolated branch; official recipe and generic local assets located. Full metadata count verified 103288; fixed split 101592 train /1696 dev, no missing metadata, no token/log overlap. Independent vehicle schema and generic initialization checks under implementation. No training or planning results yet.

Resource state: authorized local/vla-zt2 GPUs mostly run unrelated real inference and will not be stopped. Recheck before allocation. Explicit total GPU-hour cap requested asynchronously and pending; continue code/data/startup work, do not silently inherit old quota or launch unbounded formal runs.

Next: verify generic hashes and full auxiliary data availability; new vehicle GT cache; original DDP DiT joint dimension and current-camera Qwen route; tests and real camera gradient/throughput checks; register full cost/phase plan before long training. Commit/push this branch only, no private data/weights.

Real optimizer updates: 0. Campaign GPU-hours: 0. CPU tests pending.
