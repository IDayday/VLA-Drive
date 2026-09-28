# DDP vehicle joint from-scratch campaign — SMALL FIT RUNNING

Branch: feature/ddpolicy-vehicle-joint-from-scratch-20260928
Reference base: 632cf74c4c7269228d66d569e0c54846b07c95f4
Worktree: /mnt/project/VLA-Drive-ddpolicy-vehicle-joint-20260928
Artifacts: /mnt/project/ddpolicy-vehicle-joint-artifacts/20260928
Official framework reference: youngzhou1999/DriveDreamer-Policy @ 8cefcac46e5944add529e1be19cba78bc06cc2bd

New authorization: full camera driving A/Base, B/vehicle Joint, C/Joint+role auxiliary from generic public modules and RANDOM driving modules. No old driving checkpoints, controllers, scorer, RL, second ego execution head, nonvehicle tasks or new BEV. Old V3 campaign and ledgers remain sealed; its unused 48 GPU-hour quota does not apply.

Implemented: original803M-parameter action DiT with joint actor-time dimension;32 current-camera vehicle queries and post-Qwen box/motion heads; predicted vehicle graph; role auxiliary every4updates at0.1 without replacing main FM; original Wan/PPD auxiliary tasks; ZeRO2 with bound checkpoint/RNG/data identity; FP32 current-only export, official v1 CPU scoring, vehicle miss denominators and log-paired analysis.

Verified: pinned public generic weight bytes/revisions; A/B shared driving initialization and B/C complete driving initialization equality;20CPU tests and one actual CUDA RNG test. Real full-camera forward/backward A/B;16 startup optimizer updates across continuous4, resumed2+2,2GPU4,8GPU4, with176 scene presentations. Full103288 labels built with0 failures, split101592/1176logs train and1696/16logs dev, zero token/log overlap. Current-only dev1696 and Navtest12146 input caches complete. No Navtest model prediction has occurred. Two-scene dev FP32/official scoring smoke completed,PDMS0, not a trained-model result. Resume stochastic states match; BF16 continuous/resumed parameters differ by at most2.98e-8, so bitwise equivalence is NOT claimed.

Active runs: `small_fit_A_seed42_001`, `small_fit_B_seed42_001`, `small_fit_C_seed42_001`, local GPU0/1,2/3,4/5 respectively. Immutable source5076336f657980d46631657d2d2dfa8c5c007281 in `/mnt/project/VLA-Drive-ddpolicy-smallfit-20260928`. Each independently starts generic/random, uses the same first64 train tokens, globalbatch16/microbatch2/two GPUs, max512updates,lr1e-5, diagnostic warmup32/horizon512, final64all-hidden. Diagnostic checkpoints MUST NOT initialize formal models. Temporary whole-campaign diagnostic ceiling20GPU-hours includes prior work. Check actual training/*/status.json before any restart; do not duplicate runs or edit active source.

Generic depth extraction is complete: both final shards `depth_full_shard{0,1}_002` finished, and all103288 expected token files exist. Current-only train64 input cache is also complete. B milestone64 camera export is running on vla-zt2 GPU4 from frozen evaluation source2f7b3c3; all evaluation costs count toward20GPUh. Original paused attempts and existing files remain unchanged.

Formal A/B/C seed42 plus B/C seed43: NOT_RUN. Official recipe100000updates/globalbatch32, warmup5000, final10000all-hidden. Eight-GPU startup measured6.6–7.0sec/update after first-step overhead: about7500GPU-hours for five training runs by extrapolation, excluding complete evaluation/I/O variance. Full campaign hard cap is8000GPU-hours, registered autonomously within existing GPU authorization after the optional preference question received no reply. User changes override this default. This is not a scientific result or completion-time promise. Small fitting does not satisfy the full experiment.

Next: finish the already running fits and depth shards; evaluate the shared small-fit endpoints on train64, report learning/role/coverage honestly; publish evidence and push this branch. After the learning diagnostic, freeze the full source/recipe and execute fresh formal models within the registered8000GPUh cap, then full dev and locked Navtest. No historical controllers; no Navtest tuning.

Read-only progress command:
`python -c 'import pathlib,json; r=pathlib.Path("/mnt/project/ddpolicy-vehicle-joint-artifacts/20260928/training"); print([(p.parent.name,json.loads(p.read_text())["status"],json.loads(p.read_text())["real_optimizer_updates"]) for p in sorted(r.glob("*/status.json"))])'`

Commands: docs/DDPOLICY_VEHICLE_FROM_SCRATCH.md. Evidence: reports/ddpolicy_vehicle_from_scratch/startup_evidence_20260928/. This state is a launch snapshot; refresh after runs finish.

Train-only scale diagnosis: B/update64 camera inference has0/333 detected GT vehicles and zero selected graph actors; all predicted x centers remain within[-0.012,0.184]m. One bounded correction decodes head xy in20m units, retaining graph thresholds/assignment/common parameters. See reports/ddpolicy_vehicle_from_scratch/SMALL_FIT_SCALE_DIAGNOSIS.md. Prepare new B correction run with128-step pause, then matched B/C512 only if actual joint supervision begins. Original runs remain unmodified. Diagnostic maximum40GPUh is included in the full8000GPUh ceiling.
