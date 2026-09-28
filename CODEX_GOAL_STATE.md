# Shared foresight campaign — ACTIVE

Goal: DDP Action-Only with64shared Wqueries, future FLUX latent supervision and frozen GT vehicle-MAE interaction-latent supervision. Deployment retains W, original ego FM head only. No old driving weights, joint actor generator, online detector, scorer or video/depth generation.

Development worktree: /mnt/project/VLA-Drive-ddp-foresight-20260928
Branch: feature/ddp-shared-foresight-gtmae-20260928
Artifacts: /mnt/project/ddp-foresight-artifacts/20260928
Old experiments/controllers remain sealed. Only this new branch may be pushed; no large artifacts/private images.

## Actual completed work

- Independent current/ego train101592scenes/1176logs and dev1696/16logs;0failures, token/log-disjoint. Full raw vehicle-only teacher data prepared with the same ordered split. No student input contains teacher/GT future data.
- Teacher real continuous4 vs2+2resume passed;8updates total. Teacher64scene small-fit512updates completed with actual learning. These weights are NOT the full teacher initialization.
- Real public Qwen gradient test: main FM and both auxiliary readouts reach W and language. Auxiliary labels synthetic only in that one plumbing test. Current-input prediction is invariant to poisoned extra target fields and to removing auxiliary heads while retaining W.
- Real R/A/B/C/D initialization pairing passed: shared language/driving tensors identical; A–D Widentical; no driving weights loaded; caller RNG preserved.
-26focused CPU tests pass. Actual2GPU NCCL empty-rank and accumulated auxiliary normalization match single-process gradients.
- Student micro1 real continuous4 vs2+2at087139a passed exact model/optimizer/RNG equality. A batch>1deterministic CUDA indexed-write failure was then reproduced before any optimizer update; fixed using unique-position scatter at56bf61e.
- At frozen56bf61ec044850e4500a3d550b7205df5451253c: actual Qwen padded batch/current images/DeepStack/mRoPE vs separate FP32 inference passes, ego max difference4.77e-6. Four-GPU global32/micro4 and micro8 each completed4real updates. Micro8 median3.56s/update, rank0peak19.35GB. SAME-source micro8 continuous4 vs2+2also exact for model,4optimizer shards, all rank RNG and data progress.
- Student startup total20real updates/416presentations; failed batch32v1 had0updates. No formal student run. Startup checkpoint FP32-master export tested on2dev scenes,0failures; no planning claim.
- Actual results: reports/ddp_shared_foresight/student_preflight and batching_fix. Complete auxiliary evaluation, training-only gradient calibration and registered teacher-freeze tools implemented; real target-dependent execution NOT_RUN.
- Official NAVSIMv1.1 CPU scoring passed on2real startup dev exports,0failed. This is only pipeline evidence. Five-inference-seed aggregation and paired log-cluster comparison tools are implemented/tested; full benchmark NOT_RUN. See planning_pipeline.
- Teacher learning snapshot at2755updates/705040presentations, fixed dev milestones through4: ego-with-peer ADE.888m vs1.040m after peer removal; vehicle3.265m vs5.214m. All7342fixed queries retained,0failures. Teacher GT-conditioned diagnostics are not student planning results. Curves, role fallback and stratified counts in teacher_progress/snapshot_v1.
- Offline ego ADE/FDE evaluation passed on the same2startup dev predictions,0failures. Pure-current Navtest cache12146scenes/136logs audited read-only: all JSON/three camera files exist, official metric population identical, no train/dev token or log overlap, no old model predictions/features reused. No Navtest model was evaluated. See planning_pipeline/NAVTEST_CURRENT_AUDIT.json.
- Formal Navtest locking now verifies complete matched R/A/B/C/D seed42 endpoints, exact five-inference-seed development models, shared data/exposure/schedule and immutable evaluator/model/population/noise identities. Main selection is the common registered complete endpoint; its actual update length remains provisional until matched short fits. Unit counterexamples pass; real formal lock NOT_RUN.
- VAE source registration now requires the pinned public config Git blob plus exact official weight SHA256, rejecting empty inventories or arbitrary self-attested weights. Recheck still finds no configured HF token and official gated401. Public metadata stored in generic/flux_public_metadata_recheck_v1.json; no weights downloaded or access restriction bypassed.
- Complete real future-frame audit atcb49722:101592train/1696dev scenes,0failures, no scene dropped. All have at least one valid1/2/4s horizon;101344train and1695dev have all9future views. Valid timestamps have all3image files; remaining missing cases are timestamp gaps masked only on label side. Maximum train timestamp deviation16.415ms (<registered50ms). No VAE encoding/model inference occurred. Audit v1's legacy meter field inference_scenes records103288file checks; not model predictions. Future code uses checked_scenes. Raw ledger retained unchanged; see FUTURE_FRAME_AVAILABILITY.json.

## Live process: DO NOT duplicate

Full GT-MAE teacher PID1346936, localGPU0, source d3c05d136949ecc14b1a8312fa12bf966ae03643 in /mnt/project/VLA-Drive-foresight-run-d3c05d1. Fresh random initialization.30epochs/11910updates registered;11910must complete before milestone selection. Read ps plus teacher_full30_v1/steps.jsonl and status.json for actual progress. At last review epoch8was complete; fixed dev summaries through8preserved in teacher_progress. At8, ego-with-peer ADE.80757m and vehicle2.75044m, vs removed-peer1.00292m/4.94131m,7342queries,0failures. Do not edit this worktree or alter its ledger.

Full command is teacher_full30_launch.json. Resume ONLY after its process is terminal and actual status inspected, using the identical frozen command plus --resume --acknowledge-stop. Never silently restart or duplicate it.

Bounded continuation controller PID1414766 is WAITING_FOR_TEACHER; artifacts/postteacher_v1_launch.json and postteacher_v1_status.json are authoritative. It uses frozen bffeaea2a6c8b4ea0c94809f53b971a7c183b6de from /mnt/project/VLA-Drive-foresight-run-bffeaea. After the existing teacher process exits COMPLETE11910, it freezes by the registered30epoch rule, releases only verified GPU1pressure parent1374704 with matching start ticks, exports fresh ego-hidden dev/train interaction targets and restores pressure. It will NOT start students, select an early teacher or restart a paused teacher. Do not duplicate these exports. A changed GPU1pressure lease causes PAUSED for inspection, not a kill of its replacement. Explicit stop files: campaign STOP_REQUESTED or postteacher_v1_STOP_REQUESTED. The controller waits at most28hours and each export is limited to7200seconds/global6000GPUh. Waiting consumes no GPU allocation.

## Resources / budget

Existing local and vla-zt2 GPUs authorized; no paid expansion. Idle cards on BOTH hosts must run pressure scripts. Before use verify exact parent command/PID and stop only its pressure process. Every test launcher restores pressure after exit; teacher-completion monitor1358057 does the same forGPU0. Current parent IDs are in gpu_pressure and *_launch.json; re-check /proc because PIDs can be reused. Never stop unrelated workloads. Pressure occupancy is recorded separately from scientific training cost.

New campaign ceiling6000GPUh; phase limits in execution_budget.yaml. Ledger under artifacts/runs includes loading, failed allocations, tests, teacher training and exports. Read it to compute actual remaining budget; do not use oldcampaign quota. Teacher source has total-run exposure in its resumed attempt records: aggregate exposures from each logical run's authoritative steps, not by summing cumulative attempt exposures.

## Blocking dependency and next work

FLUX.1-schnell official VAE at pinned741f7c3ce8b383c54771c7003378a50191e9efe9 returns gated401. No verified local artifact found on local/vla-zt2. User was asked for an authorized local path/configured HF access; no answer yet. Do not bypass access restrictions or substitute a different visual teacher. Qwen public source is verified.

Continue independent work: full teacher training, complete result/plot/paired-metric tooling, official dev/Navtest export/scoring integration. After full teacher completes, freeze via registered rule and export fresh ego-hidden8x512labels. Once VAE is legally available, validate deterministic scale/shift/reconstruction, cache all horizons, calibrate shared-gradient weights on training data, run matched short fits, then freeze the complete A/B/C/D/R formal plan. Formal training length100000/global32is still provisional, not launched. Use docs/DDP_FORESIGHT_QUICKSTART.md for tested commands and clearly marked NOT_RUNcommands.

No formal A/B/C/D/R checkpoint, full development planning score, Navtest score or positive planning conclusion exists. Keep the goal active; do not present startup tests as the completed experiment.
