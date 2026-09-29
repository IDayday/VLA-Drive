# Shared foresight campaign — VAE_AVAILABLE; TEACHER_AND_INTERACTION_TARGETS_COMPLETE

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

## Completed teacher and automatic continuation: DO NOT duplicate

Full teacher source d3c05d136949ecc14b1a8312fa12bf966ae03643 completed30epochs/11910updates/3047760presentations. The registered selection chose milestone_030.pt, SHA2569730278f9c920de649536dec108226ac9d7f27430e265e8befac19896d58336c. Frozen manifest: artifacts/frozen_teacher.json. Selected dev ego-with-peer ADE0.5654269m; vehicle-with-peer1.8375817m,0failedqueries. These remain GT-conditioned teacher diagnostics.

Controller postteacher_v1_status.json is COMPLETE. It used frozen bffeaea2a6c8b4ea0c94809f53b971a7c183b6de and exported interaction_dev_v1 (1696scenes, identity6cda0349749b09f8fe6b62d1f977d8dbab8f3a8a178432047081468ece331502) and interaction_train_v1 (101592scenes, identityee7d2d3a2910290df56fc75528cb58d87a3876dace557687cea69626826d6474), both0failures. No student was started. GPU pressure restoration identities are in gpu_pressure/local_after_teacher.json and local_after_postteacher.json; re-check actual processes before any allocation.

## Resources / budget

Existing local and vla-zt2 GPUs authorized; no paid expansion. Idle cards on BOTH hosts must run pressure scripts. Before use verify exact parent command/PID and stop only its pressure process. Every test launcher restores pressure after exit; teacher-completion monitor1358057 does the same forGPU0. Current parent IDs are in gpu_pressure and *_launch.json; re-check /proc because PIDs can be reused. Never stop unrelated workloads. Pressure occupancy is recorded separately from scientific training cost.

New campaign ceiling6000GPUh; phase limits in execution_budget.yaml. Ledger under artifacts/runs includes loading, failed allocations, tests, teacher training and exports. Read it to compute actual remaining budget; do not use oldcampaign quota. Teacher source has total-run exposure in its resumed attempt records: aggregate exposures from each logical run's authoritative steps, not by summing cumulative attempt exposures.

## VAE dependency resolved on2026-09-29; next work

The prior BFL endpoint401 was real, but it was not the only authorized distribution. The Diffusers team publicly publishes exactly the same config and VAE weight bytes at diffusers/FLUX.1-vae, revisionda548cfb003bdeebaff6da0211fc8fbc67cb563a. Download completed and matched the original public config Git blob and weight SHA256. The scientific VAE identity remains the pinned BFL generic model; actual download provenance is explicit. No driving-adapted weight was used.

Local root: /mnt/project/ddp-foresight-artifacts/20260928/generic/flux_vae
Identity: /mnt/project/ddp-foresight-artifacts/20260928/generic/flux_vae_identity.json
Evidence: reports/ddp_shared_foresight/vae_download/DOWNLOAD.json and CPU_LOAD_CHECK.json.

The actual VAE loaded and encoded/decoded three real current training images on CPU. Exact repeated encodings, unchanged RNG, frozen parameters and eval enforcement passed. Dimensions16x32x57at short-side256, stride8, scaling0.3611, shift0.1159. Zero optimizer updates and no GPU used. Full future encoding and copy-current evaluation remain NOT_RUN.

Next: inspect a few actual reconstruction images offline; complete train/dev future-latent caches using the pinned VAE and existing timestamp/file audit; calibrate shared gradients with both real target types; run matched A/B/C/D short fits; then freeze and run the complete common formal plan. Existing interaction targets must be reused after identity validation, not retrained/exported unnecessarily. Formal100000/global32 remains provisional until those checks.

Reproduce the download:

```bash
cd /mnt/project/VLA-Drive-ddp-foresight-20260928
/root/miniconda3/envs/ddp/bin/python -m tools.foresight.download_flux_vae \
  --root /mnt/project/ddp-foresight-artifacts/20260928/generic/flux_vae \
  --identity-output /mnt/project/ddp-foresight-artifacts/20260928/generic/flux_vae_identity.json
```

No formal A/B/C/D/R checkpoint, full development planning result or Navtest result exists. Full objective remains unfinished; the previous VAE access blocker is resolved.

## Separate requested DINOv3 download — COMPLETE on2026-09-29

DINOv3 ViT-L/16 LVD-1689M, public timm distribution revision30c1109559f65dea34316b0d4842d35c5771fe11. Local artifact generic/dinov3_vitl16_lvd1689m_timm under the same campaign artifact root;1,212,347,640weight bytes, SHA25645172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941. Config, model card, license and provenance saved. This is timm format with documented QKV/RoPE differences from Meta format. It does not replace the registered FLUX target or change the student study.

CPU strict loading passed with no missing/unexpected keys;303,079,424parameters. Real current training image produced finite1x261x1024tokens and1x1024embedding. Repeated output exactly equal, RNG unchanged. No GPU used;0optimizer updates. Evidence and executable downloader/load-check commands: reports/ddp_shared_foresight/dinov3_download. No weight or private image is committed.
