# Complete DDP foresight resolution study

Use `configs/foresight_resolution/c0.yaml` through `c5.yaml`. These all train ego FM, current DINO, future DINO and the frozen vehicle GT-MAE representation. The historical `configs/dino_tradeoff` configurations are current-only and must not be substituted. Historical 89.41 is not a reproduced result.

For a concise Chinese record of all six configurations, their token counts, shared recipe and actual training entry points, see [C0–C5 配置记录](DDP_FULL_FORESIGHT_CONFIGURATIONS_ZH.md).

For rebuilding the physical-time data index and frozen MAE interaction targets on another server, see [新服务器重建说明](DDP_FULL_FORESIGHT_REBUILD_TARGETS_ZH.md). Teacher weights/GT records remain external assets; a git clone alone does not contain them.

The implementation and real gradient/resume evidence were pushed before the full-model cost runs. Training source is an immutable worktree; this document and analysis tools may have a later report commit. Generic Qwen/DINO provenance, real teacher provenance and validation are in `reports/ddp_full_foresight`. No old driving policy initializes a student.

Latest user speed priority is active: two standalone8GPU runs (C0 primary, C1 companion), all16authorized cards. Actual16GPU socket training was3.68x slower than8GPU; that probe was safely stopped after41updates and is not labeled a complete profile. Both fresh formal runs continue to100000 with fixed milestone evaluation, without waiting at25000 for the remaining matrix. Training source is `d1d4085`, controller `fb474a7`. See [FAST_MAIN_20260929](../reports/ddp_full_foresight/FAST_MAIN_20260929.md) for frozen registrations, live commands and guarded resume. The earlier four-GPU queues below are historical and paused; do not restart them.

5000/10000 full-development results now exist: C0 PDMS78.39/84.20 and C1 79.89/84.45, each1696scenes/16logs with0failures. Both runs have passed16000updates and remain live. C1 was rescored locally to match C0's exact CPU evaluator runtime; every submetric matched the original scores exactly. An independent CPU-only observer `tools.full_foresight.canonical_scores` source68c1bb7 now handles future paired comparisons. The10000-update C1−C0 interval includes gains and losses; full Navtest and Action-Only/task ablations remain NOT_RUN. See [DEVELOPMENT_MILESTONES_20260929](../reports/ddp_full_foresight/DEVELOPMENT_MILESTONES_20260929.md) and its full scene/log CSVs.

## Verified execution

For a new server with NAVSIM logs/images and generic Qwen/DINO already installed, use the [raw-data self-preparation commands](DDP_FULL_FORESIGHT_SELF_PREPARE_ZH.md). No large Release or GitHub API token is needed. This creates a separate campaign and, without copied teacher weights, trains its own common MAE once.

Use the `ddp` Python environment. Set paths to the actual identity-checked assets; code does not download or accept an arbitrary model in place of the recorded generic checkpoint.

```bash
export ART=/mnt/project/ddp-full-foresight-study-artifacts/20260929
export LOCAL=/var/tmp/ddp-full-foresight-20260929
export QWEN=/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct
export SOURCES=/mnt/project/ddp-foresight-artifacts/20260928/generic/sources.json

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/full_foresight tests/foresight tests/dino_tradeoff
```

Real four-loss gradients and deployment equivalence passed at source `2bde06f`. On the same deterministic four-GPU setup, continuous four real updates and two plus two resumed updates produced identical model, FP32 optimizer, task RNG and per-step losses. These are startup checks, not learning or PDMS results. Evidence is `FOUR_LOSS_GPU.json` and `REAL_RESUME.json`; the two runs exposed 128 scenes each. Common weights calibrated on training data are current 1, future 0.9794244300709714 and interaction 0.8328945981862067. Only future/interaction weights warm up over 1000 updates.

The following actual profile entry has launched from source `593a9fd`. It checks data, cache, teacher and source identities, then borrows only verified pressure-script GPUs. Existing run/queue identities are rejected. Use a new `--host-tag` for an intentional repeat and an unoccupied port. Never rerun a recorded queue blindly.

```bash
cd /mnt/project/VLA-Drive-full-foresight-run-593a9fd
/root/miniconda3/envs/ddp/bin/python -m tools.full_foresight.profile_matrix \
  --campaign-root "$ART" --local-root "$LOCAL" --index "$ART/dino_index_v1" \
  --qwen "$QWEN" --sources "$SOURCES" --host-tag local \
  --pressure-script /mnt/project/gpu_stress.py \
  --pressure-python /root/miniconda3/envs/navsim/bin/python \
  --mode paired4 --base-port 29641
```

This executes C0 and C5 concurrently, four GPUs each, global batch 32, 20 warmup plus 100 measured updates. Four losses remain active. Current/auxiliary labels are local, DINO/MAE are not run online, and one Qwen forward supplies all losses. Profile weights never initialize formal training. The single4 and single8 modes use the same optimizer/global batch/objectives; those modes must be measured before selecting execution topology.

The measured cost report can be refreshed from an analysis checkout without changing the running source:

```bash
python -m tools.full_foresight.summarize_profiles \
  --campaign-root "$ART" --output "$ART/profile_progress.json"
```

The paired throughput uses the actual overlapping measurement window, not twice one job's rate. Missing/failed/incomplete profiles remain explicit. Training costs exclude target construction only in the explicitly labeled warm-cost estimate; the campaign ledger retains teacher verification, target extraction, loading, failures, saves and evaluation separately.

## Data and formal-run boundary

The verified teacher is the method's random-initialized 30-epoch vehicle MAE trained on 101592 training scenes; it is not a JointSceneFlow model. `tools.foresight.train_trajectory_mae`, `freeze_teacher` and `export_interaction_targets` provide its real train/freeze/ego-hidden export chain. `tools.full_foresight.verify_teacher` verifies source, data, weight identity and fresh masked encodings before reuse. Teacher limitations, including stationary-vehicle error, are retained in `TEACHER_REUSE.json`.

`tools.full_foresight.build_index` and `cache_dino_targets` prepare all h0/1/2/4 targets. Four actual DINO input resolutions produce six independently identified pooled caches. Compatible existing images require recipe and SHA checks. `tools.dino_tradeoff.stage_targets` stages immutable chunks locally; `tools.full_foresight.stage_interaction` stages the common MAE targets. Full formal runs require complete caches and the full manifest, not a profile prefix.

`tools.full_foresight.run_student --scope formal` additionally requires a frozen run registration with source-compatible configuration, common calibration/teacher identities, seed, total scheduler horizon, update endpoint, global batch and GPU budget. No formal plan or PDMS result is implied by passing the short checks. Register the complete screen/full plan after measured throughput and before ranking. At the screen boundary use `--stop-after` with the full `--updates` and `--schedule-updates`; continue the same run using `--resume --acknowledge-stop`, preserving optimizer, scheduler, RNG and data progress.

For evaluation, `tools.full_foresight.evaluate_auxiliary` reports h0/1/2/4 and copy-current separately. Pure-current deployment uses `tools.foresight.export_predictions`; official scoring uses `tools.foresight.score_pdms`; `tools.full_foresight.summarize_planning` aggregates matched full-method candidates. `tools.full_foresight.lock_navtest` requires complete registered endpoint development results before final Navtest. Current C0/C1 priority runs use source d1d4085. Full-population development results exist at5000/10000; final endpoint development and Navtest remain NOT_RUN. Auxiliary regression quality cannot replace those results.

The actual frozen registration, launch/resume commands, precision-matched deployment measurements and PDMS audit are documented in [FORMAL_START_20260929](../reports/ddp_full_foresight/FORMAL_START_20260929.md). Do not rerun an active slot. The user-provided GT audit distinguishes full-precision map/reference v1 scores from the historical EpisodeDrive quantized cache and from v2 EPDMS. Our bounded 33-case parity test and all1696 cache hash checks passed; this does not certify every official entry point or a future Navtest cache.
