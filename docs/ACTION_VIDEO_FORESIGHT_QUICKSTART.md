# Action-conditioned video foresight

The existing Action-Only Qwen/DDP path retains current C1 DINO targets and the frozen vehicle-MAE teacher. A new auxiliary head predicts either eight independent DINO frame targets or four native V-JEPA2.1 tubelets. GT ego actions enter this head after current encoding; they never enter the planner. `action_plus_W` feeds the original DiT one sequence of eight action hidden states followed by 144 final W states, with its original projection applied once.

| Arm | Future target | Auxiliary GT ego action | Planner condition |
|---|---|---|---|
| S0 | Eight independent DINO frames | Absent | H_A |
| S1 | Eight independent DINO frames | Present | H_A |
| S2 | Genuine V-JEPA2.1 video | Absent | H_A |
| S3 | Genuine V-JEPA2.1 video | Present | H_A |
| S4 | Genuine V-JEPA2.1 video | Present | H_A + W |

All five retain current DINO and vehicle-MAE representation supervision. Deployment removes every auxiliary head and GT-action encoder, retains W, and executes only the original ego DiT. Driving modules start randomly; existing 100k students are used only for frozen diagnostics.

Commands below accept paths through CLI. Run from a clean, committed source checkout. `$AV_ROOT` is a new artifact directory, `$BASE_CONFIG` the resolved generic/random initialization configuration, `$TRAIN_DATA`, `$CURRENT_DINO`, `$DINO_INDEX` and `$INTERACTION_LABELS` the verified training assets. `$CLIP_LABELS` must match the requested arm. A formal invocation requires the complete training population and a complete native clip cache; engineering prefixes are rejected.

Actually executed real training, 4 GPUs, 4 continuous updates:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 \
python -m tools.action_video_foresight.run_experiments \
  --base-config "$BASE_CONFIG" --data "$TRAIN_DATA" \
  --dino-root "$CURRENT_DINO" --dino-index "$DINO_INDEX" \
  --interaction-root "$INTERACTION_LABELS" --clip-root "$CLIP_LABELS" \
  --campaign-root "$AV_ROOT" --run-id startup_continuous4_v1 \
  --arm S4 --calibration "$AV_ROOT/real_four_loss_action_video_v1.json" \
  --scope startup --limit 64 --updates 4 --schedule-updates 100000 \
  --gpus 4 --micro-batch 4 --master-port 29731 --milestones 0,4 --deterministic
```

A separate 2+2 run was explicitly resumed with the same source, world size and optimizer schedule using `--resume --acknowledge-stop --stop-after 0`. Final model, optimizer, scheduler, task/data progress and per-rank RNG are compared using:

```bash
python -m tools.action_video_foresight.verify_resume \
  --continuous "$AV_ROOT/students/startup_continuous4_v1/checkpoints/final_000004" \
  --resumed "$AV_ROOT/students/startup_resume2plus2_v1/checkpoints/final_000004" \
  --output "$AV_ROOT/real_resume_equivalence.json"
```

Formal execution uses the same wrapper, with no startup/profile weight reuse:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 \
python -m tools.action_video_foresight.run_experiments \
  --base-config "$BASE_CONFIG" --data "$TRAIN_DATA" \
  --dino-root "$CURRENT_DINO" --dino-index "$DINO_INDEX" \
  --interaction-root "$INTERACTION_LABELS" --clip-root "$CLIP_LABELS" \
  --campaign-root "$AV_ROOT" --run-id formal_S4_seed42_v1 \
  --arm S4 --calibration "$AV_ROOT/real_four_loss_action_video_v1.json" \
  --scope formal --updates 100000 --schedule-updates 100000 \
  --gpus 8 --micro-batch 4 --master-port 29844 \
  --milestones 0,5000,10000,25000,50000,75000,100000
```

The four full-model120-update topology profiles are complete. Single8 measured18.7364scenes/s; the actual overlapping2x4 window measured18.6262scenes/s. The five-host campaign therefore uses8GPUs/model to shorten the first complete comparison. Complete development clip caches already exist; full101592scene training clip caches remain a strict requirement. Startup/profile checkpoints are never formal initialization.

The host-local formal controller binds a frozen plan/source, waits for complete identity-matched native targets and local replicas, and then launches fresh formal training. It blocks unrelated compute processes and only releases ledger-verified pressure reserves. It exports registered development milestones with the canonical FP32-master/TF32off protocol; it does not start a Navtest observer. Controller/source CLI checks have been executed; formal progress must be read from the actual student ledger.

```bash
python -m tools.action_video_foresight.run_formal_campaign \
  --plan "$AV_ROOT/formal_plan_seed42_v2.json" --arm S4
```

Resume a safely paused controller/run only after checking its state, from the same immutable source and plan:

```bash
python -m tools.action_video_foresight.run_formal_campaign \
  --plan "$AV_ROOT/formal_plan_seed42_v2.json" --arm S4 \
  --resume-controller --acknowledge-stop
```

`stage_native_targets` copies only the required native cache onto a host's local disk, verifies byte hashes, and publishes local COMPLETE only after the full source population is complete. It can stage finished chunks while extraction continues. No teacher is rerun or target transformed. `run_experiments --local-image-root` uses the existing byte-checked current-image replica. Exact launches and resource meters live outside git; compact verified evidence is under `reports/action_video_foresight/`.

Use `tools.action_video_foresight.audit_existing_auxiliary`, `audit_W_usage`, `probe_representation` and `train_frozen_W_probe` for the prescribed read-only/probe diagnostics. Official PDMS remains canonical NAVSIM v1 FP32 master loading, TF32 off, ten FM steps, one ego candidate. Live BF16 timing is a separate deployment point and must not be paired with FP32 scores.


Formal deployment is now active from immutable1493ded. Check actual ledgers before running recovery commands; launching again while the controller owns its flock is an error. Snapshot and exact five-host commands: reports/action_video_foresight/FORMAL_START_20261002.md and FORMAL_CONTROLLER_LAUNCHES_V2.json. S2/S4 are performing real optimizer updates; other arms start after the full source/cache and GPU readiness gates. All old formal0 statements are historical.

The completed old-model W intervention scores are a fixed128scene development diagnostic and do not establish a new method gain. Frozen future probes use train-only means in the channel-LayerNorm loss space and a separately encoded static-current-repeat video reference. These diagnostic reference caches never enter formal student inputs.


Latest actual warmup evidence: FORMAL_FULL_WEIGHT_1000.json binds unchanged1493ded and real full-weight observations for S2/S4. S3 now performs formal updates too. Full1696-scene frozen video reference evaluation completed on2c4d659; changed-region/fullC1 and preselected128sceneC0 audits run on2df1341. These evaluation commits do not change the training source. Later hardening checks invalid arithmetic-mean references without applying a second normalization; original valid measurements and their identities are preserved.


All five formal controllers now perform real updates (18:00UTC snapshot), with complete source and local native caches. Their source remains1493ded. See `FORMAL_ALL_FIVE_RUNNING_20261002.md`; do not launch another controller on top of the existing live run. Source/report changes for finite-value checks, resource coverage and learning-curve rendering do not alter training. To regenerate a new aggregate curve snapshot without discarding completed journal records:

```bash
python -m tools.action_video_foresight.summarize_live_campaign --plan "$AV_ROOT/formal_plan_seed42_v2.json" --output "$AV_ROOT/new_progress_snapshot.json"
python -m tools.action_video_foresight.plot_learning_curves --snapshot "$AV_ROOT/new_progress_snapshot.json" --campaign-root "$AV_ROOT" --output-prefix "$AV_ROOT/new_learning_curves"
```

Both commands were executed against the actual formal ledgers; use new output names to preserve old evidence. The plot uses every completed record up to the recorded snapshot, including a final partial averaging window. It does not rank different target families by raw MSE.


For already complete development exports whose original CPU scoring failed at SSH transport, use the independent canonical-host recovery. This is actually deployed on immutable6659529; no model is loaded and no training source is edited. It preserves the original failed states, binds inherited C1 checkpoint identity to the S0–S4 formal registration, and limits concurrency to32CPUworkers. Read its observer state and flock before restarting; do not duplicate the live service.

```bash
python -m tools.action_video_foresight.reconcile_development \
  --plan "$AV_ROOT/formal_plan_seed42_v2.json" \
  --reference-score "$AV_ROOT/scores/formal_S4_seed42_full100k_v2_dev10000_seed42/summary.json" \
  --workers 16 --slots 2 --interval 60 \
  --recovery-directory reconciled_development_v2
```

This command must run on the plan's canonical scoring host, from a clean source. Corrected score/ego sidecars retain the original failed-state hash and are included by `summarize_live_campaign`; incompatible identities or failed scenes are rejected. It never starts a Navtest observer or retries GPU inference.
