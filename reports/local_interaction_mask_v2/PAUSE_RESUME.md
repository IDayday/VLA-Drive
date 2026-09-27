# Pause and recovery boundary

**PAUSED_BY_USER.** Do not run training/evaluation without a new user instruction. Resuming these checkpoints continues the old design; it does not implement the proposed shared motion-supervision correction. A new recipe needs separate provenance and matched controls.

Three workers cooperatively stopped at optimizer step1887, epoch8, offset2016,60288presentations. Model/optimizer/scheduler/Python+NumPy+Torch+CUDA RNG/seen tokens were saved and CPU-deserialized for verification. Checkpoint hashes and exact worker argv are in [PAUSE_STATE.json](PAUSE_STATE.json). Exact resume is supported at an optimizer boundary with the same code/data/device topology; previous real resume tests are in ENGINEERING_EVIDENCE.md. No new resume experiment was run for this handoff.

The historical epoch8 metrics describe step1824, not the rolling paused checkpoints. The16pass target is incomplete.

## Safe read-only command now

```bash
cat /mnt/project/local-interaction-mask-v2-artifacts/20260927/formal_public_epoch24/user_pause_verified.json
```

## Old-recipe recovery — NOT_RUN, only after explicit user resume

Check actual processes, available devices, checkpoint hashes and remaining budget first. All waiting controllers are stopped, so none should be assumed to monitor a newly launched worker. Pause only authorized occupancy if needed; leave unrelated jobs alone.

For each arm, archive its STOP_REQUESTED under a unique name without overwriting history. Use the immutable training worktree/source955097588152912eeaca72eeb4ee9f736e40b530, unchanged worker argv (already includes --resume), a new ledger run ID and supervisor --initial-step1887. This preserves the already charged63updates. Preserve seed42/batch32/epochs16/schedule-epochs16 and all model/cache identities.

Concrete CURRENT recovery command, documented but not executed after pause:

```bash
cd /mnt/project/VLA-Drive-local-v2-runs-main
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="$PWD"
FINAL=/mnt/project/local-interaction-mask-v2-artifacts/20260927/formal_public_epoch24
LEDGER=/mnt/project/local-interaction-mask-v2-artifacts/20260927/budget_ledger.json
mv -n "$FINAL/P_CURRENT/STOP_REQUESTED" "$FINAL/P_CURRENT/STOP_REQUESTED.user_pause_20260927"
test ! -e "$FINAL/P_CURRENT/STOP_REQUESTED" && \
/root/miniconda3/envs/ddp/bin/python -m tools.local_interaction_mask_v2.supervise \
  --ledger "$LEDGER" --run-id formal_public_epoch24_P_CURRENT_resume_after_user_pause \
  --output "$FINAL/P_CURRENT" --gpus 1 --max-gpu-hours 1 --initial-step 1887 -- \
  /root/miniconda3/envs/ddp/bin/python -m tools.local_interaction_mask_v2.train_planner \
  --cache "$FINAL/train_refined" \
  --targets /mnt/project/joint-world-artifacts/20260927/extended_training/corpus_v1/world_targets \
  --meta-root /mnt/project/structured-world-v1-artifacts/20260926/dataset_v1/meta/train \
  --foundation "$FINAL/foundation/checkpoint.pt" --holdout "$FINAL/relation_holdout.json" \
  --graph-config configs/local_interaction_mask_v2/graph.json --mode current \
  --seed 42 --epochs 16 --schedule-epochs 16 --batch 32 --workers 2 \
  --output "$FINAL/P_CURRENT" --resume
```

ALL/MASK use their own saved argv and graph checkpoints from PAUSE_STATE.json, separate GPU/output/run IDs. Reserve remaining evaluation budget before continuing all three. Do not rerun completed foundation, features or32pass graphs. Before formal scores, complete actual-trained online parity and final model lock. Evaluation source is pinnedc66e6f86636377c575990598e4f33671e67fb495; no formal model scores exist and no controller is active. Inspect any controller's children/ledger before explicit recovery, never blindly restart a queue.

[Quickstart](../../docs/LOCAL_INTERACTION_MASK_V2_QUICKSTART.md) contains public-data rebuilding and configurable CLI paths. Machine-specific paths here are recovery locations. Git contains no weights, raw data or private scene visualizations.
