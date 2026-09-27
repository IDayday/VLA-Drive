# Joint trajectory world model

Current stage: working image-conditioned joint flow and prediction-only planner bridge;64-scene masked/control pilots completed. BEV task objectives and planning comparison remain pending. No result below is a PDMS improvement claim.

## Environment and tests (executed)

```bash
cd /mnt/project/VLA-Drive-masked-trajectory-world-20260927
export PYTHONPATH=.:/mnt/project/DriveVLA-M0/nuplan-devkit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DEPTH_MODEL_CKPTS=/mnt/project/DriveDreamer-Policy/depth_model_ckpts
export JOINT_PYTHON=/root/miniconda3/envs/ddp/bin/python
"$JOINT_PYTHON" -m pytest -q tests/joint_world tests/structured_world_v1p1
```

19 tests passed. Covers whole-actor masks, hidden context poisoning including NaN, non-ego permutation equivariance, empty surrounding labels, gradients, graph save/load and previous coordinate/append-tail regressions.

## Real GPU path (executed, passes)

Use a NEW output directory and run ID to repeat; the ledger rejects duplicates.

```bash
CUDA_VISIBLE_DEVICES=0 "$JOINT_PYTHON" tools/joint_world/check_real_pipeline.py \
  --world-checkpoint /mnt/project/structured-world-v1p1-artifacts/20260927/W_POST_64_bounded696_d537401/checkpoint.pt \
  --graph-config configs/joint_world/image_graph.json \
  --targets /mnt/project/structured-world-v1p1-artifacts/20260927/targets_v6_full128 \
  --output /mnt/project/joint-world-artifacts/20260927/real_pipeline_native_noise \
  --ledger /mnt/project/joint-world-artifacts/20260927/budget_ledger.json \
  --run-id real_pipeline_native_noise
```

This is one actual scene, three current images, frozen released Qwen/DiT and previously trained current world Reader/heads. It runs two charged optimizer updates. Gate0, future-label pollution independence and independently initialized restore are exact. First-step gate receives ego gradient; second-step graph receives ego gradient. This verifies connectivity only. Original BF16 ego noise is preserved explicitly while the learned condition residual remains FP32; the default original action API is unchanged. Earlier dtype/parity failures remain in the ledger.

## Frozen current-only cache and paired graph training (executed)

Conditions use a strict whitelist and do not contain future GT. They may be reused only while the upstream Qwen/vision/Reader/heads remain frozen. The graph checkpoint contains the source cache identity and separate labels fingerprint. Online inference reconstructs the same features from images.

```bash
CUDA_VISIBLE_DEVICES=1 "$JOINT_PYTHON" tools/joint_world/extract_conditions.py \
  --world-checkpoint /mnt/project/structured-world-v1p1-artifacts/20260927/W_POST_64_bounded696_d537401/checkpoint.pt \
  --graph-config configs/joint_world/image_graph.json \
  --manifest /mnt/project/structured-world-v1-artifacts/20260926/overfit_tokens.json \
  --output /mnt/project/joint-world-artifacts/20260927/image_conditions64 \
  --ledger /mnt/project/joint-world-artifacts/20260927/budget_ledger.json --run-id image_conditions64
```

Exact paired training source is pinned at dfe8122 in `/mnt/project/VLA-Drive-joint-runs-dfe8122`. Both variants use seed42, batch8,1000 updates,8000 scene presentations/125 passes. Only all-hidden probability differs:0.5 for random actor masking;1.0 for all-hidden control. Fixed evaluations at0/500/1000. These train the graph only; the bridge and original ego head are not being optimized in these pilots.

```bash
cd /mnt/project/VLA-Drive-joint-runs-dfe8122
CUDA_VISIBLE_DEVICES=1 "$JOINT_PYTHON" tools/joint_world/train_graph.py \
  --cache /mnt/project/joint-world-artifacts/20260927/image_conditions64 \
  --targets /mnt/project/structured-world-v1p1-artifacts/20260927/targets_v6_full128 \
  --data-root /mnt/project/structured-world-v1-artifacts/20260926/dataset_v1 \
  --config configs/joint_world/image_graph.json \
  --output /mnt/project/joint-world-artifacts/20260927/graph_randommask64 \
  --ledger /mnt/project/joint-world-artifacts/20260927/budget_ledger.json \
  --run-id graph_randommask64 --steps 1000 --batch 8 --all-hidden-probability 0.5
```

Control used GPU2, output/run-id `graph_allmask64`, probability1.0. Both runs are terminal and must not be restarted. Their full optimizer/RNG/order state is saved; a checked resume entry is still pending, and exact continuation has NOT been validated for this new graph runner.

Heldout feature extraction used the same command with the pre-existing complete-log manifest `/mnt/project/structured-world-v1p1-artifacts/20260927/train_log_holdout64_tokens.json`, output/run-id `image_conditions_log_holdout64`. No holdout labels adapt upstream features. There is no overlap with the64 training logs according to the inherited verified split manifest.

## Independent full-object evaluation

```bash
cd /mnt/project/VLA-Drive-masked-trajectory-world-20260927
CUDA_VISIBLE_DEVICES=2 "$JOINT_PYTHON" tools/joint_world/evaluate_graph.py \
  --checkpoint /mnt/project/joint-world-artifacts/20260927/graph_randommask64/checkpoint_1000.pt \
  --cache /mnt/project/joint-world-artifacts/20260927/image_conditions_log_holdout64 \
  --targets /mnt/project/structured-world-v1p1-artifacts/20260927/targets_v6_full128 \
  --data-root /mnt/project/structured-world-v1-artifacts/20260926/dataset_v1 \
  --output /mnt/project/joint-world-artifacts/20260927/graph_randommask64_image_conditions_log_holdout64_full_eval \
  --ledger /mnt/project/joint-world-artifacts/20260927/budget_ledger.json \
  --run-id graph_randommask64_image_conditions_log_holdout64_full_eval
```

This exports all-masked generated trajectories, sceneCSV, all GT object rows and dynamic/static stationary comparisons. Uses independent2m geometric matching, not the training assignment; unmatched motion is missing and reported as coverage. Graph ego ADE is NOT the deployed DiT trajectory score.

## Remaining work

Interaction-use/context-shuffle diagnostics, uncertainty/multi-seed analysis, robust resume, BEV spatial/interaction objectives and actual integration, matched planning bridge training, original1696 paired planning evaluation, resource/latency results, and final statuses. No navtest tuning or new scorer. New campaign retains the previous combined ceiling of24000 updates/48 GPU-hours; at most2 convergence repair hypotheses.
