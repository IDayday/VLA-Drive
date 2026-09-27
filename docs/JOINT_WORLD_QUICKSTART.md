# Joint trajectory world model

Current stage: image-conditioned joint flow and prediction-only planner bridge;64-scene pilots and7284-scene8-pass matched graph trials completed, downstream planning trials active. Real BEV task supervision and online BEV→graph→DiT connectivity have now been tested; matched planner training/comparison remain pending. No result below is a PDMS improvement claim.

## Environment and tests (executed)

```bash
cd /mnt/project/VLA-Drive-masked-trajectory-world-20260927
export PYTHONPATH=.:/mnt/project/DriveVLA-M0/nuplan-devkit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DEPTH_MODEL_CKPTS=/mnt/project/DriveDreamer-Policy/depth_model_ckpts
export JOINT_PYTHON=/root/miniconda3/envs/ddp/bin/python
"$JOINT_PYTHON" -m pytest -q tests/joint_world tests/structured_world_v1p1
```

27 tests passed. Covers whole-actor masks, hidden context poisoning including NaN, non-ego permutation equivariance, empty surrounding labels, gradients, graph save/load and previous coordinate/append-tail regressions.

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

Control used GPU2, output/run-id `graph_allmask64`, probability1.0. Both runs are terminal and must not be restarted. Their full optimizer/RNG/order state is saved; the later checked resume entry passed a realGPU6 vs3+3 exact-state comparison (source779dfe1). Do not use a different code revision to resume pinned historical runs.

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

Extended corpus learning curves, interaction-use/context-shuffle diagnostics, uncertainty/multi-seed analysis, matched planning bridge and BEV training, original1696 paired planning evaluation, final resource/latency results and statuses. No navtest tuning or new scorer. New campaign retains the previous combined ceiling of24000 updates/48 GPU-hours; at most2 convergence repair hypotheses.

## Longer matched phase and convergence (executed, complete)

Use the immutable source worktree `/mnt/project/VLA-Drive-joint-runs-fed64aa`. A realGPU4-step continuous run exactly matches2+2 resumed updates, including scheduler/RNG/order; see CORPUS_RESUME_CHECK.json. This validates the runner, not learning. CPU resident cache is bounded at256 scenes by default; upstream frozen current features are loaded on demand. Original fullGT is retained, including233 raw-log rebuilt over-capacity scenes.

```bash
cd /mnt/project/VLA-Drive-joint-runs-fed64aa
"$JOINT_PYTHON" tools/joint_world/prepare_corpus.py \
  --manifest /mnt/project/joint-world-artifacts/20260927/extended_training/train_tokens.json \
  --shards /mnt/project/joint-world-artifacts/20260927/extended_training \
  --original-targets /mnt/project/structured-world-v1-artifacts/20260926/targets_v6_train8192 \
  --rebuilt-overflow /mnt/project/joint-world-artifacts/20260927/extended_training/full_overflow_targets \
  --output /mnt/project/joint-world-artifacts/20260927/extended_training/corpus_v1
CUDA_VISIBLE_DEVICES=0 "$JOINT_PYTHON" tools/joint_world/train_corpus.py \
  --cache /mnt/project/joint-world-artifacts/20260927/extended_training/corpus_v1/conditions \
  --targets /mnt/project/joint-world-artifacts/20260927/extended_training/corpus_v1/world_targets \
  --data-root /mnt/project/structured-world-v1-artifacts/20260926/dataset_v1 \
  --config configs/joint_world/image_graph_residual.json \
  --output /mnt/project/joint-world-artifacts/20260927/extended_randommask7284 \
  --ledger /mnt/project/joint-world-artifacts/20260927/budget_ledger.json \
  --run-id extended_randommask7284 --epochs 8 --batch 16 --all-hidden-probability 0.5
```

Control uses another verified free GPU, output/run-id `extended_allmask7284` and probability1.0, otherwise identical. Both start fresh from seed42, not from their small pilots. Each3642 updates covers exactly8 passes/58272 presentations. Fixed milestones0/1/2/4/6/8 passes, with warmup/cosine schedule. Retain all checkpoint metrics rather than selecting a favorable endpoint. Neither small-set1000 steps nor reaching this cap establishes convergence; unstable training or holdout trends remain INCONCLUSIVE.

For an explicitly paused, fully accounted run only, repeat its exact original arguments plus `--resume <same-output>/checkpoint_<step>.pt`. The script refuses identity changes, live/unreconciled runs or already completed phases. Abrupt process death requires budget/step reconciliation before resume; no cross-code/device exactness claim.

Real BEV task probe and online-path results are archived in reports/joint_world/bev_tasks_probe64/ and BEV_PIPELINE.json. Exact executed arguments, source hashes and immutable artifact locations are in their manifests and RUN_LEDGER.json. The online test verifies actual current-image provider cost and label-poison invariance; the manually opened-gate action difference is tiny and is not evidence of better planning.

## Completed longer result and actual planner transfer

The graph pair completed exactly3642 updates each,8 passes over7284 scenes. Every prescribed checkpoint was evaluated on the same64/59-log holdout. `reports/joint_world/extended_learning/` contains all scene/object CSV, training loss JSONL, curves and log-cluster confidence intervals. Final ego graph ADE1.206/1.055m (masked/control), matched-agent6.556/6.704m, stationary2.437m, coverage11.46%. Do not mistake graph ego predictions for the deployedDiT output. With one training seed and unstable rankings, results remain INCONCLUSIVE.

Source7bd8901 adds deterministic downstream bridge training; Qwen/currentheads/graph/originalDiT stay fixed. Only graph_to_world and action attention train from the original ego FM loss. This is a representation transfer probe. A realGPU4 vs2+2 interrupted comparison passes bitwise model/optimizer/scheduler/RNG/order. An earlier lazy-import RNG failure remains documented in PLANNER_RESUME_INITIAL_FAILURE.json. The successful one-scene cached-planner test also verifies exact native action labels, gate0 fidelity, target-poison invariance and strict independent restore.

Actual launched image-graph transfer pair: `/mnt/project/VLA-Drive-joint-runs-planner-deterministic`,4passes,batch16,1821updates per variant. Example executed command (do not relaunch an existing run):

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NO_ALBUMENTATIONS_UPDATE=1 \
PYTHONPATH=.:/mnt/project/DriveVLA-M0/nuplan-devkit \
DEPTH_MODEL_CKPTS=/mnt/project/DriveDreamer-Policy/depth_model_ckpts \
/root/miniconda3/envs/ddp/bin/python tools/joint_world/train_planner_bridge.py \
  --cache /mnt/project/joint-world-artifacts/20260927/extended_training/corpus_v1/conditions \
  --targets /mnt/project/joint-world-artifacts/20260927/extended_training/corpus_v1/world_targets \
  --data-root /mnt/project/structured-world-v1-artifacts/20260926/dataset_v1 \
  --base-checkpoint /mnt/project/DriveDreamer-Policy/models/DriveDreamer-Policy \
  --graph-checkpoint /mnt/project/joint-world-artifacts/20260927/extended_randommask7284/checkpoint_3642.pt \
  --output /mnt/project/joint-world-artifacts/20260927/planner_randommask7284 \
  --ledger /mnt/project/joint-world-artifacts/20260927/budget_ledger.json \
  --run-id planner_randommask7284 --epochs 4 --batch 16
```

Control uses GPU0, replaces randommask with allmask in graph checkpoint/output/run-id. Source/cache/sampling/optimizer budget are otherwise identical. Current code `evaluate_planner.py` exports the originalDiT trajectory from whitelisted current-only cache fields, never opening targets; it is COMPILED, NOT YET GPU-EXECUTED. Original1696 development conditions are ready at `development_conditions/merged1696`. The final separate PDMS comparison is NOT_RUN until those complete outputs are scored.

Same7284-scene current BEV extraction is running on GPUs2/3. An initial shard1 CLI SHA transcription was rejected before extraction; ledger preserves its failed cost, corrected shard1_retry is a separate run. The extraction completion marker is extraction.json. Larger BEV task-on/task-off planner training remains NOT_RUN; do not describe the600-step task-only probe or tiny manually gated online effect as a planning gain.
