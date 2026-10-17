# Existing-runtime T3/T4 integration

Commands are staged, not dispatched here. Lead is the sole dispatcher. Preserve
active jobs and all old failed receipts. Do not run a bare builder/trainer outside
the existing admitted supervisor and storage/lease boundaries. No new gate or
monitor framework is introduced.

The new CLI `--help` was executed. Existing builder/teacher CLI flags were
checked in their actual parsers. Choose **one** cell for the first real bounded
observation. A second cell is a separate invocation/registration, not automatic.

```bash
RT=/mnt/project/VLA-Drive-st-temporal-combination-20261017
PY=/root/miniconda3/envs/ddp/bin/python
ROOT=/mnt/project/st-world-policy-20261016
SELECTED=$ROOT/audits/indexed_composition_preparation_v1/SELECTED_SCENES.json
HISTORY=$ROOT/audits/history_data/temporal_composition32_history_v1
CELL=T3
PREP=$ROOT/audits/temporal_${CELL}_indexed32_preparation_v1
TRAIN=$ROOT/profile/temporal_${CELL}_indexed32_updates4_v1
cd "$RT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
```

1. **Matching history metadata (pending).** Reuse `prepare_causal_history build`
   through the existing metadata_storage callback/controller. The following is
   the builder argv to register, **not an alternative unmonitored launch**.
   Its matching ≤32 scene resource/storage registration must first exist; use
   a new immutable namespace. It writes causal metadata/current requests only,
   no DINO/teacher tensor. Source paths are from the registered history handoff.

```bash
: "${HISTORY32_BUILDER_REGISTRATION:?Lead must register exact32 causal metadata budget}"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.st_world_policy.prepare_causal_history build \
  --student-root /mnt/project/ddp-foresight-artifacts/20260928/student_train_v1 \
  --raw-log-root /mnt/navsim/trainval_navsim_logs/trainval \
  --sensor-root /mnt/navsim/trainval_all/trainval_sensor_blobs/trainval \
  --fallback-sensor-root /mnt/navsim/trainval_sensor_blobs/trainval \
  --partition /mnt/project/VLA-Drive-st-history-population-agent-20261016/reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json \
  --scene-list "$SELECTED" --split train --mode registered --max-scenes 32 \
  --seed 20261017 --tolerance 0.05 --target-times 0 --record-layout packed_log_v1 \
  --registration "$HISTORY32_BUILDER_REGISTRATION" --output "$HISTORY"
```

The already present `smoke_train64_v1` cannot substitute: it has no scene overlap.
No original future scene list is changed. Actual image/hash cost and storage
claim use the existing builder's accounting; this report does not invent a
real-data cost or authorize a build.

2. **Source binding and final configuration (pending).** After review and final
   integration, use the final frozen source tree as `RT`. The first command
   explicitly rebinds changed control bytes while retaining all protected math.
   The second uses the existing indexed/PCA validator and the new narrow input
   binding; it emits immutable metadata, no model/optimizer or new teacher tensor.
   Run once under the existing bounded metadata supervisor, accounting for its
   actual index validation reads; do not call it a small-JSON-only operation.

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.st_world_policy.qualify_indexed_provider prepare-profile-migration \
  --parent-descriptor "$ROOT/smoke/world_indexed_provider_native2_v1/PROVIDER.json" \
  --parent-runtime /mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017 \
  --output "$PREP/SOURCE_MIGRATION.json"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.st_world_policy.temporal_composition \
  --cell "$CELL" --parent-descriptor "$ROOT/smoke/world_indexed_provider_native2_v1/PROVIDER.json" \
  --reference-config "$ROOT/smoke/world_indexed_provider_training2_v1/CONFIG.json" \
  --data-manifest "$SELECTED" --history-complete "$HISTORY/COMPLETE.json" \
  --source-migration "$PREP/SOURCE_MIGRATION.json" \
  --pca-fit-cache-root "$ROOT/audits/history_data/native384_train64_cost_v1" \
  --pca-fit-cache-index "$ROOT/caches/original_pca_fit64_metadata_index_v1" \
  --output "$PREP/configured" --training-output-root "$TRAIN/runs" --max-seconds 1800
```

3. **One real bounded native invocation (not authorized here).** Lead registers
   final source SHA, `$PREP/configured/training_config.json`, unchanged selected
   manifest and target contract, new `PROVIDER.json`, original BASE_BINDING,
   actual environment and a fresh one-GPU/global2 resource lease/checkpoint
   reservation. `weights_usable_for_formal_init=false`. No automatic approval
   file is fabricated. The actual native command is:

```bash
: "${T34_REGISTRATION:?New exact PROFILE registration required}"
: "${T34_CPUSET:?Actual admitted CPU affinity required}"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.st_world_policy.check_formal_approval \
  --registration "$T34_REGISTRATION" --repo "$RT"
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
taskset -c "$T34_CPUSET" "$PY" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=1 \
  -m tools.st_world_policy.train_world --registration "$T34_REGISTRATION"
```

The existing owned training supervisor supplies whole-process resource caps,
duplicate-run locks and safe boundary cleanup. The four updates are cumulative,
not four more per invocation. Native resume is unchanged; only after an actual
interior checkpoint and separately admitted continuation, append
`--resume --acknowledge-stop` to the same trainer command. No resume experiment
is required or started here.

Inspect actual history/W/readout gradients, finite current/future/delta loss and
global denominators, 4 updates / original global2 exposure, complete checkpoint,
and original label-only inference boundary. Do not use this PROFILE endpoint as
formal initialization or planning-gain evidence. Scientific coefficient choice,
spatial-parent selection and full-history T matrix remain separate research work.
