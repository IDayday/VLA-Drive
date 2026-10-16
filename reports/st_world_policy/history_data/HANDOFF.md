# History/data/teacher handoff — NEEDS_REVIEW

Data agent branch: `agent/st-history-data-20261016`, original source `1493deda247107efe076b9294863a6753391298a`. Latest implementation at this handoff: `7c20be00a3da3aacf053edae39fc73a704eafc96`. This report is an engineering handoff, not formal experiment approval. No full cache, full model training, navtest, SSH dispatch, or GPU allocation was performed by this agent.

## Changes ready for review

| Commit | Behavior |
| --- | --- |
| `49297f0e` | Thin ST dataset; teacher verification binds the exact supplied requests and complete scene/image coverage. |
| `4b34123c15d5a395c79ab3f999cc7599b5bbdbea` | Selected scenes pin current JSON and original ego labels before registration. Legacy selections require explicit upgrade into a new namespace. |
| `9a1049b9e60f429300fc37f22624d57831dccbbb` | Explicit channel scope permits one unchanged PCA basis across exact registered SP0/1/2 caches; strict exact-contract behavior remains the default. CPU SP1 pooling reuses native FP32 tensors. |
| `43f7144225df65815490ea806f9d76a3ba62f924` | Reader RLock protects receipt/version/LRU transactions. Tensor mmap/digest work remains independent; verified shard versions prevent repeated whole-file hashing. Spawn rebuilds the lock. |
| `ad4404060553c989dea9a4e71c6e807799edbb42` | Future training refuses any teacher timeline other than `[0,0.5,1,2,4]`. Same-shape `[0,0.5,1,3,4]` is rejected. Current-only SP0 remains legal with t0 only. |
| `7c20be00a3da3aacf053edae39fc73a704eafc96` | Registered current LiDAR depth and supported small-vehicle region PCA readouts, with distortion/occlusion behavior tests. |

The concurrency defect was a check/read/move-to-end race: an independent prefetch thread could evict the first thread's LRU entry. The new lock serializes only shared validation metadata. Eight simultaneous first reads perform one shard checksum; LRU eviction and pickle tests preserve tensors and record order. This does not hot-patch a running worker.

## Actual immutable assets

Root for all bounded derived artifacts:
`/mnt/project/st-world-policy-20261016/audits/history_data/`

Original S0 is the completed DiT/flow run, not an IQE Query checkpoint:
`/mnt/project/action-video-foresight-artifacts/20261002/students/formal_S0_seed42_full100k_v2/checkpoints/milestone_100000`.
Run identity: `8dd7d914f10559b0fbfc9593ee1f480a8bb2da80cccb53cba0d684a468e209cd`.

Current data roots:
`/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1` and `student_dev_v1`.
The unchanged source populations contain 101,592 train scenes/1,176 logs and 1,696 dev scenes/16 logs. The bounded selection is 64 train scenes/64 logs and 16 dev scenes/16 logs. Source logs do not overlap between roles; a source recording/session prefix can overlap, so this is not claimed to be a recording-session-independent split.

Generic teacher:
`/mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm`.
Weights SHA256 `45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941` is pinned in the actual cache identity; the runtime cache receipts are authoritative. Compute is FP32 with TF32/autocast off, last-block post-normalization patches, no added token normalization. No trained driving evaluation model is involved.

Raw source logs: `/mnt/navsim/trainval_navsim_logs/trainval/<log>.pkl`.
Sensor lookup: `/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval`, then `/mnt/navsim/trainval_sensor_blobs/trainval`.

### Fixed selections

- `smoke_train64_pinned_v2/SELECTED_SCENES.json`: SHA256 `58387c3779b083cd83b55f2339e49dc99ff12b136e20ea672b9dc0c50e5b40c1`.
- `smoke_dev16_pinned_v2/SELECTED_SCENES.json`: SHA256 `b544911f882b44631b515107f955c96bc43f6933a4c49bfdad1fc5dc4798d148`.
- History remains in `smoke_train64_v1/` and `smoke_dev16_v1/`; old evidence was not overwritten by the pin upgrade.
- History indexes only t0 and requested -1/-0.5 seconds from the same exact source log. Missing history is a false mask, not a duplicated current image. All 384 requested train past-view inputs were available and decoded in the registered 64-scene audit.

### Teacher caches

| Cache directory | Scene scope | Unique images | Tensor bytes | Content |
| --- | --- | ---: | ---: | --- |
| `native384_train64_cost_v1` | train64, t0 + .5/1/2/4s | 957 | 1,693,808,448 | FP32 18×24×1024 |
| `native384_dev16_cost_v1` | dev16, same timeline | 240 | 424,779,712 | FP32 18×24×1024 |
| `pooled384_train64_v1` | same train requests | 957 | 188,209,872 | Native FP32 average-pool3 → 6×8 |
| `pooled384_dev16_v1` | same dev requests | 240 | 47,200,128 | Native FP32 average-pool3 → 6×8 |
| `legacy_c1_train64_current_v1` | train64 t0 only | 192 | 18,885,600 | Exact old C1 FP16 stored 6×8 values |
| `legacy_c1_dev16_current_v1` | dev16 t0 only | 48 | 4,721,568 | Exact old C1 FP16 stored 6×8 values |

Native builds were dispatched by Lead: 53.67s train and 17.46s dev. This agent verified all 957/240 records on CPU, including future images, using the corrected request identity checks. CPU pooling took 10.79s/3.01s with zero teacher forwards. FP16 legacy storage is retained explicitly rather than mislabeled as FP32 source values.

Old C1 is `/mnt/project/ddp-full-foresight-study-artifacts/20260929/dino_targets_v1/C1`, identity `7663c45b77dd711e8304e8d202644dcaa8467f7e265ff8b4c1816d59d2c131cb`. Its images are 192×256, native 12×16 then pool2, not native384. Train64 legacy .5s coverage is incomplete; only t0 is copied for SP0. No missing future labels were fabricated.

## Public training API

```python
from starVLA.dataloader.st_world_dataset import (
    STWorldTrainingDataset, collate_st_world, projected_teacher_contract,
)

dataset = STWorldTrainingDataset(
    data_root, manifest_path, teacher_root,
    expected_teacher_identity, expected_contract,
    history_root=None, pca_path=None,
    require_split="train", need_future=False,
)
model_inputs, targets = dataset[index]
```

`model_inputs.current` preserves the original `image/state/lang/token` whitelist. History is separate and never added to that whitelist. Targets contain original ego IL, current `[V,H,W,C]`, future `[T,V,H,W,C]`, masks and exact future times. Collation returns `list[ModelInputs]` plus stacked targets. Dataset identity separately binds all source content, masks, PCA, role and teacher identity.

Current-only dataset reads no future tensor blocks. For PCA, derive a new `expected_contract` with `projected_teacher_contract`; do not relabel a raw1024 contract as256. The immutable raw cache and PCA outputs remain different identities.

`TeacherCacheReader(root, expected_identity=...)` exposes `read_scene(token, expected_split=..., need_future=...)`. Cache records are mapped independently of training rank. It validates complete coverage, immutable shard version/receipt and accessed record digest. First shard access verifies the full shard checksum; later unchanged accesses reuse that verification. Missing/corrupt records fail explicitly.

## PCA and two real capability probes

Basis root `pca_train64_256_384_v1`, identity `268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46`.
One train-only list of 32,768 patches, channel mean/std, non-whitened basis and all tensor hashes are preserved. Dimensions256 and384 truncate the same basis. No per-token LN or delta normalization is added.

The explicit `CHANNEL_SCOPE.json` identity is `a7d2a7cf55c3ba3cda96cb5756158177e6c8b96d8b923a7e7cdfd56500397947`. It admits only the six exact real caches above and pins their identities. It verifies identical teacher/channel/layer/normalization/software provenance while explicitly retaining resize, pooling and storage precision differences. It does not assert cross-resolution numerical equality.

Reconstruction dev relative SSE: 256 = 0.13670651, 384 = 0.09361588. This alone is not a spatial capability probe.

Two real probes were preregistered in `SPATIAL_PROBE_REGISTRATION.json`, identity `c73ce137b93dae5a8cd2b28729e15ca1232e0047f0dd82ca9f3775f6cc9e083f`, at `2026-10-16T21:23:20.994086+00:00`. Actual execution used 4 CPU threads, 0 GPU, 24.999s, within the 1200s ceiling. Each probe/dimension uses a zero-initialized single linear readout, AdamW, identical 300×512 train-patch exposure, train-only statistics and fixed final-step evaluation.

Labels use current raw LiDAR and current raw vehicle boxes, original camera extrinsics/K/distortion, and full-field pixel-center resize to 384×288. The existing rectified448×256 depth labels were deliberately not resized into the distorted teacher image. DA3 pseudo-depth was audited as available but not used. Future labels do not enter either probe.

| Dev metric | PCA256 | PCA384 | Common valid denominator |
| --- | ---: | ---: | --- |
| Current camera-z MAE (m) | 5.332822 | 5.197710 | 15,922 patches |
| Current camera-z RMSE (m) | 10.336443 | 10.148164 | 15,922 patches |
| Current camera-z AbsRel | 0.415399 | 0.409726 | 15,922 patches |
| Supported small-vehicle region PR-AUC | 0.916009 | 0.914696 | 291 positive/6,889 negative patches |
| Supported small-vehicle region ROC-AUC | 0.993617 | 0.992880 | same |
| Balanced accuracy, fixed logit0 | 0.948489 | 0.952942 | same |

Train denominators: 64,067 depth patches; 1,231 positive/18,232 negative vehicle patches. Supported small boxes: 510 train/127 dev view projections. These are view projections, not independent unique vehicles. Box-region positives require at least three nearest-LiDAR pixels within the projected box depth interval. Occluded/unsupported/ambiguous regions and missing LiDAR are excluded from negatives. This is a bounded readout diagnostic, not semantic segmentation ground truth, planning performance, statistical significance or a final feature-width selection. Lead must select the width once after reviewing this evidence; `dimension_selected` remains null.

Results and source/checksum manifests: `spatial_probes_train64_dev16_v1/{REGISTRATION,SOURCE,LABEL_MANIFEST,PROBE_RESULTS,COMPLETE}.json`. Four tiny probe checkpoints are outside git and do not modify Base or teacher weights.

## Actual regression evidence

`DATA_SIDE_FINAL_REGRESSION.log`: **43 passed**, one existing pynvml deprecation warning, 3.95s.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
 tests/st_world/test_history.py tests/st_world/test_data_schema.py \
 tests/st_world/test_teacher_cache.py tests/st_world/test_pca.py \
 tests/st_world/test_dataset.py
```

`REAL_DATASET_PCA_LOAD.json`: 12 actual CPU dataset load cases, train/dev × SP0/SP1/SP2 × 256/384, 6.46s. Real images, original ego labels, exact teacher tensors and explicit PCA scope were read. SP2 also read real causal history. Future timeline/masks and finite tensors checked. This was not a new Base forward; Lead's independent S0 input32 and two-GPU training smoke provide the model-level evidence separately.

## Commands and next boundary

From this worktree, the completed probe is reproducible/idempotent with:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. \
 /root/miniconda3/envs/ddp/bin/python reports/st_world_policy/history_data/spatial_probes.py \
 --registration /mnt/project/st-world-policy-20261016/audits/history_data/SPATIAL_PROBE_REGISTRATION.json \
 --artifact-root /mnt/project/st-world-policy-20261016/audits/history_data \
 --pca-root /mnt/project/st-world-policy-20261016/audits/history_data/pca_train64_256_384_v1 \
 --raw-log-root /mnt/navsim/trainval_navsim_logs/trainval \
 --sensor-root /mnt/navsim/trainval_all/trainval_sensor_blobs/trainval \
 --fallback-sensor-root /mnt/navsim/trainval_sensor_blobs/trainval \
 --output /mnt/project/st-world-policy-20261016/audits/history_data/spatial_probes_train64_dev16_v1
```

Production tool interfaces remain `python -m tools.st_world_policy.prepare_history --help`, `cache_teacher --help`, and `teacher_pca --help`. Native teacher budgets remain the registered64/16 populations; do not promote them to full-data approval. Full formal caching, history model integration and training are Lead-owned next steps. Data preparation/reader code is reviewable; no formal deployment/training gate is changed by this handoff.
