# Frozen teacher contracts and training-only readouts

Scope: full Qwen-Flow world/post-action experiment, source starting at
`425eb5420ce1b592786f455ceea9bd6d124d2c54`. This report records implemented and
tested contracts separately from target-generation coverage and planning gain.

Final target generation is **COMPLETE** for the fixed TRAIN90,890 and DEV1,696
populations. No model-quality or planning-gain claim follows from cache
completion. The immutable source snapshot is
`teacher_resolved_v3_final_20261025T160330Z` (SHA256
`67d8738cf13124e6cf1f57804b30de7e99fd036093d9d81b1085962336baa0c3`).

| Population | Unique / physical records | Shards | Payload bytes | All3 future views valid | True timing gaps, future masked |
|---|---:|---:|---:|---:|---:|
|TRAIN|90,890 /90,890|450|303,972,516,000|90,052|838|
|DEV|1,696 /1,696|14|5,672,102,400|1,691|5|

The final matching-time audit found **zero duplicate records and zero duplicate
payload bytes**. Every scene has current DINO targets; genuine missing/gapped
future observations are explicitly masked, with original timestamps retained.
Each record occupies3,344,400 bytes including FP32 DINO, BF16 WA targets, masks
and timestamps. Indices/manifests and teacher checkpoints are separate storage.

Final manifests are under
`/mnt/project/qwen-world-postcot-20261025/teachers/{TRAIN,DEV}_resolved_v3/`.
TRAIN manifest file SHA256 is
`fc2d94f2ef70b4b195f1ee0a94377047c593fc651d94fab3b4e238d6586b367e`;
DEV is `3ec622acfaf490510367c2693e93a3f2734cd04ed6ec0dcb091a8392c41a93c5`.
TRAIN cache identity is
`b83c252eda6b3a4cdd3fb8ed8c249639f070955276ec445c8ade6fb0bde63b5b`;
DEV is `b067384ea294f65920120920e61f9752cae20b1d6dd43a774e68b9f82c8f1bd5`.
The separate final signoff is `runtime/FINAL_TEACHER_CACHE_SIGNOFF.json`.
Whole-shard hashes were computed while building all shards, and finalization
rechecked64 accessed records per population. DEV additionally rehashed every
whole shard. TRAIN's immutable `whole_shards_rehashed=False` accurately means
there was no additional304GB whole-file re-read; it is not rewritten to claim
that additional verification. Reader record checks remain enabled.

The last1,351 missing scenes were partitioned into14 disjoint nonempty ranges
after all previous writers stopped. Fourteen actual GPUs processed those
ranges; the partition-count768 argument is not a claim of768 GPUs. All14
workers exited0, followed by exact-coverage finalization. This completed the
same locked population and numeric contract without introducing a new dataset.

## Actual assets, not random replacements

| Teacher | Actual release | SHA256 | Strict CPU loading |
|---|---|---|---|
| DINOv3 ViT-L/16 | `timm/vit_large_patch16_dinov3.lvd1689m`, revision `30c1109559f65dea34316b0d4842d35c5771fe11` | `45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941` | 303,079,424 frozen parameters; strict state loading succeeded |
| WA-JEPA target encoder/projector | `AFARI-Research/WA-JEPA`, revision `15c0770ebd233665214590bb2190a90907499f9e` | `e1d3be013de7dbaa5a6c30925a0d2f91e6ce1a6b17a75ca73fe01a1abe6d6408` | 302/302 encoder tensors and 4/4 projector tensors; no missing/unexpected keys; 306,786,816 frozen parameters |

DINO's local file is
`/mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm/model.safetensors`
(1,212,347,640 bytes). WA's public checkpoint was actually downloaded and
verified at
`/mnt/project/qwen-world-postcot-20261025/assets/wa-jepa-model_state_dict.pt`
(1,575,763,741 bytes). An interrupted HTTP body was resumed with an explicit
validated HTTP206 Range; the complete SHA was checked before publication.
The incomplete file was never admitted as a model.

The WA checkpoint contains 1,162 tensors. Only `target_encoder.*` and
`target_scene_projector.*` are instantiated and loaded. The learned projector
has LayerNorm weight/bias `[4096]`, Linear weight `[512,4096]` and bias `[512]`.
Neither the predictor nor its fourth-camera embeddings is loaded. Module
constructors require the exact published weight hash and strict loading; there
is no random-projector or alternative-model fallback.

Official sources are [WA-JEPA at the locked source commit](https://github.com/AFARI-Research/WA-JEPA/tree/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad),
[the locked public weight revision](https://huggingface.co/AFARI-Research/WA-JEPA/tree/15c0770ebd233665214590bb2190a90907499f9e),
[DINOv3](https://github.com/facebookresearch/dinov3), and
[the exact timm conversion](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m/tree/30c1109559f65dea34316b0d4842d35c5771fe11).
External source/weights remain outside Git; their original licenses remain with
the downloaded source. The wrapper validates hashes of the pinned WA encoder,
normalization/projector code, configuration and license before importing them.

## DINO current target

- Cameras: `cam_l0`, `cam_f0`, `cam_r0`; one actual current frame.
- Full RGB field, PIL bicubic resize to H288×W512, ImageNet normalization once.
  No implicit center crop, padding or random augmentation.
- ViT-L/16 dense final post-norm features. Remove the **actual five prefix
  tokens**. Do not add token L2 normalization.
- Native grid18×32×1024; fixed2×2 mean pooling yields9×16×1024 per view.
- Frozen FP32 parameters and FP32 compute in the main target builder; output
  and cache are FP32. This is teacher execution, not a change to student AMP.
- Actual prior current-DINO caches use6×8/view and are incompatible. They are
  not reused as if they were9×16 targets.

The student's `DINOReadout` reads only current `memory.spatial`. Its queries
carry explicit camera/y/x identities; output is `[B,3,9,16,1024]`. A valid patch
MSE is reduced in FP32, first within scene then over scenes with supervision.

## WA-JEPA future target

- Actual chronological4 history +8 future frames, 2Hz; frame index3 is current.
  Store integer microsecond timestamps, not inferred action-point matches.
  An unavailable real timestamp is explicitly-1. Incomplete views or clips
  whose actual adjacent timestamps depart from2Hz by more than50ms receive
  no future supervision; current DINO/perception remain enabled. Missing
  zero-filled inputs are never encoded as valid teacher features. Counts of
  missing timestamps/views and timing validity are retained per record.
- Official preprocessing: OpenCV BGR decode→RGB, INTER_AREA resize H256×W512,
  dataset normalization to[-1,1], cast to the released BF16 teacher's dtype,
  then official ImageNet conversion in that dtype.
- Run frozen `target_encoder(..., training=True)` while the **module remains
  eval**. Here `training=True` selects the official hierarchical-return API;
  it does not enable teacher dropout or trainable buffers.
- Four1024-wide hierarchy outputs, official per-chunk LayerNorm, then select
  four future tubelets after the two history tubelets. Apply the strictly
  loaded4096→512 target_scene_projector **before** spatial pooling.
- Native spatial16×32; pooled8×16. Output `[B,3,4,8,16,512]`, BF16 losslessly
  stored. Each future tubelet binds the two actual source-frame timestamps.
- Camera independence was verified in source: the official encoder reshapes
  `[B,V]` into independent batch entries; the shared projector is tokenwise.
  The three actual student views can therefore be extracted without invoking
  the released four-camera predictor.
- Full-clip history features may contain future context. **No history slice is
  returned as an actor input.** All teacher tensors are detached targets only.

The student's `FutureReadout` is four512-wide Transformer readout blocks.
Camera/time/grid queries are modulated by the corresponding group of eight
current-derived predictive tokens. Keys/values come from current student
spatial/global/predictive memory.128 queries are stratified over all12
camera/tubelet buckets per scene; no replacement occurs inside a bucket.
The sampler has a separately saveable CPU generator. There is no feature
diffusion sampler, second VLM, or teacher input on the actor path.

## Current perception

`CurrentObjectReadout` uses32 queries and current spatial/global memory. It
predicts existence, one of seven current-object classes, and physical ego
`[x,y,z,l,w,h,sin(yaw),cos(yaw)]`. Hungarian matching uses fixed-scale geometry
and class costs. Classification/box/existence losses have distinct valid-scene
denominators. Datasets must supply current calibrated visibility and explicit
`coverage_valid`; unmatched predictions are not background negatives when
camera coverage/occlusion is unknown. No future actor labels are read.

Depth is explicitly camera optical z in metres after the registered
`camera_from_ego` transform; it is not Euclidean distance. Calibrated current
box center visibility helpers and depth-coordinate tests are included. The
parent data builder supplies actual camera calibration and current ego boxes;
3D regression already supervises their transformed depth, without an additional
unrelated depth head.

## Bounded target storage and data identity

`teacher_cache build` is a separate offline program. It has no optimizer,
backward, student, action loss or PDM scorer. It reads only registered
`teacher_frames`, not GT trajectory, objects or scores. Fixed legal population
manifest/record hashes, input observation identity, actual image hashes,
teacher weights/source/precision, preprocessing and camera/timestamp schema
are recorded. Changing trainable Qwen weights or an optimizer step does not
invalidate a frozen external teacher function.

Raw immutable binary shards default to256 records (856,166,400 bytes), with
per-record and whole-shard SHA256 and atomic sealed completion receipts.
Training reads only the requested record via `pread`; an eight-handle LRU
limits open shards and no multi-GB deserialization occurs per scene. Every
access checks the file version and full record checksum. Optional verification
rehashes whole shards once; it is not repeated each training update. Decoder
image preprocessing uses a bounded64-image LRU. Building rank does not restrict
consumer access or change the training sampler order. Missing/uncommitted or
corrupted records fail explicitly. A global COMPLETE manifest requires exact
coverage of the locked population, not merely the existence of shard files.

At the declared precisions the actual layout is **3,344,400 bytes/scene**:
1,769,472 DINO, 1,572,864 WA, 1,968 validity bytes, 96 timestamp bytes.
For the physically isolated90,890-scene TRAIN this is303,972,516,000 bytes
before index/receipt overhead; DEV1,696 is5,672,102,400 bytes. This is a
capacity calculation, **not completed-cache coverage**. No multi-TB visual
student cache is requested or produced by this target pipeline.

The original101,592 TRAIN and1,696 DEV share15 physical logs despite segment
separation. The main data builder embargoes10,702 TRAIN scenes from those
logs, retaining90,890 TRAIN across147 physical logs. The public WA driving
release was trained on NAVSIM navtrain with a hash-log5% holdout. Its public
pretraining may include our DEV logs; do not call DEV wholly teacher-unseen.

## Visualization decoder

`DINORGBDecoder` is deterministic: channel projection, five2× upsampling
stages with residual convolution, RGB sigmoid. It consumes9×16×1024 features
and outputs288×512 RGB, with no RGB skip connection, VAE latent or KL. It has
its own teacher-feature→RGB L1 training CLI, RNG/sampler/optimizer checkpoints,
fixed training-token identity and budget. The same frozen decoder renders
teacher and student features; optional PCA fits one teacher basis and uses it
unchanged for both. It is absent from planning loss and is not a prerequisite
for planning training.

## Verification status

`tests/world_postcot/test_teacher_readouts.py`: **15 CPU tests passed** in the
existing flow-safe-improvement virtual environment. They cover readout
gradients/label detach, masked NaNs, empty supervision, exact full-update vs
microbatch loss gradients, calibrated depth/visibility, timestamp binding,
stratified target joins, lossless FP32/BF16 shard round trips, corruption/role/
observation identity refusal, bounded FD lifetime, teacher eval mode, and
deterministic RGB reconstruction dimensions/gradients.

Actual strict teacher CPU loading: PASS. Real64-scene GPU pilot: PASS, on eight
A80080GB devices at `training-vlawm-zt`, under the finite authorized host lease.
The sealed pilot population uses seeded hash(`[42,token]`) first64, fixed before
scores, and its first32 scenes match the candidate-label pilot. All64 have
current-three-view DINO targets;62 have future supervision and2 correctly mask
future due to actual clip gaps. The pilot has214,041,600 raw target bytes in17
immutable shards. Every shard and every record was rechecked. Pilot-only
COMPLETE is `/mnt/project/qwen-world-postcot-20261025/teachers/pilot64_v3/MANIFEST.json`.
It is not the full TRAIN manifest.

Per-worker extraction windows were15.24–18.06 seconds for7–8 newly generated
scenes (0.443–0.515scene/s), with2,182,347,776 peak allocated CUDA bytes.
Including model loading and worker polling, observed invocation wall was35–40s.
These scattered64 scenes are a cost pilot, **not a full-production throughput
claim**. Full production uses contiguous log/time producer chunks with bounded
image reuse; the original training sampler is unaffected. The original-sampler
first1024 targets are prioritized and committed in16-record shards so full
training qualification can begin before bulk caching ends. Subsequent bulk
shards return to the256-record bound.

Real full-tensor replay audit on two actual scenes: original per-scene3-view
production shape reproduces **all**442,368 DINO elements and786,432 WA elements
bitwise; repeated calls are also bitwise and leave CPU/CUDA RNG unchanged.
Regrouping, shuffling and repeating scenes into batch2 produces DINO max
absolute difference4.3847e-6, RMSE3.6147e-7, relative L2 1.3576e-6; WA remains
bitwise equal. The batch-shape difference is reported separately rather than
called bitwise equivalent. The real-image audit peak allocation is2,549,889,536
bytes. Full evidence is `pilot64_v3/{PILOT_VERIFICATION,REAL_IMAGE_AUDIT}.json`.

Two failed attempts are preserved: an unmatched parenthesis was detected before
teacher allocation; the next real forward exposed the official projector's
required4D `[B,V,N,D]` interface. The corrected wrapper retains an explicit
single-view dimension for each independently batched real view and invokes the
original strictly loaded projector. No failed attempt admitted a target.

The production image audit found complementary storage views. The project
mirror omits current images for136 otherwise valid TRAIN scenes, while the
apparently canonical `/mnt/navsim/trainval_sensor_blobs/trainval` contains
many current observations but is not a complete future-frame store. Treating
it as complete incorrectly masked future supervision in about two thirds of
an early16,846-record prefix. Those old partial targets are **diagnostic only**,
preserved and excluded from formal full-target admission. This was a storage
resolution error, not evidence that those scenes lack a real future clip.

The final `resolved_v3` loader keeps current offset0 **exactly `row.images`, as
used by Qwen**. Other frame identities come from the real raw log's token,
offset, camera and actual timestamp. It first resolves against
`/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval`, with an explicit
complementary fallback to `/mnt/navsim/trainval_sensor_blobs/trainval`.
There is no scene guessing or synthetic frame. Each chosen image records its
full SHA; the first16 both-available mirror pairs per producer additionally
verify equal SHA and refuse disagreement. A diagnostic pair chosen to expose
the two opposite mount omissions now has all three WA views in both scenes,
and all sampled mirror hashes agree. Both single-root diagnostic variants
remain archived with their original receipts and tensors.

Only current DINO images pass through PIL. Future frames use the official
WA OpenCV decode/INTER_AREA pipeline directly. Applying unused PIL processing
to every future frame had also imposed a nonofficial JPEG rejection path;
it has been removed, without changing the WA tensor preprocessing formula.
Missing files across both real mounts and true timestamp gaps remain explicit
invalid targets. No missing or erroneous current observation is replaced by
an invented image. The two logs and64 decoded images are bounded LRUs.

Final90,890-scene TRAIN production began separately at
`/mnt/project/qwen-world-postcot-20261025/teachers/TRAIN_resolved_v3`, initially
with13 leased workers on the8-card temporary host and5 permanent GPUs. DEV1,696 uses
`teachers/DEV_resolved_v3` on a separate permanent GPU. The immutable source is
`snapshots/teacher_resolved_v3_final_20261025T160330Z`; the initial final TRAIN
teacher identity is
`b83c252eda6b3a4cdd3fb8ed8c249639f070955276ec445c8ade6fb0bde63b5b`.
Each scene polls the host-expiry stop contract and commits recoverable targets.
Both full caches were marked COMPLETE only after exact coverage and verification
succeeded, as recorded in the final signoff above. This teacher evidence does not establish representation quality,
planning gain, or completed IL/RL.

After profiling freed another8-card temporary host,22,498 records were already
committed and68,392 remained. All13 writers first exited at native boundaries;
the same cache/function then resumed with **21 GPU workers, one process/GPU**,
skipping globally committed tokens. No source, precision, population or training
sample order changed. That phase's mapping is
`runtime/teacher_TRAIN_resolved_v3/CURRENT_WORKERS.json`; it includes a GPU6
relocation after a real legacy77k-RL DEV job acquired GPU5. The legacy job was
preserved. Temporary producers retain the task's9-hour host guard.

The exact original-sampler first1,024 scenes are ready, with1,017 complete
three-view WA clips and7 true timestamp gaps (99.3164% valid future targets).
DEV is now **COMPLETE**:1,696 records,1,691 full-WA clips and5 true gaps;
whole DEV shards were rehashed and64 accessed records rechecked. DEV identity
is `b067384ea294f65920120920e61f9752cae20b1d6dd43a774e68b9f82c8f1bd5`.
A CPU watcher on a permanent host published TRAIN COMPLETE only after exact
coverage and manifest/record checks, using the final worker mapping.

The matching-time duplicate audit captured85,453 unique records and exactly
85,453 receipted physical records in418 shards. Physical data bytes were
285,789,013,200, with **zero duplicated records, zero duplicate bytes and zero
unreceipted bins**. The apparent comparison with an earlier62,848 count was
between different observation times. `_cache_index` and writer resume skip
globally committed tokens across all ranks. Audit timestamps and full results
are in `runtime/teacher_TRAIN_resolved_v3_world21/DUPLICATE_AUDIT.json`.
Neither bytes alone nor this partial audit implies full TRAIN completion.

The RGB decoder completed a real bounded run with a fixed640-scene TRAIN
subset from valid committed DINO targets, batch16 and a2,000-update upper bound.
Its90-minute wall budget ended at **1,473 updates**,23,568 scene presentations,
with last training L1=0.06207012 and a complete step1,473 checkpoint/receipt.
Its first invocation was `STOPPED` at the native optimizer boundary, not
prematurely marked2,000-step COMPLETE.
After full target-cache completion, the original run was explicitly authorized
to resume from step1,473 to the original2,000-update upper bound, with at most
45additional minutes on independently verified spare `training-vlawm-zt`GPU7.
It preserves the fixed640training tokens, old valid DINO cache identity,
optimizer, RNG and sampler state. This is not a new decoder or new training
population. That invocation completed with `EXITED`, returncode0, in
900.287additional training seconds, under the45-minute bound. The decoder is
now **COMPLETE at2,000 updates**,32,000 scene presentations (50passes over the
fixed640scenes), with last training L1=0.05740567. The final sealed checkpoint is
`rgb_decoder/train_main_seed42/step_002000.pt`, SHA256
`d8ce226adde1f0f72b486ea6dc68c143644fcfefe62db1ba8a2313974ac7cc52`.
The original initialization/data identity remains
`fceae2c3b102f3d6d18d0e5338dcf0040577b6c14f03ec25a7899d7930243746`.
The same preselected16DEV scenes, CPUFP32 decoder and full-image L1 protocol
at the fixed final endpoint give **0.07258202**. This diagnostic did not select
the checkpoint. The result and private teacher-reconstruction figure are in
`rgb_decoder/train_main_seed42/eval_step2000_DEV16/`; no raw scene image enters
Git. The worker has exited and released its GPU. The step1,473 checkpoint
remains preserved for the separately preselected first-student visualization.
All1,920 current-image SHA values in that subset were independently matched
against canonical current images (zero mismatch), so its valid DINO-only
training remains legitimate across the future-cache routing repair. It does
not read WA targets. Early training L1 fell from0.157 at step10 to roughly0.08
around step160; this is **training fit**, not planning gain. At the preselected
step1,000 checkpoint,16 fixed DEV scenes (hash(`[42,token]`) first16, chosen
before reconstruction errors) have teacher-feature RGB mean absolute error
**0.07564984** in[0,1], using the same frozen decoder in CPU FP32. This is a
held-out reconstruction diagnostic, not a planning metric. The original/
teacher-reconstruction/error figure remains private under the artifact root;
no raw scene image is committed. Student reconstruction waits for an actual
student checkpoint and is not fabricated. The decoder has its own recoverable
optimizer/RNG/sampler checkpoints
and yields its GPU at a native boundary when the formal IL needs it.

A finite single-GPU producer1p/2p benchmark on the first64 chronological scenes
initially measured1.38× extraction throughput and exact payload agreement, but
only8/64 had valid WA under the incomplete single-root loader. It is retained
as a diagnostic and **cannot justify production scaling**. The two same
configurations were then measured on corrected complete clips. All64 scenes
had all three views and four future tubelets; every record payload is bitwise
equal across1p/2p.

| Full-WA configuration | Extraction wall for64 | Scenes/s | Wall including initialization | Sampled GPU reserved peak | Sampled CPU RSS peak |
|---|---:|---:|---:|---:|---:|
|1 process/GPU|45.510s|1.406|68.012s|3,329MiB|2,906,192KiB|
|2 processes/GPU|47.160s|1.357|87.562s|6,156MiB|5,899,028KiB|

Speedup is0.965×: the second process did not improve this matched full-target
workload and is **not adopted**. GPU and CPU resource peaks also increase.
The benchmark warmed its own images, did not flush shared OS caches, and does
not claim physical cold-disk performance. No root cause is assigned from GPU
utilization alone. The pipeline retains one producer perGPU;13→21 only uses
additional idle devices for the same fixed data.

Actual separate parameter counts: DINO readout7,374,336; future readout
18,153,472; current-object readout2,385,168; visualization RGB decoder2,487,395.
The first three are training-only student readouts; the RGB decoder is neither
part of the planner optimizer nor deployment. Frozen teachers total609,866,240
parameters and are absent from student deployment.

Commands (run GPU builds through the resource-lease worker):

```bash
python -m tools.qwen_world_postcot.teacher_assets \
  --dino-root /mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm \
  --wa-source /mnt/project/qwen-world-postcot-20261025/assets/WA-JEPA-source \
  --wa-checkpoint /mnt/project/qwen-world-postcot-20261025/assets/wa-jepa-model_state_dict.pt \
  --output /mnt/project/qwen-world-postcot-20261025/assets/ASSET_AUDIT.json

python -m tools.qwen_world_postcot.teacher_cache build \
  --population /mnt/project/qwen-world-postcot-20261025/data/TRAIN/manifest.json \
  --root /mnt/project/qwen-world-postcot-20261025/teachers/TRAIN_resolved_v3 \
  --dino-root /mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm \
  --wa-source /mnt/project/qwen-world-postcot-20261025/assets/WA-JEPA-source \
  --wa-checkpoint /mnt/project/qwen-world-postcot-20261025/assets/wa-jepa-model_state_dict.pt \
  --device cuda:0 --rank 0 --world-size 21 --max-scenes 4500 \
  --max-seconds 22000 --shard-records 256 \
  --priority-scenes /mnt/project/qwen-world-postcot-20261025/data/TRAIN_PROFILE_PRIORITY1024.json

python -m tools.qwen_world_postcot.teacher_cache verify \
  --population /mnt/project/qwen-world-postcot-20261025/data/TRAIN/manifest.json \
  --root /mnt/project/qwen-world-postcot-20261025/teachers/TRAIN_resolved_v3 \
  --sample-records 64 --full-shards --finalize
```

Each builder rank is an independently registered process under its GPU lease.
Above rank0 is one worker, not an instruction to run all workers on one device.
Budgets are per worker. Resume the production cache with the **same immutable
source snapshot and identity**; a different producer topology is allowed only
after old writers quiesce, with globally committed tokens skipped. The64-scene
pilot used8 scenes on each of eight workers and its own population/root.
