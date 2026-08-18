# VGGT V3: reconstructable 195-token knowledge and DLC training

## Why V3 is different

V1/V2 established that the student representation is scene-sensitive and can
closely match the VGGT teacher, but the V2 trajectory path did not reliably
turn that knowledge into a better trajectory. V3 therefore separates three
questions which cosine alignment alone cannot answer:

1. **Codec sufficiency:** can a 195-token teacher representation still drive
   VGGT's original post-layer-11 camera task?
2. **Student knowledge:** can the student representation replace the teacher
   representation in the same frozen VGGT downstream path, without fitting a
   new probe?
3. **Planner utilization:** if the first two pass, does the trajectory model
   improve and causally respond to the correct scene memory?

The feature source is fixed: **only VGGT layer 11 global**, namely
`aggregator.global_blocks[11]_before_dpt` with native shape
`[B,3,5+37*37,1024]`. The encoder receives no layer-4 tensor and no frame
branch. It emits exactly `15 special + 3*6*10 spatial = 195` tokens, each with
dimension 1024. Relative to the sole native source tensor, the scalar
compression ratio is about 21.14x. Training inserts the same BF16 bottleneck
used by the offline cache, so the teacher gate does not rely on an unquantized
latent.

The decoder directly reconstructs only layer-11 global. Frozen native VGGT
layers 12--23 then continue normally from that reconstructed state. The
original camera head consumes the resumed final tap and supplies the native
downstream gate. Depth/point are deliberately not used: their DPT heads need
layer-4 and layer-11 frame skip taps, which cannot be obtained by strictly
continuing after layer-11 global. No synthetic early tap is created. Codec
training is teacher-only and sees neither student features nor trajectories.

Student training reads only the compact offline cache; it never imports VGGT.
All 195 student slots enter the trajectory DiT through one bounded centered
residual:

```text
A_v3 = A + alpha * (CrossAttention(A, R195) - CrossAttention(A, slot_mean195))
alpha in [0.05, 0.50]
```

There is no second `extra_context` path. Subtracting the fixed slot-template
readout prevents a static teacher mean from looking like useful scene
knowledge while preserving the scene residual.

## Exact V3 algorithm and reasons for every V1/V2 change

Let `G11` be only the global half of VGGT aggregated tap 11:

```text
G11 = aggregator(images)[11][..., 1024:]       # [B,3,1374,1024]
Zt  = LN(Encoder11Global(G11))                 # [B,195,1024]
Zc  = BF16(Zt)                                 # exact cache bottleneck
G11_hat = Decoder(Zc)
H17_hat,H23_hat = FrozenVGGTBlocks12to23(G11_hat)
camera_hat = FrozenCameraHead(H23_hat)
```

`Encoder11Global` retains all 15 native special tokens and adaptively pools
each view's 37x37 patch grid to 6x10 before a shared token MLP. Its only direct
feature objective is layer-11-global cosine plus Smooth-L1 reconstruction.
`H17/H23` similarity is validation-only, not another training feature target.
The frozen camera output provides the task-preservation loss. The codec loss
is therefore:

```text
Lcodec = Lreconstruct(G11_hat,G11)
       + 0.5 * Lnative_camera
```

After the teacher codec passes held-out gates, only `Zc` is cached. Student
Qwen features produce 15 special plus 180 spatial slots, and the shared
adapter/aligner learns `Zs ~= Zc`. The trajectory path is:

```text
M(z) = CrossAttention(action_queries, Structured195(z))
alpha = 0.05 + 0.45 * sigmoid(scale_logit)
action_queries_v3 = action_queries + alpha * (M(Zs) - M(slot_mean))
DiT(action_queries_v3, extra_context=None)
```

The formal student objective is action flow matching plus global, spatial and
cross-scene latent alignment with weights `1.0/0.05/0.10/0.05`. Student
checkpoints are evaluated by substituting `Zs` for `Zt` in the same frozen
decoder, VGGT tail and native camera head; no downstream probe is fitted.

| Change | V1 | V2 | V3 design and reason |
|---|---|---|---|
| Teacher source | DPT-pre/multi-scale-style 63-token target | layer-11 global, pooled to 195 | Still **only layer-11 global**. The source is deliberately unchanged from V2, so a result difference cannot be attributed to adding layer 4/frame features. |
| Capacity | 63 tokens (coarser 4x4 spatial layout) | 195 tokens | 195 tokens retained. V1 could discard scene detail; V3 does not reintroduce that bottleneck. |
| Teacher representation | Similarity target without native reconstruction contract | Fixed pooling of layer-11 global | Learned 195-token compressive code that must reconstruct layer-11 global and retain frozen native VGGT task outputs. This tests knowledge, not just cosine similarity. |
| Additional VGGT tensors | Not reusable as a native-tail state | No decoder | None. Decoder output is exactly layer-11 global; frozen layers 12--23 continue normally. This keeps the experiment causally clean. |
| Downstream knowledge test | Learned probes/trajectory score could confound representation and reader | Alignment and a fitted geometry probe | Frozen original VGGT tail plus camera head, identical for teacher and student, with no fitted student probe. Native depth/point are excluded because their pre-11 skip inputs would violate the source constraint. |
| Planner entry | Multiple/coarse reader paths | Action queries plus separately appended memory readout (`extra_context`) | One centered residual modifies the eight action queries and returns `extra_context=None`. This removes the V2 bypass/dual-entry ambiguity. |
| Static template | No explicit cancellation | Teacher slot mean could contribute a scene-independent prior | The same reader processes real memory and slot mean, and their outputs are subtracted. Only scene-varying information can change planning. |
| Optimization safety | Direct conditioning could perturb the clean action path | Reader could dominate or be ignored | A learnable scale is bounded to `[0.05,0.50]` and starts at `0.10`, preserving a near-clean initialization while guaranteeing a nonzero route. |
| Attribution | Alignment and planning were mixed in final PDMS | Strong alignment still did not explain poor utilization | Three gates are separated: teacher codec sufficiency, student native-task retention, and planner real/zero/shuffled/slot-mean interventions plus PDMS. |
| Artifact safety | No V3 source identity | V2 cache identity | Codec schema 3 and cache manifest hard-code layer `11`, branch `global`, codec hash and gate results; obsolete or synthetic-tap artifacts fail closed. |

## DLC paths

Create an ignored `env.local.sh` inside the checkout. Do not put these
machine-specific mounts in shared YAML:

```bash
export DRIVEDREAMER_SHARED_ROOT=/mnt/shared/DriveDreamer-Policy
export SHARED_WEIGHT_ROOT=/mnt/shared/model_weights
export DATA_ROOT=/mnt/shared/navsim_dataset
export NAVSIM_DATALIST_PATH=/mnt/shared/train_meta.json
export NAVSIM_PUBLIC_ROOT=/mnt/shared/navsim_dataset_raw
export NAVSIM_TRAINVAL_SENSOR_ROOT=/mnt/shared/navsim_dataset_raw/sensor_blobs/trainval
export NAVSIM_EXP_ROOT=/mnt/shared/navsim_exp

export VGGT_REPO=/mnt/shared/models/vggt
export VGGT_CHECKPOINT=/mnt/shared/models/VGGT-1B/model.safetensors
export VGGT_SOURCE_VLM=/mnt/shared/models/Qwen3-VL-2B-WorldAction
export VGGT_BASE_VLM=/mnt/shared/models/Qwen3-VL-2B-VGGTAction-V2-G15

export VGGT_V3_CODEC_ROOT=/mnt/shared/navsim_feature_cache/vggt_native_codec_v3_l11_global
export VGGT_V3_CODEC=$VGGT_V3_CODEC_ROOT/native_codec.pt
export NAVSIM_VGGT_V3_CACHE_ROOT=/mnt/shared/navsim_feature_cache/vggt_query_train_v3_layer11_global_codec_m195
```

Explicit CLI values retain highest priority, then `env.local.sh`, then the
portable defaults in `env.sh`.

## One-command PAI-DLC run

To start the required data path immediately, without runtime/tail/cache
validation, smoke training or post-training gates, use:

```bash
cd /path/to/DriveDreamer-Policy

PER_DEVICE_BATCH_SIZE=2 \
TARGET_EFFECTIVE_BATCH_SIZE=32 \
VGGT_CODEC_STEPS=10000 \
MAX_TRAIN_STEPS=100000 \
SAVE_INTERVAL=5000 \
RUN_ID=vggt-v3-layer11-global-codec-m195-seed20260814 \
bash 13-run_vggt_v3_direct_cache_train.sh
```

This direct launcher executes only `codec training -> V3 cache -> formal
student training`. The codec stage is not a preflight: it is required to
create the reconstructable 195-token cache.

Use one official PAI-PPU DLC node with 16 visible PPUs. Keep the image's
preinstalled PPU PyTorch/communication stack. From the repository root:

```bash
cd /path/to/DriveDreamer-Policy
chmod +x 12-run_vggt_v3_pipeline.sh 8-train_vggt_v3_action.sh \
  tools/train_vggt_native_codec.sh tools/cache_vggt_v3_queries.sh

VGGT_EXPECTED_PPU_COUNT=16 \
PER_DEVICE_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=1 \
TARGET_EFFECTIVE_BATCH_SIZE=32 \
VGGT_CODEC_NUM_PROCESSES=16 \
VGGT_CACHE_NUM_PROCESSES=16 \
MAX_TRAIN_STEPS=100000 \
SAVE_INTERVAL=5000 \
RUN_ID=vggt-v3-layer11-global-codec-m195-seed20260814 \
bash 12-run_vggt_v3_pipeline.sh
```

The script is restart-safe. Codec training writes an atomic resumable state
every 250 steps by default; completed token-model, gated-codec and V3-cache
stages are validated and skipped. It never deletes or silently overwrites an
incomplete artifact. Its stages are:

1. PPU/BF16/package/collective preflight.
2. Validate or create the 15-token Qwen model.
3. Verify exact continuation from the real layer-11 state through frozen VGGT
   layers 12--23.
4. Train the native codec on 4096 teacher samples and gate it on 512 held-out
   samples (defaults: 10k steps).
5. Materialize and fully validate the strict V3 LMDB cache.
6. Run a two-step V3 forward/backward/intervention smoke test.
7. Train the 100k-step V3 student/planner.
8. Summarize correct/zero/shuffled/slot-mean planner interventions and gradient
   flow from the training diagnostics.
9. Decode the final student features through the frozen codec/VGGT native
   tail and camera head, then write `v3_native_downstream.json`.

Before purchasing a full job, inspect the exact topology and commands without
creating artifacts:

```bash
VGGT_PIPELINE_DRY_RUN=1 \
VGGT_EXPECTED_PPU_COUNT=16 \
LOCAL_NUM_PROCESSES=16 \
bash 12-run_vggt_v3_pipeline.sh
```

Runtime-only validation is also available:

```bash
VGGT_PIPELINE_PREFLIGHT_ONLY=1 bash 12-run_vggt_v3_pipeline.sh
```

## Manual/recovery commands

Each stage can be run independently:

```bash
# 1. Teacher-only codec and held-out native downstream gates
bash tools/train_vggt_native_codec.sh

# 2. Strict offline V3 cache
bash tools/cache_vggt_v3_queries.sh
python tools/precompute_vggt_query_cache.py \
  --validate-only \
  --datalist-path "$NAVSIM_DATALIST_PATH" \
  --data-root "$DATA_ROOT" \
  --cache-root "$NAVSIM_VGGT_V3_CACHE_ROOT"

# 3. Formal student/planner training; no VGGT package is needed in this phase
RUN_ID=vggt-v3-layer11-global-codec-m195-seed20260814 \
bash 8-train_vggt_v3_action.sh

# 4. Direct downstream knowledge gate for any saved V3 checkpoint
python tools/evaluate_vggt_v3_native_downstream.py \
  --run-dir "$NAVSIM_EXP_ROOT/vggt-v3-layer11-global-codec-m195-seed20260814" \
  --checkpoint-step 80000 \
  --base-vlm "$VGGT_BASE_VLM" \
  --native-codec "$VGGT_V3_CODEC" \
  --vggt-repo "$VGGT_REPO" \
  --vggt-checkpoint "$VGGT_CHECKPOINT" \
  --cache-root "$NAVSIM_VGGT_V3_CACHE_ROOT" \
  --datalist-path "$NAVSIM_DATALIST_PATH" \
  --data-root "$DATA_ROOT" \
  --samples 256 \
  --output "$NAVSIM_EXP_ROOT/vggt-v3-layer11-global-codec-m195-seed20260814/downstream-80k.json"
```

If the codec gate fails, do not build the full cache. If the codec passes but
the student native-downstream retention fails, improve representation
learning/alignment. If both pass while PDMS or the correct-vs-shuffled
trajectory gap does not, the remaining problem is planner utilization rather
than feature knowledge.

The default teacher-codec thresholds are layer-11-global reconstruction cosine
`>= 0.90`, resumed layer-23 cosine `>= 0.85`, and camera R² `>= 0.25` on the
held-out codec split. The student gate requires at least 70% of the teacher
codec's camera R² through the identical frozen downstream path.
These thresholds are embedded into the codec checkpoint and copied into the
V3 cache manifest so the provenance is auditable.
