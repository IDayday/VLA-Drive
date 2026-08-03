# DriveDreamer-Policy on PPU

This file records the environment and paths tested in this workspace. The
preinstalled PPU builds of PyTorch and Triton are intentionally retained; do
not install the upstream NVIDIA PyTorch wheels from the original README.

## Tested core runtime

| Component | Version |
|---|---|
| Python | 3.10.13 |
| PyTorch | 2.4.0 (PPU-adapted build) |
| torchvision | 0.19.0 |
| torchaudio | 2.4.0 |
| Triton | 3.0.0+ppu1.7.0.oe |
| CUDA-compatible SDK | 12.4 |
| Accelerator | 2 x PPU-ZW810E, 97,920 MiB each |

The environment was completed incrementally around this runtime. Important
tested package versions include:

```text
accelerate==1.13.0
albucore==0.0.17
albumentations==1.4.18
deepspeed==0.16.9
diffusers==0.35.2
librosa==0.11.0
moviepy==1.0.3
numpy==1.26.4
opencv-python-headless==4.11.0.86
plyfile==1.0.3
pycolmap==4.1.1
pytorch-lightning==2.2.1
qwen-vl-utils==0.0.14
scikit-learn==1.2.2
transformers==4.57.0
trimesh==4.12.2
wandb==0.22.2
```

NAVSIM v2 is installed editable from the bundled `navsim/` directory. The
local Depth-Anything-3 source under `depth_process/Depth-Anything-3/` is used
directly. `diffusers==0.35.2` and `numpy==1.26.4` are deliberate compatibility
pins for this PyTorch/runtime combination.

## Weight layout

Reusable open models are stored under the shared root with one consistent
Hugging Face-style hierarchy:

```text
/mnt/data_and_weight/VLA_Group/LLM_weight/
├── Qwen/
│   └── Qwen3-VL-2B-Instruct/
├── alibaba-pai/
│   └── Wan2.1-Fun-V1.1-1.3B-InP/
├── depth-anything/
│   ├── da3metric-large/
│   └── Depth-Anything-V2-Large/
└── gangweix/
    └── Pixel-Perfect-Depth/
```

The hierarchy is always:

```text
LLM_weight/<owner-or-organization>/<repository>/
```

Project-derived and project-specific weights stay inside this repository:

```text
weights/
├── derived/
│   └── Qwen3-VL-2B-WorldAction/
└── yangzhou99/
    └── DriveDreamer-Policy/
        └── pytorch_model.pt
```

`env.sh` is the source of truth for all model paths.

## Dataset layout

The source NAVSIM mirror is:

```text
/mnt/data_and_weight/Public_Space/navsim
```

Project-local symlinks under `navsim_dataset_raw/` expose logs, maps, and
sensor blobs without copying the source dataset. Processed metadata, video,
and depth targets are written under `navsim_dataset/`.

Create the project-local links with:

```bash
source env.sh
bash tools/link_local_navsim.sh
```

This local mirror already has an expanded complete camera tree under
`trainval_all/trainval_sensor_blobs/trainval`; the linker prefers it over the
sparse standard trainval sensor package. All 1,192 training logs and a
representative 32,184 future-frame paths were checked successfully. If only
the 200 camera archives are available on another machine,
`tools/generate_videos_from_camera_archives.sh` provides a resumable streaming
fallback that never modifies the source archives.

The original MoviePy/libx264 `medium` encoder preset remains the default. For
faster preprocessing with the same 1920x1080 resolution, 2 FPS, 9 frames, and
H.264 output format, set `VIDEO_ENCODER_PRESET=veryfast` or `ultrafast`. On
this machine the same nine-frame sample took 2.175 s with `medium`, 0.842 s
with `veryfast`, and 0.242 s with `ultrafast`. Source-image PSNR was 41.76,
41.10, and 41.20 dB respectively; all three outputs contain the same source
frames, while the faster presets trade storage size for encoder throughput.

## Reproducible pipeline

Source the environment before every command:

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy
source env.sh
```

Prepare metadata, videos, and metric depth:

```bash
SPLIT=train bash 0-process_data.sh
SPLIT=train bash 1-gen_data_meta_list.sh
# Optional: only needed when training with NAVSIM_VIDEO_SOURCE=mp4.
SPLIT=train VIDEO_WORKERS=8 VIDEO_ENCODER_PRESET=ultrafast bash 3-gen_videos.sh

SPLIT=train WORLD_SIZE=4 RANK=0 CUDA_VISIBLE_DEVICES=0 bash 2-gen_depth.sh
SPLIT=train WORLD_SIZE=4 RANK=1 CUDA_VISIBLE_DEVICES=1 bash 2-gen_depth.sh
SPLIT=train WORLD_SIZE=4 RANK=2 CUDA_VISIBLE_DEVICES=0 bash 2-gen_depth.sh
SPLIT=train WORLD_SIZE=4 RANK=3 CUDA_VISIBLE_DEVICES=1 bash 2-gen_depth.sh
```

The four depth commands are intended to run concurrently, with two lightweight
Depth-Anything-3 processes per PPU. All preparation steps skip valid existing
outputs, so the same commands resume safely.

`8-train.sh` defaults to `NAVSIM_VIDEO_SOURCE=images` on this machine. It reads
the same nine future frames directly from the verified complete camera tree,
then applies the same tensor resize and normalization as the MP4 path. This
removes the redundant JPEG-to-H.264-to-tensor round trip and allows training
to start without pre-encoding 309,864 MP4 files. Set
`NAVSIM_VIDEO_SOURCE=mp4` to use the original video path instead. The script
also defaults to three data-loader workers per PPU on this 8-CPU host; override
that with `NAVSIM_NUM_WORKERS` when CPU resources differ.

Create the extended Qwen vocabulary once:

```bash
bash 7-add_token.sh
```

Run the released checkpoint on both PPUs:

```bash
SPLIT=test WORLD_SIZE=2 RANK=0 GPU=0 BATCH_SIZE=16 NUM_WORKERS=2 bash 4-infer.sh
SPLIT=test WORLD_SIZE=2 RANK=1 GPU=1 BATCH_SIZE=16 NUM_WORKERS=2 bash 4-infer.sh
```

Inference writes atomic per-token predictions and, by default, resumes by
processing only missing files. Use `OVERWRITE=1` only when predictions should
be regenerated.

Generate the NAVSIM v2 metric cache when necessary and evaluate EPDMS:

```bash
CACHE_WORKERS=2 bash 6-eval_v2.sh
```

Run a one-step end-to-end training smoke test:

```bash
MAX_TRAIN_STEPS=1 GPU=0 bash debug.sh
```

Start the two-PPU paper-scale training configuration:

```bash
NUM_PROCESSES=2 \
PER_DEVICE_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
MAX_TRAIN_STEPS=100000 \
bash 8-train.sh
```

With two processes, batch size 2 per PPU, and eight accumulation steps, the
effective batch size is 32, matching the paper's effective batch size. Batch
size 4 reached the 95.6 GiB PPU memory limit once gradients were retained
across microsteps; batch size 2 uses about 89.9 GiB per PPU. The DeepSpeed
configuration delegates accumulation to Accelerate with `"auto"`, and
evaluation/logging/checkpointing only run after synchronized optimizer steps.
The paper used eight H20 GPUs, so wall-clock throughput is not expected to
match.

### Frozen-feature cache and formal 16-PPU launch

The formal launcher preserves the released effective batch size at 32:

```text
16 processes × 2 samples/process × 1 accumulation step = 32
```

Generate reusable frozen features first. The script is non-interactive,
resumable, uses four samples per PPU only for cache generation, writes one LMDB
per component/rank, and only writes below the dedicated project cache folder.
The throughput-optimized default generates Wan+PPD; Qwen visual features are
recomputed online because reading their 2.7 TiB cache was slower on 16 PPU:

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy
bash ./pre_cache.sh
```

The default durable output is:

```text
/mnt/zhangt_workspace/project/DriveDreamer-Policy/navsim_feature_cache/navsim_v1_qwen3vl2b_wan21_ppd_v1
```

Cached values used by default are Wan VAE posterior parameters and CLIP
context, and PPD semantic features. Qwen visual/deep-stack caching remains
available with `PRECACHE_COMPONENTS=qwen,wan,ppd`, but is not selected by
formal training unless `NAVSIM_CACHE_COMPONENTS=qwen,wan,ppd` is also set.

The tmpfs staging copy uses 8 parallel CPFS streams by default
(`NAVSIM_RAM_COPY_WORKERS=8`). This only affects one-time job startup; it does
not change samples, model inputs, or optimizer behavior.

Wan posterior sampling,
diffusion noise/timesteps, classifier-free dropout, Qwen language layers, all
trainable DiTs, and all losses remain online. Raw tensor payloads project to
about 303 GiB for the default components (about 132 GiB Wan and 171 GiB PPD).

On the formal 16-PPU / 128-CPU / 1-TB-RAM container, `training.sh` allocates two
DataLoader workers per rank (32 total), gives each worker three CPU threads,
and prefetches two batches per worker. It atomically stages Wan+PPD from durable
CPFS into `/dev/shm`. It also stages the 103,288 raw metadata pickles (about
4 GiB), while deliberately excluding the roughly 41 GiB depth-pickle tree
already replaced by the PPD cache. It reserves 400 GiB for model/runtime memory
and falls back to CPFS in `auto` mode if tmpfs is too small. The node-local
copies are disposable; durable cache and dataset files remain in their shared
locations.

After all selected component manifests report `complete: true`, start formal
training:

```bash
bash ./training.sh
```

The training preflight verifies 16 shards, all 103,288 samples, model and
datalist fingerprints, preprocessing settings, FlashAttention, and effective
batch 32 before launching. Set `NAVSIM_USE_FEATURE_CACHE=0` only for the tested
online fallback. Set `NAVSIM_STAGE_CACHE_TO_RAM=0` to disable tmpfs staging.
Triton/compiler caches are intentionally node-local under `/tmp`; durable model
features, logs, and training results remain on shared storage.

The optimized 16-PPU run measured `model_avg=1.420 s` and
`wall_avg=1.427 s` at step 16,729. This corresponds to about 22.4 samples/s at
global batch 32 and roughly 39.6 hours for 100,000 optimizer steps, excluding
job scheduling and one-time cache staging.

## Verified checks

- Qwen3-VL BF16 visual forward on PPU with native SDPA.
- Wan transformer, VAE, and CLIP checkpoints load with no missing or unexpected
  checkpoint keys.
- Depth-Anything-3 metric depth inference on PPU.
- Full training metric-depth generation: exactly 103,288/103,288 atomic
  outputs. Twenty-one evenly spaced samples were deserialized successfully;
  all contain finite `cam_f0`, `cam_l0`, and `cam_r0` arrays of shape
  `(140, 252)` and size 423,600 bytes.
- Pixel-Perfect Depth forward on PPU.
- Full Qwen + action + Wan + PPD training step with DeepSpeed ZeRO-2 on one
  PPU: loss `35.2581`; optimizer step and final checkpoint save completed.
- The same full training step with two PPUs and the NCCL-compatible collective
  backend: rank losses `35.2996` and `39.2175`; optimizer synchronization and
  final checkpoint save completed.
- A two-PPU train-split step using the direct nine-frame image source also
  completed: rank losses `36.5116` and `39.9245`, data time `1.590` s, model
  time `3.216` s, and a 14 GB final checkpoint was saved under
  `navsim_exp/smoke-images-train-bz1-2/`.
- Formal two-PPU training is running under
  `navsim_exp/formal-100k-images-bz2-ga8-2ppu-20260731-v3/`, with 100,000
  optimizer steps, effective batch 32, and initial sustained optimizer-step
  times of about 18–22 seconds.
- Released DriveDreamer checkpoint inference on the 396-sample mini split:
  every output has shape `(8, 3)` and finite values.
- Released DriveDreamer checkpoint inference on all 12,146 navtest samples:
  the token set is complete and every output has shape `(8, 3)` with finite
  values.
- Full NAVSIM v2 evaluation of the released checkpoint: 12,146/12,146
  scenarios succeeded, 0 failed, and EPDMS was `0.8868952559565824`
  (`88.6895`, which rounds to the paper-reported `88.7`). The result CSV is
  `navsim_exp/eval_v2/drivedreamer-policy/2026.07.31.22.21.02/2026.07.31.22.36.31.csv`.
- NAVSIM v2 scoring pipeline smoke test: 2/2 cached scenarios succeeded with
  no failed tokens and produced a valid EPDMS CSV.
- Direct camera-archive video generation: one archive containing 7 logs and
  184 tokens produced exactly 552 valid MP4 files (1920x1080, 2 FPS, 4.5 s)
  and removed its temporary extraction tree.

The one-step training checkpoint is under:

```text
navsim_exp/debug-3d-2d-1d-lr1e5-3d_loss_1e1-decay1e3-mini_data-bz_1_1/
navsim_exp/debug-3d-2d-1d-lr1e5-3d_loss_1e1-decay1e3-mini_data-bz_1_2/
```
