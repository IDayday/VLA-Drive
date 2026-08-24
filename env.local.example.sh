#!/usr/bin/env bash
# Copy this file to env.local.sh and uncomment only the values that differ on
# your machine. env.local.sh is ignored by git and must never contain shared
# defaults that other developers are expected to use.

# Accelerator runtime
# export CUDA_HOME="${CUDA_HOME:-/absolute/path/to/cuda}"

# Durable storage and model weights
# export DRIVEDREAMER_SHARED_ROOT="${DRIVEDREAMER_SHARED_ROOT:-/absolute/path/to/project-storage}"
# export SHARED_WEIGHT_ROOT="${SHARED_WEIGHT_ROOT:-/absolute/path/to/model-weights}"
# export HF_HOME="${HF_HOME:-$SHARED_WEIGHT_ROOT/.cache/huggingface}"

# NAVSIM source data. Set the explicit split roots when your downloaded layout
# differs from navsim_dataset_raw/{navsim_logs,sensor_blobs}/...
# export NAVSIM_PUBLIC_ROOT="${NAVSIM_PUBLIC_ROOT:-/absolute/path/to/navsim}"
# export NAVSIM_TRAINVAL_SENSOR_ROOT="${NAVSIM_TRAINVAL_SENSOR_ROOT:-/absolute/path/to/trainval-sensors}"
# export NAVSIM_TEST_LOG_ROOT="${NAVSIM_TEST_LOG_ROOT:-/absolute/path/to/test-logs}"
# export NAVSIM_TEST_SENSOR_ROOT="${NAVSIM_TEST_SENSOR_ROOT:-/absolute/path/to/test-sensors}"
# export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/absolute/path/to/nuplan-maps}"
# export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$NAVSIM_PUBLIC_ROOT}"

# Processed metadata, caches, experiment outputs and datalists
# export TRAIN_CONFIG_YAML="${TRAIN_CONFIG_YAML:-$DRIVEDREAMER_ROOT/starVLA/config/training/cfg_yaw_1225.yaml}"
# export DATA_ROOT="${DATA_ROOT:-/absolute/path/to/processed-navsim}"
# export NAVSIM_DATALIST_PATH="${NAVSIM_DATALIST_PATH:-/absolute/path/to/train_meta.json}"
# export NAVSIM_VIDEO_ROOT="${NAVSIM_VIDEO_ROOT:-$DATA_ROOT/navsim_video}"
# export NAVSIM_GS_ROOT="${NAVSIM_GS_ROOT:-/absolute/path/to/navsim-storm}"
# export NAVSIM_REWARD_ROOT="${NAVSIM_REWARD_ROOT:-/absolute/path/to/navsim-reward}"
# export NAVSIM_QA_ROOT="${NAVSIM_QA_ROOT:-/absolute/path/to/navsim-qa-output}"
# export NAVSIM_VQA_ROOT="${NAVSIM_VQA_ROOT:-/absolute/path/to/navsim-vqa-output}"
# export NAVSIM_LIDAR_DEPTH_ROOT="${NAVSIM_LIDAR_DEPTH_ROOT:-/absolute/path/to/lidar-depth}"
# export NAVSIM_MINI_TEST_DATALIST="${NAVSIM_MINI_TEST_DATALIST:-$DRIVEDREAMER_ROOT/mini_meta.json}"
# export NAVSIM_FEATURE_CACHE_ROOT="${NAVSIM_FEATURE_CACHE_ROOT:-/absolute/path/to/navsim-feature-cache}"
# export NAVSIM_AGENT_DINO_CACHE_ROOT="${NAVSIM_AGENT_DINO_CACHE_ROOT:-/absolute/path/to/agent-dino-cache}"
# export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/absolute/path/to/experiments}"
# export NAVSIM_EVAL_ROOT="${NAVSIM_EVAL_ROOT:-$NAVSIM_EXP_ROOT}"
# export NAVSIM_V1_METRIC_CACHE_PATH="${NAVSIM_V1_METRIC_CACHE_PATH:-/absolute/path/to/v1-metric-cache}"
# export NAVSIM_V2_METRIC_CACHE_ROOT="${NAVSIM_V2_METRIC_CACHE_ROOT:-/absolute/path/to/v2-metric-caches}"
# Frozen/trainable visual action-only checkpoint comparison. The frozen run
# may live in a collaborator-specific shared mount; keep it out of shared YAML.
# export FROZEN_ACTION_ONLY_RUN_DIR="${FROZEN_ACTION_ONLY_RUN_DIR:-$NAVSIM_EXP_ROOT/frozen-action-only-run}"
# export VISUAL_ACTION_ONLY_RUN_DIR="${VISUAL_ACTION_ONLY_RUN_DIR:-$NAVSIM_EXP_ROOT/qwen-visual-action-only-run}"
# export ACTION_ONLY_COMPARISON_ROOT="${ACTION_ONLY_COMPARISON_ROOT:-$NAVSIM_EVAL_ROOT/action-only-visual-comparison}"
# Weight-only continuation of the completed trainable-visual action-only run.
# The launcher verifies that the selected datalist has 103,288 unique samples.
# export QWEN_VISUAL_SOURCE_RUN="${QWEN_VISUAL_SOURCE_RUN:-$NAVSIM_EXP_ROOT/qwen-visual-action-only-20260814_001706}"
# export QWEN_VISUAL_SOURCE_STEP="${QWEN_VISUAL_SOURCE_STEP:-100000}"
# export EXPECTED_TRAIN_SAMPLES="${EXPECTED_TRAIN_SAMPLES:-103288}"

# Models/checkpoints
# export SOURCE_VLM="${SOURCE_VLM:-/absolute/path/to/Qwen3-VL-2B-Instruct}"
# export BASE_VLM="${BASE_VLM:-/absolute/path/to/Qwen3-VL-2B-WorldAction}"
# export VIDEO_MODEL="${VIDEO_MODEL:-/absolute/path/to/Wan2.1-Fun-V1.1-1.3B-InP}"
# export VIDEO_CONFIG="${VIDEO_CONFIG:-$DRIVEDREAMER_ROOT/starVLA/model/modules/video_model/config/wan2.1/wan_civitai.yaml}"
# export PPD_MODEL="${PPD_MODEL:-/absolute/path/to/ppd.pth}"
# export DEPTH_ANYTHING_MODEL="${DEPTH_ANYTHING_MODEL:-/absolute/path/to/depth-anything-v2.pth}"
# export DA3_MODEL="${DA3_MODEL:-/absolute/path/to/da3metric-large}"
# export GS_MODEL_PATH="${GS_MODEL_PATH:-/absolute/path/to/gs-model.pth}"
# export RELEASE_MODEL="${RELEASE_MODEL:-/absolute/path/to/pytorch_model.pt}"
# Optional VGGT teacher. These are required only to generate the offline cache.
# export VGGT_REPO="${VGGT_REPO:-/absolute/path/to/facebookresearch/vggt}"
# export VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-/absolute/path/to/VGGT-1B/model.safetensors}"
# Separate Qwen checkpoint extended with 15 VGGT V2 global tokens (step 7 VGGT).
# export VGGT_SOURCE_VLM="${VGGT_SOURCE_VLM:-$BASE_VLM}"
# export VGGT_BASE_VLM="${VGGT_BASE_VLM:-/absolute/path/to/Qwen3-VL-2B-VGGTAction-V2-G15}"
# export VGGT_TOKEN_DEVICE="${VGGT_TOKEN_DEVICE:-cpu}"  # set cuda on a compatible GPU
# export NAVSIM_VGGT_CACHE_ROOT="${NAVSIM_VGGT_CACHE_ROOT:-/absolute/path/to/vggt-query-v2-layer11-global-m195-cache}"
# V3 teacher-only codec and its separate reconstructable 195-token cache.
# export VGGT_V3_CODEC_ROOT="${VGGT_V3_CODEC_ROOT:-/absolute/path/to/vggt-native-codec-v3-layer11-global}"
# export VGGT_V3_CODEC="${VGGT_V3_CODEC:-$VGGT_V3_CODEC_ROOT/native_codec.pt}"
# export NAVSIM_VGGT_V3_CACHE_ROOT="${NAVSIM_VGGT_V3_CACHE_ROOT:-/absolute/path/to/vggt-query-v3-layer11-global-codec-m195-cache}"
# Final-layer, unpooled VGGT patch cache used by QwenOFT_VGGT_Bottleneck.
# export NAVSIM_VGGT_DENSE_CACHE_ROOT="/path/to/vggt_dense_cache"
# Split-specific SQ-3D-Mix navtest cache and evaluation outputs. The PDMS and
# EPDMS overrides are optional when the shared NAVSIM metric-cache defaults
# already point at complete navtest caches.
# export NAVSIM_VGGT_DENSE_TEST_CACHE_ROOT="/path/to/vggt_dense_navtest_cache"
# export SQ3DMIX_EVAL_OUT_ROOT="/path/to/sq3dmix_navtest_results"
# export PDMS_METRIC_CACHE_PATH="/path/to/navsim_v1_1_navtest_metric_cache"
# export EPDMS_METRIC_CACHE_PATH="/path/to/navsim_v2_navtest_metric_cache"
# Dense-cache generation. Leave VGGT_DENSE_CACHE_MAX_SAMPLES unset for the
# complete datalist. map_size is a per-rank upper bound, not eager allocation.
# export VGGT_DENSE_CACHE_NUM_PROCESSES="${VGGT_DENSE_CACHE_NUM_PROCESSES:-16}"
# export VGGT_DENSE_CACHE_BATCH_SIZE="${VGGT_DENSE_CACHE_BATCH_SIZE:-1}"
# export VGGT_DENSE_CACHE_MAP_SIZE_GB="${VGGT_DENSE_CACHE_MAP_SIZE_GB:-80}"
# export VGGT_DENSE_CACHE_FULL="${VGGT_DENSE_CACHE_FULL:-1}"  # ignores a stale smoke max-samples setting
# Full bottleneck training may initialize from an action-only model checkpoint.
# export ACTION_ONLY_CHECKPOINT="/path/to/action_only/pytorch_model.pt"
# GP-SQ3D-Mix Stage-A-v2 uses a new immutable asset/run namespace.  Keep these
# separate from the legacy gp_sq3dmix_slot_stats and historical experiment dirs.
# export GP_SQ3DMIX_TRAIN_CACHE_ROOT="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
# export GP_SQ3DMIX_TEST_CACHE_ROOT="${GP_SQ3DMIX_TEST_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_TEST_CACHE_ROOT}"
# export GP_SQ3DMIX_TRAIN_DATALIST="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
# export GP_SQ3DMIX_V2_STATS_ROOT="${GP_SQ3DMIX_V2_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_slot_stats}"
# export GP_SQ3DMIX_V2_NEGATIVE_MAP_ROOT="${GP_SQ3DMIX_V2_NEGATIVE_MAP_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_negative_maps}"
# export GP_SQ3DMIX_V2_RUN_ROOT="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
# export GP_SQ3DMIX_V2_EVAL_ROOT="${GP_SQ3DMIX_V2_EVAL_ROOT:-$NAVSIM_EVAL_ROOT/gp_sq3dmix_stage_a_v2_eval}"
# export GP_SQ3DMIX_STATS_WORKERS="${GP_SQ3DMIX_STATS_WORKERS:-8}"
# export GP_SQ3DMIX_STATS_SHARDS="${GP_SQ3DMIX_STATS_SHARDS:-16}"
# export GP_SQ3DMIX_NEGATIVE_MAP_WORKERS="${GP_SQ3DMIX_NEGATIVE_MAP_WORKERS:-16}"
# export VGGT_DENSE_LEARNING_RATE="${VGGT_DENSE_LEARNING_RATE:-5e-5}"
# export VGGT_DENSE_VLM_ATTN_IMPLEMENTATION="${VGGT_DENSE_VLM_ATTN_IMPLEMENTATION:-sdpa}"
# export VGGT_DENSE_INTERVENTION_INTERVAL="${VGGT_DENSE_INTERVENTION_INTERVAL:-500}"
# One-command dense-cache + bottleneck training auto-selects batch 2 on 16 PPU
# and batch 4 on 8 PPU. Set this only when an exact device-count gate is wanted.
# export VGGT_DENSE_EXPECTED_PPU_COUNT="${VGGT_DENSE_EXPECTED_PPU_COUNT:-8}"
# export VGGT_DENSE_RUN_SMOKE_BEFORE_FORMAL="${VGGT_DENSE_RUN_SMOKE_BEFORE_FORMAL:-1}"
# Direct launcher normally detects all visible devices; set only to use fewer.
# export LOCAL_NUM_PROCESSES="${LOCAL_NUM_PROCESSES:-8}"
# export VGGT_CODEC_NUM_PROCESSES="${VGGT_CODEC_NUM_PROCESSES:-$LOCAL_NUM_PROCESSES}"
# export VGGT_CODEC_TRAIN_SAMPLES="${VGGT_CODEC_TRAIN_SAMPLES:-4096}"
# export VGGT_CODEC_VALIDATION_SAMPLES="${VGGT_CODEC_VALIDATION_SAMPLES:-512}"
# export VGGT_CODEC_STEPS="${VGGT_CODEC_STEPS:-10000}"
# export VGGT_CODEC_CHECKPOINT_INTERVAL="${VGGT_CODEC_CHECKPOINT_INTERVAL:-250}"
# export VGGT_CODEC_RESUME="${VGGT_CODEC_RESUME:-1}"
# export VGGT_CACHE_NUM_PROCESSES="${VGGT_CACHE_NUM_PROCESSES:-$LOCAL_NUM_PROCESSES}"
# export VGGT_CACHE_BATCH_SIZE="${VGGT_CACHE_BATCH_SIZE:-1}"
# export VGGT_CACHE_MAP_SIZE_GB="${VGGT_CACHE_MAP_SIZE_GB:-16}"  # per rank
# Low-cost VGGT spatial-resolution probe. The report is written atomically and
# does not create or modify feature caches.
# export VGGT_RESOLUTION_PROBE_OUTPUT="${VGGT_RESOLUTION_PROBE_OUTPUT:-$NAVSIM_EXP_ROOT/vggt_resolution_probe/probe.json}"
# export VGGT_RESOLUTION_PROBE_SAMPLES="${VGGT_RESOLUTION_PROBE_SAMPLES:-1024}"
# export VGGT_RESOLUTION_PROBE_BATCH_SIZE="${VGGT_RESOLUTION_PROBE_BATCH_SIZE:-1}"
# export VGGT_RESOLUTION_PROBE_DEVICE="${VGGT_RESOLUTION_PROBE_DEVICE:-auto}"
# Layer-11 global geometry probe. It reads the local VGGT checkpoint and
# token-matched NAVSIM PCDs, but never modifies the query cache.
# export VGGT_GEOMETRY_PROBE_OUTPUT="${VGGT_GEOMETRY_PROBE_OUTPUT:-$NAVSIM_EXP_ROOT/vggt_geometry_probe/probe.json}"
# export VGGT_GEOMETRY_PROBE_TRAIN_SAMPLES="${VGGT_GEOMETRY_PROBE_TRAIN_SAMPLES:-96}"
# export VGGT_GEOMETRY_PROBE_VAL_SAMPLES="${VGGT_GEOMETRY_PROBE_VAL_SAMPLES:-32}"
# export VGGT_GEOMETRY_PROBE_GRID_ROWS="${VGGT_GEOMETRY_PROBE_GRID_ROWS:-6}"
# export VGGT_GEOMETRY_PROBE_GRID_COLS="${VGGT_GEOMETRY_PROBE_GRID_COLS:-10}"
# export VGGT_GEOMETRY_PROBE_LIDAR_MIN_POINTS="${VGGT_GEOMETRY_PROBE_LIDAR_MIN_POINTS:-3}"
# export VGGT_GEOMETRY_PROBE_DEVICE="${VGGT_GEOMETRY_PROBE_DEVICE:-auto}"
# One-command PAI-DLC PPU pipeline. Defaults target one 16-PPU node and an
# effective batch of 32; only uncomment intentional machine/run overrides.
# export VGGT_EXPECTED_PPU_COUNT="${VGGT_EXPECTED_PPU_COUNT:-16}"
# export VGGT_VLM_ATTN_IMPLEMENTATION="${VGGT_VLM_ATTN_IMPLEMENTATION:-sdpa}"
# export VGGT_CACHE_FULL_VALIDATE="${VGGT_CACHE_FULL_VALIDATE:-1}"
# export VGGT_RUN_SMOKE_BEFORE_FORMAL="${VGGT_RUN_SMOKE_BEFORE_FORMAL:-1}"
# export VGGT_INTERVENTION_INTERVAL="${VGGT_INTERVENTION_INTERVAL:-500}"
# export TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
# export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
# export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
# export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
# export BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-5}"
# export ACTION_LEARNING_RATE="${ACTION_LEARNING_RATE:-1e-5}"
# export VGGT_LEARNING_RATE="${VGGT_LEARNING_RATE:-3e-5}"
# export OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1e-3}"
# export TRITON_LOCAL_CACHE_ROOT="${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}"
# export GAUSSIAN_STORM_THIRD_PARTY="${GAUSSIAN_STORM_THIRD_PARTY:-/absolute/path/to/GaussianSTORM/third_party}"

# Optional VLM co-training datasets
# export VLM_ONEVISION_ROOT="${VLM_ONEVISION_ROOT:-/absolute/path/to/LLaVA-OneVision-Data}"
# export VLM_DATA_ROOT="${VLM_DATA_ROOT:-/absolute/path/to/coco}"
# export VLM_LLAVA_FORMAT_ROOT="${VLM_LLAVA_FORMAT_ROOT:-/absolute/path/to/qwen-formatted-jsonl}"

# Optional experiment tracking
# export WANDB_API_KEY="${WANDB_API_KEY:-}"
# export WANDB_ENTITY="${WANDB_ENTITY:-your-entity}"
# export WANDB_PROJECT="${WANDB_PROJECT:-drivedreamer-policy}"
# export WANDB_MODE="${WANDB_MODE:-offline}"
