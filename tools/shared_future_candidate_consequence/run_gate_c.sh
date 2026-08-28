#!/usr/bin/env bash
set -euo pipefail

MODE=smoke
SPLIT=trainval
NUM_SCENES=0
NUM_CANDIDATES=16
NUM_FOLDS=5
NUM_SEEDS=3
GPU_LIST="3,5,6,7"
OUTPUT_DIR=reports/shared_future_candidate_consequence_gate_c
CACHE_ROOT=outputs/shared_future_candidate_consequence_gate_c
NAVSIM_PYTHON=/root/miniconda3/envs/navsim/bin/python
VLA_PYTHON=/mnt/project/DriveVLA-M0-env/bin/python

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --num-scenes) NUM_SCENES="$2"; shift 2 ;;
    --num-candidates) NUM_CANDIDATES="$2"; shift 2 ;;
    --num-folds) NUM_FOLDS="$2"; shift 2 ;;
    --num-seeds) NUM_SEEDS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --navsim-python) NAVSIM_PYTHON="$2"; shift 2 ;;
    --vla-python) VLA_PYTHON="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "${SPLIT}" != "train" && "${SPLIT}" != "trainval" ]]; then
  echo "Only legal training splits train/trainval are accepted" >&2
  exit 2
fi
if [[ "${MODE}" != "smoke" && "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "mode must be smoke, pilot or full" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}"

if [[ "${MODE}" == "smoke" ]]; then
  scenes=${NUM_SCENES}
  if (( scenes <= 0 )); then scenes=32; fi
  cache_dir="${CACHE_ROOT}/smoke_gate_c"
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.build_balanced_split \
    --mode smoke --split "${SPLIT}" --num-scenes "${scenes}" --num-folds "${NUM_FOLDS}" \
    --output-dir "${OUTPUT_DIR}"
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.build_controlled_candidates \
    --split "${SPLIT}" --num-scenes "${scenes}" --num-candidates "${NUM_CANDIDATES}" \
    --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.reproduce_gate_c0 \
    --split "${SPLIT}" --num-scenes "${scenes}" \
    --output-dir "${OUTPUT_DIR}" --cache-dir "${CACHE_ROOT}"
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.build_gate_c_targets \
    --split "${SPLIT}" --num-scenes "${scenes}" --actor-slots 16 \
    --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.run_oracle_decomposition \
    --mode smoke --num-scenes "${scenes}" --models linear,mlp --epochs 8 \
    --batch-scenes 16 --device cpu --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
  exit 0
fi

if [[ "${MODE}" == "pilot" ]]; then
  scenes=${NUM_SCENES}
  if (( scenes <= 0 )); then scenes=500; fi
  cache_dir="${CACHE_ROOT}/pilot_gate_c"
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.build_balanced_split \
    --mode pilot --split "${SPLIT}" --num-scenes "${scenes}" --num-folds "${NUM_FOLDS}" \
    --output-dir "${OUTPUT_DIR}"
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.build_controlled_candidates \
    --split "${SPLIT}" --num-scenes "${scenes}" --num-candidates "${NUM_CANDIDATES}" \
    --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.build_gate_c_targets \
    --split "${SPLIT}" --num-scenes "${scenes}" --actor-slots 16 \
    --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.run_oracle_decomposition \
    --mode pilot --num-scenes "${scenes}" --models linear,mlp --epochs 20 \
    --batch-scenes 64 --device cpu --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
  exit 0
fi

cache_dir="${CACHE_ROOT}/all"
if (( NUM_SCENES > 0 )); then
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.build_balanced_split \
    --mode full --split "${SPLIT}" --num-scenes "${NUM_SCENES}" --min-logs 40 \
    --per-log-cap 50 --num-folds "${NUM_FOLDS}" --output-dir "${OUTPUT_DIR}"
else
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.build_balanced_split \
    --mode all_logs --split "${SPLIT}" --per-log-cap 50 --num-folds "${NUM_FOLDS}" \
    --output-dir "${OUTPUT_DIR}"
fi
"${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.reproduce_gate_c0 \
  --split "${SPLIT}" --num-scenes 32 --output-dir "${OUTPUT_DIR}" --cache-dir "${CACHE_ROOT}"

# Current-only proposal export and CPU PDM relabeling are independent and can
# overlap safely; both consume the same immutable log-level fold manifest.
bash tools/shared_future_candidate_consequence/run_episode_drive_export.sh \
  --gpus "${GPU_LIST}" --scenes-per-log 2 --python "${VLA_PYTHON}" \
  --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}" &
export_pid=$!

bash tools/shared_future_candidate_consequence/run_all_log_shards.sh \
  --start-shard 0 --end-shard 63 --num-shards 64 --max-parallel 16 \
  --split "${SPLIT}" --num-candidates "${NUM_CANDIDATES}" --actor-slots 16 \
  --python "${NAVSIM_PYTHON}" --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
wait "${export_pid}"

bash tools/shared_future_candidate_consequence/run_current_actor_augmentation.sh \
  --num-shards 32 --max-parallel 8 --actor-slots 16 --python "${NAVSIM_PYTHON}" \
  --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
"${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.aggregate_all_log_pipeline \
  --num-candidates "${NUM_CANDIDATES}" --actor-slots 16 --sample-targets 512 \
  --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
"${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.build_oracle_store \
  --num-candidates "${NUM_CANDIDATES}" --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"

bash tools/shared_future_candidate_consequence/run_model_candidate_bank.sh \
  --start-shard 0 --end-shard 7 --num-shards 8 --max-parallel 8 \
  --num-candidates "${NUM_CANDIDATES}" --actor-slots 16 --python "${NAVSIM_PYTHON}" \
  --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
"${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.analyze_model_candidate_diversity \
  --num-candidates "${NUM_CANDIDATES}" --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
"${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.build_model_oracle_store \
  --num-candidates "${NUM_CANDIDATES}" --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"

bash tools/shared_future_candidate_consequence/run_oracle_full.sh \
  --gpus "${GPU_LIST}" --epochs 15 --batch-scenes 512 --python "${NAVSIM_PYTHON}" \
  --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
bash tools/shared_future_candidate_consequence/run_model_candidate_oracle.sh \
  --gpu "${GPU_LIST%%,*}" --epochs 15 --python "${NAVSIM_PYTHON}" \
  --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"

gate_c1=$("${NAVSIM_PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["gate_c1"])' \
  "${OUTPUT_DIR}/oracle_decomposition_results.json")
if [[ "${gate_c1}" == "FAIL" ]]; then
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.finalize_gate_c_failure \
    --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
  "${NAVSIM_PYTHON}" -m tools.shared_future_candidate_consequence.visualize_gate_c \
    --output-dir "${OUTPUT_DIR}" --cache-dir "${cache_dir}"
  echo "Gate C1 failed; C2/C3 model development was not entered."
else
  echo "Gate C1 passed. num-seeds=${NUM_SEEDS} is reserved for conditional C2/C3 training."
fi
