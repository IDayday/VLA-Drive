#!/usr/bin/env bash

# Run one Wave-15 held-out-log fold.  Wave-15 is the Wave-14 semantic-BEV
# scorer with one additional current-actor constant-velocity relabeling path.

set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 FOLD GPU" >&2
  exit 2
fi

fold="$1"
gpu="$2"
[[ "${fold}" =~ ^[0-4]$ ]] || { echo "invalid fold: ${fold}" >&2; exit 2; }
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "invalid GPU: ${gpu}" >&2; exit 2; }

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
strict_python=/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python
python_bin="${DRIVEVLA_PYTHON:-${strict_python}}"
shared_base_root="${NO_VQA_BASE_CACHE_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_base_features_labels_full_v1}"
source_root="${NO_VQA_SOURCE_ROOT:-${shared_base_root}/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-${shared_base_root}/no_vqa_e35_labels_full_v1}"
private_root="${NO_VQA_DENSE_TRAIN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_trainval_pool4_tiles4_v1_8shard}"
actor_root="${NO_VQA_FULL_CURRENT_ACTOR_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_full_current_actor_targets_v1}"
semantic_root="${NO_VQA_SEMANTIC_BEV_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_full_current_semantic_bev_targets_v1}"
fold_root="${NO_VQA_WAVE15_FOLD_ROOT:-${repo_root}/reports/m0_independent_scorer_representation/no_vqa_e35_risk_cv_wave12_v1/folds}"
run_root="${NO_VQA_WAVE15_RUN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_semantic_cv_wave15_v1_distributed}"
log_root="${NO_VQA_WAVE15_LOG_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93_logs/no_vqa_e35_semantic_cv_wave15_v1_distributed}"
poll_seconds="${NO_VQA_WAVE15_POLL_SECONDS:-30}"
output="${run_root}/fold_${fold}"
log_file="${log_root}/fold_${fold}.log"
marker="${run_root}/.fold_${fold}_complete"

for path in "${source_root}" "${label_root}" "${fold_root}/fold_${fold}.json"; do
  [[ -e "${path}" ]] || { echo "missing Wave-15 input: ${path}" >&2; exit 2; }
done
for root in "${private_root}" "${actor_root}" "${semantic_root}"; do
  [[ -f "${root}/.complete" ]] || { echo "incomplete Wave-15 cache: ${root}" >&2; exit 2; }
done
[[ -x "${python_bin}" ]] || { echo "missing strict Python runtime: ${python_bin}" >&2; exit 2; }
[[ "$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.9" ]] || {
  echo "Wave-15 requires Python 3.9" >&2
  exit 2
}
if [[ -e "${output}" || -e "${log_file}" || -e "${marker}" ]]; then
  echo "Wave-15 fold output already exists; refusing overwrite: fold=${fold}" >&2
  exit 2
fi

while true; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d '[:space:]')"
  if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= 1024 )); then
    break
  fi
  echo "NO_VQA_WAVE15_FOLD waiting_for_gpu fold=${fold} gpu=${gpu} used_mib=${used} utc=$(date -u +%FT%TZ)"
  sleep "${poll_seconds}"
done

mkdir -p "${run_root}" "${log_root}"
export PYTHONPATH="/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/transformers_4_48_3:/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/lightning_2_2_1:${repo_root}:${repo_root}/nuplan-devkit:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${gpu}"

echo "NO_VQA_WAVE15_FOLD_STARTED fold=${fold} gpu=${gpu} host=$(hostname) python=$("${python_bin}" -c 'import sys; print(sys.version.split()[0])') utc=$(date -u +%FT%TZ)"
"${python_bin}" "${repo_root}/local_stage2/train_m0_private_residual_scorer.py" \
  --source no_vqa_e35 "${source_root}" "${label_root}" \
  --private-observation-root "${private_root}" \
  --selection-source no_vqa_e35 \
  --epochs 8 \
  --save-last-artifact \
  --batch-size 32 \
  --eval-batch-size 64 \
  --num-workers 0 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --bootstrap-replicates 1000 \
  --model-dim 256 \
  --dynamic-queries 16 \
  --private-layers 2 \
  --trajectory-layers 2 \
  --trajectory-observation-attention \
  --semantic-bev-target-root "${semantic_root}" \
  --semantic-bev-weight 0.5 \
  --semantic-bev-fusion \
  --candidate-layers 1 \
  --fine-layers 2 \
  --private-fine-top-k 16 \
  --residual-layers 2 \
  --max-residual 0.5 \
  --minimum-pair-delta 0.02 \
  --factor-rank-minimum-delta 0.05 \
  --score-mode hybrid \
  --seed 2 \
  --scene-sampling-mode risk_balanced \
  --risk-scene-max-multiplier 4 \
  --conservative-reference \
  --reference-hidden-dim 512 \
  --reference-layers 2 \
  --reference-gain-quantile-index 1 \
  --reference-minimum-lcb-gain 0 \
  --reference-maximum-safety-worse-probability 0.1 \
  --reference-minimum-safe-improvement-probability 0.7 \
  --reference-weight 1 \
  --reference-quantile-weight 1 \
  --reference-median-rank-weight 0.25 \
  --reference-safety-weight 1 \
  --reference-improvement-weight 0.5 \
  --reference-false-switch-weight 0.5 \
  --reference-missed-improvement-weight 0 \
  --reference-safety-worse-positive-weight 10 \
  --reference-safe-improvement-positive-weight 3 \
  --reference-switch-margin-temperature 0.05 \
  --reference-minimum-improvement-target 0.005 \
  --reference-factor-epsilon 1e-6 \
  --pairwise-weight 0 \
  --base-pairwise-weight 0 \
  --listwise-weight 0 \
  --top-set-weight 0 \
  --expected-regret-weight 0 \
  --top-regret-weight 0 \
  --factor-weight 0 \
  --private-factor-weight 0.25 \
  --factor-rank-weight 0 \
  --relative-safety-weight 0 \
  --residual-l2-weight 0 \
  --safety-negative-weight 1 \
  --factor-loss-scope topk \
  --current-actor-target-root "${actor_root}" \
  --current-actor-weight 0.5 \
  --current-actor-cv-relabeling \
  --m0-candidate-fusion \
  --residual-top-k 32 \
  --split-manifest "${fold_root}/fold_${fold}.json" \
  --output-dir "${output}" \
  >"${log_file}" 2>&1

touch "${marker}"
all_complete=1
for check_fold in 0 1 2 3 4; do
  [[ -f "${run_root}/.fold_${check_fold}_complete" ]] || all_complete=0
done
if (( all_complete == 1 )); then
  touch "${run_root}/.wave12_folds_complete"
fi
echo "NO_VQA_WAVE15_FOLD_COMPLETE fold=${fold} gpu=${gpu} host=$(hostname) utc=$(date -u +%FT%TZ)"
