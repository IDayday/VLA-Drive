#!/usr/bin/env bash

# One predeclared Top-64 path-attention experiment.  It tests whether the
# Base-relative conservative scorer is limited by the Top-16/32 deployment
# shortlist.  It consumes only the frozen No-VQA current-observation cache,
# proposals and deployable M0 context; no actor/future/PDM target is an input.

set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
python_bin="${DRIVEVLA_PYTHON:-/mnt/project/DriveVLA-M0-env/bin/python}"
source_root="${NO_VQA_SOURCE_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1}"
label_root="${NO_VQA_LABEL_ROOT:-/root/scorer_pdms93_cache/no_vqa_e35_labels_full_v1}"
private_root="${NO_VQA_MULTIVIEW_TRAIN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_multiview_trainval_pool2_tiles4_v1_8shard}"
split_manifest="${NO_VQA_SPLIT_MANIFEST:-${repo_root}/reports/scorer_pdms93/OFFICIAL_SCORER_SPLIT.json}"
name="rawpointcombined_top64_reference_q50_strict_noactor_seed2"
run_root="${NO_VQA_WAVE9_RUN_ROOT:-/root/scorer_pdms93_runs/no_vqa_e35_point_top64_wave9_v1}"
log_root="${NO_VQA_WAVE9_LOG_ROOT:-/root/scorer_pdms93_logs/no_vqa_e35_point_top64_wave9_v1}"
gpu="${NO_VQA_WAVE9_GPU:-0}"
poll_seconds="${NO_VQA_WAVE9_POLL_SECONDS:-30}"

for path in "${source_root}" "${label_root}" "${split_manifest}"; do
  [[ -e "${path}" ]] || { echo "missing Wave-9 input: ${path}" >&2; exit 2; }
done
[[ -f "${private_root}/.complete" ]] || {
  echo "incomplete Wave-9 current-observation cache: ${private_root}" >&2
  exit 2
}
if [[ -e "${run_root}" || -e "${log_root}" ]]; then
  echo "Wave-9 output already exists; refusing overwrite" >&2
  exit 2
fi

while true; do
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= 1024 )) && break
  echo "NO_VQA_WAVE9 waiting_for_gpu=${gpu} utc=$(date -u +%FT%TZ)"
  sleep "${poll_seconds}"
done

mkdir -p "${run_root}" "${log_root}"
export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

"${python_bin}" "${repo_root}/local_stage2/train_m0_private_residual_scorer.py" \
  --source no_vqa_e35 "${source_root}" "${label_root}" \
  --private-observation-root "${private_root}" \
  --trajectory-observation-attention \
  --split-manifest "${split_manifest}" \
  --selection-source no_vqa_e35 \
  --epochs 8 \
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
  --candidate-layers 1 \
  --fine-layers 2 \
  --private-fine-top-k 16 \
  --residual-layers 2 \
  --max-residual 0.5 \
  --minimum-pair-delta 0.02 \
  --factor-rank-minimum-delta 0.05 \
  --score-mode hybrid \
  --seed 2 \
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
  --private-factor-weight 0 \
  --factor-rank-weight 0 \
  --relative-safety-weight 0 \
  --residual-l2-weight 0 \
  --safety-negative-weight 1 \
  --factor-loss-scope topk \
  --m0-candidate-fusion \
  --residual-top-k 64 \
  --current-actor-weight 0 \
  --output-dir "${run_root}/${name}" \
  >"${log_root}/${name}.log" 2>&1

touch "${run_root}/.wave9_complete"
echo "NO_VQA_WAVE9_COMPLETE run_root=${run_root}"
