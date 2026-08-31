#!/usr/bin/env bash
set -euo pipefail

repo_root="${DRIVEVLA_REPO_ROOT:-/mnt/project/DriveVLA-M0-scorer-pdms93}"
run_root="${DRIVEVLA_SCORER_RUN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93}"
source_root="${DRIVEVLA_SCORER_SOURCE_ROOT:-${run_root}/public_base_features_full_v1}"
label_root="${DRIVEVLA_SCORER_LABEL_ROOT:-${run_root}/public_base_labels_full_v1}"
base_checkpoint="${DRIVEVLA_PUBLIC_BASE:-/mnt/project/DriveVLA-M0-modelscope/best-epoch_26-step_174312.server_merged.ckpt}"
python_bin="${DRIVEVLA_EXACT_PYTHON:-/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python}"
poll_seconds="${DRIVEVLA_SCORER_POLL_SECONDS:-30}"
wait_pids="${DRIVEVLA_WAIT_PIDS:-}"
gpu_csv="${DRIVEVLA_SCORER_GPUS:-3,5,6,7}"

export PYTHONPATH="${repo_root}:${repo_root}/nuplan-devkit:/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/transformers_4_48_3:/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/lightning_2_2_1:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -r -a gpus <<< "${gpu_csv}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
  echo "DRIVEVLA_SCORER_GPUS must list exactly four GPU indices" >&2
  exit 2
fi

cache_complete() {
  local source_manifests label_manifest
  source_manifests="$(find "${source_root}" -mindepth 2 -maxdepth 2 -name manifest.json -type f 2>/dev/null | wc -l)"
  label_manifest="${label_root}/worker_manifest_000-of-001.json"
  [[ "${source_manifests}" -eq 3 ]] \
    && [[ -f "${label_manifest}" ]] \
    && grep -q '"worker_complete": true' "${label_manifest}" \
    && grep -q '"failed_chunk_count": 0' "${label_manifest}"
}

incumbents_finished() {
  local pid
  for pid in ${wait_pids}; do
    if kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

gpus_are_free() {
  local gpu
  for gpu in "${gpus[@]}"; do
    if [[ -n "$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" ]]; then
      return 1
    fi
  done
  return 0
}

while ! cache_complete || ! incumbents_finished || ! gpus_are_free; do
  printf 'SCORER_FULL_WAIT utc=%s source_manifests=%s label_manifest=%s incumbents=%s\n' \
    "$(date -u +%FT%TZ)" \
    "$(find "${source_root}" -mindepth 2 -maxdepth 2 -name manifest.json -type f 2>/dev/null | wc -l)" \
    "$([[ -f "${label_root}/worker_manifest_000-of-001.json" ]] && echo present || echo absent)" \
    "${wait_pids:-none}"
  sleep "${poll_seconds}"
done

names=(
  full_local_hybrid_top16_safetyw1_seed2_v1
  full_local_residual_top16_safetyw1_seed2_v1
  full_local_hybrid_top8_safetyw1_seed2_v1
  full_local_residual_top8_safetyw1_seed2_v1
  full_local_hybrid_top16_safetyw1_seed0_v1
  full_local_residual_top16_safetyw1_seed0_v1
  full_local_hybrid_top16_safetyw1_seed1_v1
  full_local_residual_top16_safetyw1_seed1_v1
)
modes=(local local local local local local local local)
score_modes=(hybrid residual hybrid residual hybrid residual hybrid residual)
top_ks=(16 16 8 8 16 16 16 16)
safety_weights=(1 1 1 1 1 1 1 1)
seeds=(2 2 2 2 0 0 1 1)
pids=()
pid_names=()

wait_for_wave() {
  local index status
  for index in "${!pids[@]}"; do
    if wait "${pids[index]}"; then
      printf 'SCORER_FULL_DONE name=%s pid=%s\n' \
        "${pid_names[index]}" "${pids[index]}"
    else
      status="$?"
      printf 'SCORER_FULL_FAILED name=%s pid=%s status=%s\n' \
        "${pid_names[index]}" "${pids[index]}" "${status}" >&2
      failed=1
    fi
  done
  pids=()
  pid_names=()
}

failed=0

for index in "${!names[@]}"; do
  gpu_index=$((index % ${#gpus[@]}))
  gpu="${gpus[gpu_index]}"
  output_dir="${run_root}/${names[index]}"
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to reuse output directory: ${output_dir}" >&2
    exit 3
  fi
  mkdir -p "${output_dir}"
  command=(
    "${python_bin}"
    "${repo_root}/local_stage2/train_public_base_residual_scorer.py"
    --repo-root "${repo_root}"
    --source-root "${source_root}"
    --label-root "${label_root}"
    --base-checkpoint "${base_checkpoint}"
    --output-dir "${output_dir}"
    --mode "${modes[index]}"
    --score-mode "${score_modes[index]}"
    --top-k "${top_ks[index]}"
    --safety-negative-weight "${safety_weights[index]}"
    --seed "${seeds[index]}"
    --epochs 20
    --batch-size 128
    --eval-batch-size 256
    --require-complete-cache
  )
  printf '%q ' "${command[@]}" > "${output_dir}/COMMAND.sh"
  printf '\n' >> "${output_dir}/COMMAND.sh"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${command[@]}" \
    > "${output_dir}/train.log" 2>&1 &
  launched_pid="$!"
  pids+=("${launched_pid}")
  pid_names+=("${names[index]}")
  printf 'SCORER_FULL_LAUNCH name=%s gpu=%s pid=%s\n' \
    "${names[index]}" "${gpu}" "${launched_pid}"
  if [[ "${#pids[@]}" -eq "${#gpus[@]}" ]]; then
    wait_for_wave
  fi
done
if [[ "${#pids[@]}" -gt 0 ]]; then
  wait_for_wave
fi
exit "${failed}"
