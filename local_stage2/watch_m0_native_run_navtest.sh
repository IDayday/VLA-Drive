#!/usr/bin/env bash
set -euo pipefail

# Promote one completed M0-native scorer run by its held-out physical-log CI,
# then serialize strict full-Navtest inference on one GPU via flock.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 {independent|residual} RUN_DIR GPU [CAMPAIGN_NAME]" >&2
  exit 2
fi

kind="$1"
run_dir="$2"
gpu="$3"
campaign_name="${4:-$(basename "${run_dir}")}"
if [[ "${kind}" != "independent" && "${kind}" != "residual" ]]; then
  echo "Unknown scorer run kind: ${kind}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/mnt/project/DriveVLA-M0-m0-scorer-representation}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/project/DriveVLA-M0-env/bin/python}"
RUN_ROOT="${RUN_ROOT:-/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93}"
PRIVATE_NAVTEST_ROOT="${PRIVATE_NAVTEST_ROOT:-${RUN_ROOT}/m0_native_multiview_navtest_pool2_tiles4_v1_2shard}"
FEATURE_CACHE="${FEATURE_CACHE:-${RUN_ROOT}/../ke_candidate_audit/public_base_navtest_scorer_features_full_fp32_v2/proposal_predictions.pkl}"
CANDIDATE_MATRIX="${CANDIDATE_MATRIX:-${RUN_ROOT}/../ke_candidate_audit/public_base_navtest_all_candidate_factors_fp32_v1/candidate_scores.npz}"
PUBLIC_AUDIT="${PUBLIC_AUDIT:-${RUN_ROOT}/../ke_candidate_audit/public_base_navtest_all_candidate_factors_fp32_v1}"
PUBLIC_REFERENCE="${PUBLIC_REFERENCE:-/mnt/project/DriveVLA-M0-runs/ke/public_base_navtest_full/08.27_09.46/2026.08.27.10.03.42.csv}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/mnt/project/DriveVLA-M0-modelscope/best-epoch_26-step_174312.server_merged.ckpt}"
PROMOTION_ROOT="${PROMOTION_ROOT:-${RUN_ROOT}/m0_native_promotion_manifests_v1}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-${RUN_ROOT}/checkpoint_snapshots}"
COMPARISON_ROOT="${COMPARISON_ROOT:-${RUN_ROOT}/comparisons}"
LOCK_PATH="${LOCK_PATH:-/tmp/m0_native_navtest_gpu.lock}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-1440}"

mkdir -p "${PROMOTION_ROOT}" "${SNAPSHOT_ROOT}" "${COMPARISON_ROOT}"
manifest="${PROMOTION_ROOT}/${campaign_name}.json"

run_complete() {
  local summary="${run_dir}/training_summary.json"
  [[ -f "${summary}" ]] || return 1
  "${PYTHON_BIN}" - "${summary}" <<'PY' >/dev/null
import json
import sys

payload = json.load(open(sys.argv[1]))
history = payload.get("history", [])
if not history:
    raise SystemExit(1)
fold = json.load(open(sys.argv[1].replace("training_summary.json", "fold_manifest.json")))
if len(history) != int(fold["args"]["epochs"]):
    raise SystemExit(1)
PY
}

navtest_cache_complete() {
  [[ "$(find "${PRIVATE_NAVTEST_ROOT}" -name manifest.json -type f 2>/dev/null | wc -l)" -eq 2 ]]
}

for ((attempt = 0; attempt < WAIT_ATTEMPTS; attempt++)); do
  if run_complete && navtest_cache_complete; then
    break
  fi
  sleep 30
done
if ! run_complete; then
  echo "Timed out before scorer run completed: ${run_dir}" >&2
  exit 1
fi
if ! navtest_cache_complete; then
  echo "Timed out before M0-native Navtest cache completed" >&2
  exit 1
fi

test ! -e "${manifest}"
promotion_args=(--output "${manifest}" --minimum-ci-lower 0)
if [[ "${kind}" == "independent" ]]; then
  promotion_args+=(--independent-run "${run_dir}")
else
  promotion_args+=(--residual-run "${run_dir}")
fi
env PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script" \
  "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/build_m0_native_promotion_manifest.py" \
  "${promotion_args[@]}"

mapfile -t promoted_rows < <(
  "${PYTHON_BIN}" - "${manifest}" <<'PY'
import json
import sys

for record in json.load(open(sys.argv[1]))["promoted"]:
    print("\t".join((
        record["name"],
        record["architecture"],
        record["score_mode"],
        record["artifact"],
        record["artifact_sha256"],
    )))
PY
)
if [[ "${#promoted_rows[@]}" -eq 0 ]]; then
  echo "No artifact passed the strict held-out-log promotion gate: ${run_dir}"
  exit 0
fi

exec 9>"${LOCK_PATH}"
flock -x 9
export CUDA_VISIBLE_DEVICES="${gpu}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navsim/planning/script${PYTHONPATH:+:${PYTHONPATH}}"

for row in "${promoted_rows[@]}"; do
  IFS=$'\t' read -r name architecture score_mode artifact artifact_sha <<<"${row}"
  audit_dir="${RUN_ROOT}/${name}_navtest_v1"
  comparison_dir="${COMPARISON_ROOT}/${name}_vs_public_v1"
  if [[ "${architecture}" == "IndependentProposalRanker" ]]; then
    snapshot="${SNAPSHOT_ROOT}/${name}_all64_v1.pt"
    if [[ ! -f "${snapshot}" ]]; then
      "${PYTHON_BIN}" "${REPO_ROOT}/local_stage2/package_independent_shortlist_scorer.py" \
        --ranker-artifact "${artifact}" \
        --base-checkpoint "${BASE_CHECKPOINT}" \
        --shortlist-size 64 \
        --score-mode "${score_mode}" \
        --output "${snapshot}"
    fi
    "${PYTHON_BIN}" - "${snapshot}" "${artifact_sha}" <<'PY'
import hashlib
import sys
import torch

snapshot, expected = sys.argv[1:]
payload = torch.load(snapshot, map_location="cpu", weights_only=False)
if payload["source_ranker_artifact_sha256"] != expected:
    raise SystemExit("snapshot/source artifact SHA256 mismatch")
PY
    evaluated_artifact="${snapshot}"
    evaluator="${REPO_ROOT}/local_stage2/evaluate_independent_shortlist_navtest_cache.py"
  elif [[ "${architecture}" == "M0PrivateResidualRanker" ]]; then
    evaluated_artifact="${artifact}"
    evaluator="${REPO_ROOT}/local_stage2/evaluate_m0_private_residual_navtest_cache.py"
  else
    echo "Unsupported promoted architecture: ${architecture}" >&2
    exit 1
  fi

  if [[ ! -f "${audit_dir}/summary.json" ]]; then
    if [[ -e "${audit_dir}" ]]; then
      echo "Refusing incomplete/nonempty Navtest output: ${audit_dir}" >&2
      exit 1
    fi
    "${PYTHON_BIN}" "${evaluator}" \
      --artifact "${evaluated_artifact}" \
      --feature-cache "${FEATURE_CACHE}" \
      --private-observation-root "${PRIVATE_NAVTEST_ROOT}" \
      --candidate-matrix "${CANDIDATE_MATRIX}" \
      --public-audit-dir "${PUBLIC_AUDIT}" \
      --output-dir "${audit_dir}" \
      --device cuda \
      --batch-size 32 \
      --bootstrap-replicates 10000
  fi

  /root/.codex/skills/navsim-scorer-evaluation/scripts/validate_audit.sh \
    "${audit_dir}"
  if [[ ! -f "${comparison_dir}/comparison.json" ]]; then
    if [[ -e "${comparison_dir}" ]]; then
      echo "Refusing incomplete/nonempty comparison output: ${comparison_dir}" >&2
      exit 1
    fi
    /root/.codex/skills/navsim-scorer-evaluation/scripts/compare_audits.sh \
      "${audit_dir}" "${PUBLIC_AUDIT}" "${comparison_dir}" \
      "${PUBLIC_REFERENCE}" "${name}" public_open_weight
  fi
done

echo "Strict M0-native promoted Navtest run completed: ${campaign_name}"
