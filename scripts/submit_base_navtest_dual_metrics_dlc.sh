#!/usr/bin/env bash
# Submit one single-worker/single-PPU full-navtest PDMS + EPDMS evaluation.

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
source "$project_root/load_env.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/submit_base_navtest_dual_metrics_dlc.sh [options]

  --run-id ID                Required stable experiment identity
  --project-root PATH        Shared-FS project path visible inside DLC
  --output-root PATH         Exact shared-FS run directory
  --workspace-id ID          PAI workspace (default: PAI_WORKSPACE_ID)
  --region REGION            DLC region (default: current DSW region)
  --endpoint HOST            DLC API endpoint
  --resource-id ID           Optional dedicated DLC resource group
  --worker-image IMAGE       Existing audited PPU image
  --worker-spec SPEC         Optional DLC machine spec
  --worker-gpu-type TYPE     Default: PPU (ignored with --worker-spec)
  --worker-cpu N             Default: 32
  --worker-memory SIZE       Default: 256Gi
  --worker-shared-memory S   Default: 64Gi
  --preflight-only           Submit only the non-writing runtime preflight
  --resume                   Resume identity-checked artifacts
  --overwrite                Archive exact prior run and restart
  --dry-run                  Print dlc submit command; do not submit
  -h, --help                 Show this help

Authentication is read from the local `dlc` CLI config. No token or key is
placed in the job command. The image is used as-is: no dependency installation
and no torch/triton/flash-attn mutation occurs in the job.
EOF
}

require_value() {
  if (( $# < 2 )) || [[ -z "$2" ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

run_id="${RUN_ID:-}"
dlc_project_root="${DLC_PROJECT_ROOT:-$project_root}"
output_root="${DLC_OUTPUT_ROOT:-}"
workspace_id="${DLC_WORKSPACE_ID:-${PAI_WORKSPACE_ID:-}}"
dlc_region="${DLC_REGION:-${REGION:-cn-wulanchabu}}"
dlc_endpoint="${DLC_ENDPOINT:-pai-dlc.${dlc_region}.aliyuncs.com}"
resource_id="${DLC_RESOURCE_ID:-}"
worker_image="${DLC_WORKER_IMAGE:-${DOCKER_IMAGE_URL:-}}"
worker_spec="${DLC_WORKER_SPEC:-}"
worker_gpu_type="${DLC_WORKER_GPU_TYPE:-PPU}"
worker_cpu="${DLC_WORKER_CPU:-32}"
worker_memory="${DLC_WORKER_MEMORY:-256Gi}"
worker_shared_memory="${DLC_WORKER_SHARED_MEMORY:-64Gi}"
preflight_only=0
resume=0
overwrite=0
dry_run=0

while (( $# > 0 )); do
  case "$1" in
    --run-id) require_value "$@"; run_id="$2"; shift 2 ;;
    --run-id=*) run_id="${1#*=}"; shift ;;
    --project-root) require_value "$@"; dlc_project_root="$2"; shift 2 ;;
    --project-root=*) dlc_project_root="${1#*=}"; shift ;;
    --output-root) require_value "$@"; output_root="$2"; shift 2 ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --workspace-id) require_value "$@"; workspace_id="$2"; shift 2 ;;
    --workspace-id=*) workspace_id="${1#*=}"; shift ;;
    --region) require_value "$@"; dlc_region="$2"; shift 2 ;;
    --region=*) dlc_region="${1#*=}"; shift ;;
    --endpoint) require_value "$@"; dlc_endpoint="$2"; shift 2 ;;
    --endpoint=*) dlc_endpoint="${1#*=}"; shift ;;
    --resource-id) require_value "$@"; resource_id="$2"; shift 2 ;;
    --resource-id=*) resource_id="${1#*=}"; shift ;;
    --worker-image) require_value "$@"; worker_image="$2"; shift 2 ;;
    --worker-image=*) worker_image="${1#*=}"; shift ;;
    --worker-spec) require_value "$@"; worker_spec="$2"; shift 2 ;;
    --worker-spec=*) worker_spec="${1#*=}"; shift ;;
    --worker-gpu-type) require_value "$@"; worker_gpu_type="$2"; shift 2 ;;
    --worker-gpu-type=*) worker_gpu_type="${1#*=}"; shift ;;
    --worker-cpu) require_value "$@"; worker_cpu="$2"; shift 2 ;;
    --worker-cpu=*) worker_cpu="${1#*=}"; shift ;;
    --worker-memory) require_value "$@"; worker_memory="$2"; shift 2 ;;
    --worker-memory=*) worker_memory="${1#*=}"; shift ;;
    --worker-shared-memory) require_value "$@"; worker_shared_memory="$2"; shift 2 ;;
    --worker-shared-memory=*) worker_shared_memory="${1#*=}"; shift ;;
    --preflight-only) preflight_only=1; shift ;;
    --resume) resume=1; shift ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$run_id" ]]; then
  echo "--run-id is required; use a stable name so retries can resume safely" >&2
  exit 2
fi
if ! [[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]]; then
  echo "Invalid --run-id: $run_id" >&2
  exit 2
fi
if [[ "$resume" == "1" && "$overwrite" == "1" ]]; then
  echo "--resume and --overwrite are mutually exclusive" >&2
  exit 2
fi
if [[ -z "$workspace_id" || -z "$worker_image" ]]; then
  echo "PAI workspace ID and an existing PPU worker image are required" >&2
  exit 2
fi
if [[ ! -d "$dlc_project_root" ]]; then
  echo "DLC project root is not visible on the shared filesystem: $dlc_project_root" >&2
  exit 2
fi
if [[ ! -f "$dlc_project_root/scripts/run_base_navtest_dual_metrics_dlc.sh" ]]; then
  echo "DLC evaluator entrypoint is missing below: $dlc_project_root" >&2
  exit 2
fi
if [[ -z "$output_root" ]]; then
  output_base="${DRIVEVLA_DLC_EVAL_ROOT:-$NAVSIM_EXP_ROOT/dlc_navtest_dual_metrics}"
  output_root="$output_base/$run_id"
fi

entry_args=(
  bash "$dlc_project_root/scripts/run_base_navtest_dual_metrics_dlc.sh"
  --run-id "$run_id"
  --output-root "$output_root"
)
if [[ "$preflight_only" == "1" ]]; then entry_args+=(--preflight-only); fi
if [[ "$resume" == "1" ]]; then entry_args+=(--resume); fi
if [[ "$overwrite" == "1" ]]; then entry_args+=(--overwrite); fi

printf -v quoted_project '%q' "$dlc_project_root"
job_command="cd $quoted_project && exec"
for argument in "${entry_args[@]}"; do
  printf -v quoted_argument '%q' "$argument"
  job_command+=" $quoted_argument"
done

job_suffix="$(printf '%s' "$run_id" | tr '[:upper:]_.' '[:lower:]--' | cut -c1-38)"
job_name="drivevla-navtest-dual-$job_suffix"
submit_command=(
  dlc submit pytorchjob
  --region "$dlc_region"
  --endpoint "$dlc_endpoint"
  --name "$job_name"
  --workspace_id "$workspace_id"
  --workers 1
  --worker_image "$worker_image"
  --command "$job_command"
)
if [[ -n "$resource_id" ]]; then
  submit_command+=(--resource_id "$resource_id")
fi
if [[ -n "$worker_spec" ]]; then
  submit_command+=(--worker_spec "$worker_spec")
else
  submit_command+=(
    --worker_cpu "$worker_cpu"
    --worker_gpu 1
    --worker_gpu_type "$worker_gpu_type"
    --worker_memory "$worker_memory"
    --worker_shared_memory "$worker_shared_memory"
  )
fi

printf 'DLC command:'
printf ' %q' "${submit_command[@]}"
printf '\n'
printf 'Run output: %s\n' "$output_root"
printf '%s\n' "Topology: workers=1, PPU=1; installed flash-attn remains unchanged"

if [[ "$dry_run" == "1" ]]; then
  echo "DRY RUN: no DLC job was submitted."
  exit 0
fi

if ! command -v dlc >/dev/null 2>&1; then
  echo "dlc CLI is unavailable" >&2
  exit 2
fi
"${submit_command[@]}"
