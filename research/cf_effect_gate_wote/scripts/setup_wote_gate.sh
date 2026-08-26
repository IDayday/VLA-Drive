#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_common.sh"

expected_wote_sha="298957c128a91d41a1c6075bd0bb6e7e845e093f"
wote_root="${CF_GATE_WOTE_ROOT:-$cf_gate_project_root/../third_party/WoTE}"
release_root="${CF_GATE_WOTE_RELEASE_ROOT:-$cf_gate_project_root/../third_party/WoTE_release/wote}"
data_root="${CF_GATE_NAVSIM_DATA_ROOT:-}"
report_dir="${CF_GATE_REPORT_DIR:-$cf_gate_project_root/reports/cf_effect_gate_wote}"
dry_run=0
preflight_only=0
download_release=0
write_manifest=0

usage() {
  cat <<'EOF'
Usage: setup_wote_gate.sh [options]

Options:
  --wote-root PATH          External WoTE checkout.
  --release-root PATH       Directory containing the published WoTE assets.
  --data-root PATH          NAVSIM root containing maps/navsim_logs/sensor_blobs.
  --report-dir PATH         Small report output directory.
  --download-release       Download the official Google Drive release if the
                           release root does not exist.
  --write-manifest         Hash assets and atomically write ASSET_MANIFEST.json.
  --preflight-only         Read-only validation; do not apply the patch.
  --dry-run                Resolve and print the contract without imports or writes.
  -h, --help               Show this help.

Precedence: explicit CLI > CF_GATE_* environment > task-local defaults.
The script refuses a wrong WoTE commit, a dirty incompatible checkout, existing
download targets, or missing required assets. It never trains WoTE.
EOF
}

while (($#)); do
  case "$1" in
    --wote-root)
      [[ $# -ge 2 ]] || { printf 'missing value for --wote-root\n' >&2; exit 2; }
      wote_root="$2"
      shift 2
      ;;
    --release-root)
      [[ $# -ge 2 ]] || { printf 'missing value for --release-root\n' >&2; exit 2; }
      release_root="$2"
      shift 2
      ;;
    --data-root)
      [[ $# -ge 2 ]] || { printf 'missing value for --data-root\n' >&2; exit 2; }
      data_root="$2"
      shift 2
      ;;
    --report-dir)
      [[ $# -ge 2 ]] || { printf 'missing value for --report-dir\n' >&2; exit 2; }
      report_dir="$2"
      shift 2
      ;;
    --download-release)
      download_release=1
      shift
      ;;
    --write-manifest)
      write_manifest=1
      shift
      ;;
    --preflight-only)
      preflight_only=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown option: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

patch_path="$cf_gate_project_root/research/cf_effect_gate_wote/patches/0001-export-wote-intermediate-features.patch"
checkpoint_path="$release_root/epoch=29-step=19950.ckpt"
resnet_path="$release_root/resnet34.pth"
anchor_path="$release_root/extra_data/planning_vb/trajectory_anchors_256.npy"
score_path="$release_root/extra_data/planning_vb/formatted_pdm_score_256.npy"

printf '[cf-effect-gate] project_root=%s\n' "$cf_gate_project_root"
printf '[cf-effect-gate] wote_root=%s\n' "$wote_root"
printf '[cf-effect-gate] release_root=%s\n' "$release_root"
printf '[cf-effect-gate] data_root=%s\n' "${data_root:-unset}"
printf '[cf-effect-gate] report_dir=%s\n' "$report_dir"
printf '[cf-effect-gate] mode=dry_run:%s preflight_only:%s download_release:%s write_manifest:%s\n' \
  "$dry_run" "$preflight_only" "$download_release" "$write_manifest"

if ((dry_run)); then
  cf_gate_print_command git -C "$wote_root" rev-parse HEAD
  cf_gate_print_command git -C "$wote_root" apply --unidiff-zero --check "$patch_path"
  dry_preflight=(
    python -m research.cf_effect_gate_wote.src.cache_wote_features preflight
    --wote-root "$wote_root"
    --release-root "$release_root"
  )
  if [[ -n "$data_root" ]]; then
    dry_preflight+=(--data-root "$data_root")
  fi
  cf_gate_print_command "${dry_preflight[@]}"
  exit 0
fi

if ((download_release)) && [[ ! -e "$release_root" ]]; then
  command -v gdown >/dev/null 2>&1 || {
    printf '[cf-effect-gate] gdown is required for --download-release\n' >&2
    exit 3
  }
  mkdir -p "$(dirname -- "$release_root")"
  gdown --folder \
    'https://drive.google.com/drive/folders/1dIHK8nXkzhIhGCRQOpKibaizwH-7fHqs?usp=sharing' \
    --output "$(dirname -- "$release_root")/" --remaining-ok
fi

if [[ ! -e "$wote_root/.git" ]]; then
  printf '[cf-effect-gate] missing WoTE Git checkout: %s\n' "$wote_root" >&2
  exit 3
fi
cf_gate_require_file "$patch_path" 'WoTE feature-export patch' || exit 3

actual_wote_sha="$(git -C "$wote_root" rev-parse HEAD)"
if [[ "$actual_wote_sha" != "$expected_wote_sha" ]]; then
  printf '[cf-effect-gate] WoTE SHA mismatch: expected %s, got %s\n' \
    "$expected_wote_sha" "$actual_wote_sha" >&2
  exit 3
fi

patch_state="incompatible"
if git -C "$wote_root" apply --unidiff-zero --reverse --check "$patch_path" >/dev/null 2>&1; then
  patch_state="already_applied"
elif git -C "$wote_root" apply --unidiff-zero --check "$patch_path" >/dev/null 2>&1; then
  patch_state="clean"
fi
printf '[cf-effect-gate] patch_state=%s\n' "$patch_state"
if [[ "$patch_state" == "incompatible" ]]; then
  printf '[cf-effect-gate] patch cannot be safely applied or identified; refusing checkout mutation\n' >&2
  exit 3
fi

preflight_args=(
  -m research.cf_effect_gate_wote.src.cache_wote_features preflight
  --wote-root "$wote_root"
  --release-root "$release_root"
)
if [[ -n "$data_root" ]]; then
  preflight_args+=(--data-root "$data_root")
fi
if ((write_manifest)); then
  preflight_args+=(--report-dir "$report_dir" --write-reports)
fi

if ((preflight_only)); then
  cf_gate_run_python "${preflight_args[@]}"
  exit 0
fi

if [[ "$patch_state" == "clean" ]]; then
  git -C "$wote_root" apply --unidiff-zero "$patch_path"
  printf '[cf-effect-gate] applied patch to external WoTE checkout\n'
fi

cf_gate_run_python "${preflight_args[@]}"

printf '[cf-effect-gate] setup complete\n'
printf '[cf-effect-gate] checkpoint=%s\n' "$checkpoint_path"
printf '[cf-effect-gate] resnet34=%s\n' "$resnet_path"
printf '[cf-effect-gate] anchors=%s\n' "$anchor_path"
printf '[cf-effect-gate] scores=%s\n' "$score_path"
