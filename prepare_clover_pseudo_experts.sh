#!/usr/bin/env bash
# Download and validate the official CLOVER Stage-1 pseudo-expert package.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

official_file_id="1JUKc9hQ2YH7Ck7ghjZm65OCavWVg6Po_"
official_size_bytes=3964573635
default_output="$project_root/navsim_exp/assets/clover_stage1_pseudo_experts/CLOVER/dataset_decoupled_v2_clean.pkl"
case "${CLOVER_PSEUDO_EXPERT_PKL:-}" in
  ""|/absolute/path/to/pseudo_experts.pkl|/absolute/path/to/official/pseudo_experts.pkl|/absolute/path/to/official-clover-pseudo-experts.pkl|/PATH/TO/pseudo_expert.pkl)
    output="$default_output"
    ;;
  *)
    output="$CLOVER_PSEUDO_EXPERT_PKL"
    ;;
esac
dry_run=0
validate_only=0
max_retries="${CLOVER_ASSET_MAX_RETRIES:-100}"
retry_delay_seconds="${CLOVER_ASSET_RETRY_DELAY_SECONDS:-15}"

while (( $# )); do
  case "$1" in
    --output)
      if (( $# < 2 )); then
        echo "[clover-assets] --output requires a path" >&2
        exit 2
      fi
      output="$2"
      shift
      ;;
    --dry-run) dry_run=1 ;;
    --validate-only) validate_only=1 ;;
    --help|-h)
      echo "Usage: $0 [--output PATH] [--dry-run] [--validate-only]"
      exit 0
      ;;
    *)
      echo "[clover-assets] unsupported argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

output="$(realpath -m -- "$output")"
source_url="https://drive.google.com/uc?id=$official_file_id"
for integer_name in max_retries retry_delay_seconds; do
  integer_value="${!integer_name}"
  if ! [[ "$integer_value" =~ ^[0-9]+$ ]]; then
    echo "[clover-assets] $integer_name must be an unsigned integer: $integer_value" >&2
    exit 2
  fi
done
if (( max_retries < 1 )); then
  echo "[clover-assets] max_retries must be positive" >&2
  exit 2
fi

print_command() {
  printf '[clover-assets] command:'
  printf ' %q' "$@"
  printf '\n'
}

validate_asset() {
  local actual_size
  if [[ ! -f "$output" ]]; then
    echo "[clover-assets] official package is missing: $output" >&2
    return 2
  fi
  actual_size="$(stat -c '%s' "$output")"
  if [[ "$actual_size" != "$official_size_bytes" ]]; then
    echo "[clover-assets] package size mismatch actual=$actual_size expected=$official_size_bytes path=$output" >&2
    return 2
  fi
  echo "[clover-assets] package size valid bytes=$actual_size"
}

if (( dry_run )); then
  echo "[clover-assets] dry_run=1 writes=0 imports=0"
  echo "[clover-assets] source_file_id=$official_file_id"
  echo "[clover-assets] expected_size_bytes=$official_size_bytes"
  echo "[clover-assets] output=$output"
  echo "[clover-assets] retries=$max_retries retry_delay_seconds=$retry_delay_seconds"
  if [[ -f "$output" ]]; then
    echo "[clover-assets] state=COMPLETE_OR_INVALID actual_size_bytes=$(stat -c '%s' "$output")"
  else
    partials=("$output"*.part)
    if [[ -e "${partials[0]}" ]]; then
      echo "[clover-assets] state=PARTIAL partial=${partials[0]} bytes=$(stat -c '%s' "${partials[0]}")"
    else
      echo "[clover-assets] state=MISSING"
    fi
  fi
  print_command gdown --continue --fuzzy --output "$output" "$source_url"
  exit 0
fi

if (( validate_only )); then
  validate_asset
  sha256sum "$output"
  exit 0
fi

if [[ ! -f "$output" ]]; then
  if ! command -v gdown >/dev/null 2>&1; then
    echo "[clover-assets] gdown is required in the asset-preparation environment; do not install packages inside the formal 16-PPU training job" >&2
    exit 2
  fi
  if ! command -v flock >/dev/null 2>&1; then
    echo "[clover-assets] flock is required to prevent concurrent writers" >&2
    exit 2
  fi
  output_parent="$(dirname -- "$output")"
  mkdir -p "$output_parent"
  exec 9>"$output.download.lock"
  if ! flock -n 9; then
    echo "[clover-assets] another process is already preparing this asset: $output" >&2
    exit 3
  fi
  # The file may have completed while this process waited for the lock.
  if [[ -f "$output" ]]; then
    validate_asset
  else
    shopt -s nullglob
    partials=("$output"*.part)
    shopt -u nullglob
    if (( ${#partials[@]} > 1 )); then
      echo "[clover-assets] multiple partial downloads exist; refusing an ambiguous resume:" >&2
      printf '  %s\n' "${partials[@]}" >&2
      exit 2
    fi
    echo "[clover-assets] downloading the 3.96-GB official package; an interrupted command is safe to rerun"
    print_command gdown --continue --fuzzy --output "$output" "$source_url"
    download_complete=0
    for (( attempt = 1; attempt <= max_retries; attempt++ )); do
      echo "[clover-assets] download attempt=$attempt/$max_retries"
      if gdown --continue --fuzzy --output "$output" "$source_url"; then
        download_complete=1
        break
      fi
      if (( attempt < max_retries )); then
        echo "[clover-assets] transient download failure; retaining partial file and retrying in ${retry_delay_seconds}s" >&2
        sleep "$retry_delay_seconds"
      fi
    done
    if (( ! download_complete )); then
      echo "[clover-assets] download exhausted $max_retries attempts; partial file remains resumable" >&2
      exit 2
    fi
  fi
fi

validate_asset
sha256="$(sha256sum "$output" | awk '{print $1}')"
manifest="$output.manifest.json"
python - "$manifest" "$output" "$official_file_id" "$official_size_bytes" "$sha256" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

manifest, output, file_id, size, sha256 = sys.argv[1:]
payload = {
    "schema_version": 1,
    "source": "WilliamXuanYu/CLOVER official Stage-1 pseudo-expert package",
    "google_drive_file_id": file_id,
    "path": str(Path(output).resolve()),
    "size_bytes": int(size),
    "sha256": sha256,
    "validated_at": datetime.now(timezone.utc).isoformat(),
}
destination = Path(manifest)
fd, temporary = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY

echo "[clover-assets] complete path=$output sha256=$sha256"
