#!/usr/bin/env bash
# Build a node-local NAVSIM data overlay containing only the raw training
# metadata needed when PPD features are cached. Camera images stay on CPFS.

set -Eeuo pipefail

source_data_root="${1:?usage: stage_navsim_metadata.sh SOURCE_DATA_ROOT DEST_DATA_ROOT DATALIST}"
destination_root="${2:?missing destination root}"
datalist_path="${3:?missing datalist path}"
stage_mode="${NAVSIM_STAGE_METADATA_TO_RAM:-auto}"
reserve_gb="${NAVSIM_RAM_RESERVE_GB:-400}"
copy_workers="${NAVSIM_METADATA_COPY_WORKERS:-16}"

case "$stage_mode" in
  0|off|false)
    printf '%s\n' "$(readlink -m "$source_data_root")"
    exit 0
    ;;
  auto|1|required|on|true) ;;
  *) echo "[metadata-stage] invalid NAVSIM_STAGE_METADATA_TO_RAM=$stage_mode" >&2; exit 2 ;;
esac
if ! [[ "$reserve_gb" =~ ^[0-9]+$ ]]; then
  echo "[metadata-stage] NAVSIM_RAM_RESERVE_GB must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$copy_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "[metadata-stage] NAVSIM_METADATA_COPY_WORKERS must be a positive integer" >&2
  exit 2
fi

source_data_root="$(readlink -m "$source_data_root")"
destination_root="$(readlink -m "$destination_root")"
datalist_path="$(readlink -m "$datalist_path")"
case "$destination_root" in
  /dev/shm/drivedreamer-navsim-data/*) ;;
  *)
    echo "[metadata-stage] refusing destination outside /dev/shm/drivedreamer-navsim-data: $destination_root" >&2
    exit 2
    ;;
esac

source_train_dir="$source_data_root/meta/train"
if [ ! -d "$source_train_dir" ] || [ ! -f "$datalist_path" ]; then
  echo "[metadata-stage] source metadata or datalist is missing" >&2
  exit 2
fi

expected_samples="$(python - "$datalist_path" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as stream:
    print(len(json.load(stream)))
PY
)"
datalist_hash="$(sha256sum "$datalist_path" | awk '{print $1}')"
source_identity="$(printf '%s\n%s\n' "$source_data_root" "$datalist_hash" | sha256sum | awk '{print $1}')"
marker="$destination_root/.stage-source-datalist.sha256"
target_train_dir="$destination_root/meta/train"

if [ -f "$marker" ] && [ "$(tr -d '[:space:]' < "$marker")" = "$source_identity" ]; then
  actual_samples="$(find "$target_train_dir" -maxdepth 1 -type f -name '*.pkl' ! -name '*-depth.pkl' -printf '.' | wc -c)"
  if [ "$actual_samples" = "$expected_samples" ]; then
    echo "[metadata-stage] reuse destination=$destination_root samples=$actual_samples" >&2
    printf '%s\n' "$destination_root"
    exit 0
  fi
fi

partial="${destination_root}.partial"
rm -rf -- "$partial" "$destination_root"
read -r source_samples required_bytes < <(
  find "$source_train_dir" -maxdepth 1 -type f -name '*.pkl' ! -name '*-depth.pkl' -printf '%s\n' \
    | awk '{count += 1; total += $1} END {printf "%d %.0f\n", count, total}'
)
if [ "$source_samples" != "$expected_samples" ]; then
  echo "[metadata-stage] raw metadata count mismatch: source=$source_samples datalist=$expected_samples" >&2
  exit 2
fi
mkdir -p "$(dirname "$destination_root")"
available_bytes="$(df -B1 --output=avail "$(dirname "$destination_root")" | tail -n 1 | tr -d ' ')"
reserve_bytes=$((reserve_gb * 1024 * 1024 * 1024))
if (( required_bytes + reserve_bytes > available_bytes )); then
  echo "[metadata-stage] tmpfs capacity is insufficient: required=$required_bytes reserve=$reserve_bytes available=$available_bytes" >&2
  if [ "$stage_mode" = "auto" ]; then
    echo "[metadata-stage] auto mode: falling back to shared CPFS metadata" >&2
    printf '%s\n' "$source_data_root"
    exit 0
  fi
  exit 2
fi

echo "[metadata-stage] source=$source_train_dir destination=$destination_root samples=$source_samples bytes=$required_bytes copy_workers=$copy_workers" >&2
mkdir -p "$partial/meta/train"
find "$source_train_dir" -maxdepth 1 -type f -name '*.pkl' ! -name '*-depth.pkl' -print0 \
  | xargs -0 -r -P "$copy_workers" -n 64 cp -a -t "$partial/meta/train"
if [ -d "$source_data_root/navsim_video" ]; then
  ln -s "$source_data_root/navsim_video" "$partial/navsim_video"
fi
printf '%s\n' "$source_identity" > "$partial/.stage-source-datalist.sha256"
mv "$partial" "$destination_root"
echo "[metadata-stage] complete destination=$destination_root samples=$source_samples" >&2
printf '%s\n' "$destination_root"
