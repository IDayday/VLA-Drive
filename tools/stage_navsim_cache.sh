#!/usr/bin/env bash
# Build a node-local cache overlay. Requested components selected for staging
# are copied atomically to tmpfs; the rest remain symlinks to durable CPFS.

set -Eeuo pipefail

source_root="${1:?usage: stage_navsim_cache.sh SOURCE_ROOT DEST_ROOT COMPONENTS [STAGE_COMPONENTS]}"
destination_root="${2:?missing destination root}"
requested_components="${3:?missing requested components}"
stage_components="${4:-wan,ppd}"
stage_mode="${NAVSIM_STAGE_CACHE_TO_RAM:-auto}"
reserve_gb="${NAVSIM_RAM_RESERVE_GB:-400}"
copy_workers="${NAVSIM_RAM_COPY_WORKERS:-8}"

case "$stage_mode" in
  0|off|false)
    printf '%s\n' "$(readlink -m "$source_root")"
    exit 0
    ;;
  auto|1|required|on|true) ;;
  *)
    echo "[cache-stage] invalid NAVSIM_STAGE_CACHE_TO_RAM=$stage_mode" >&2
    exit 2
    ;;
esac
if ! [[ "$reserve_gb" =~ ^[0-9]+$ ]]; then
  echo "[cache-stage] NAVSIM_RAM_RESERVE_GB must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$copy_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "[cache-stage] NAVSIM_RAM_COPY_WORKERS must be a positive integer" >&2
  exit 2
fi

source_root="$(readlink -m "$source_root")"
destination_root="$(readlink -m "$destination_root")"
case "$destination_root" in
  /dev/shm/drivedreamer-navsim-cache/*) ;;
  *)
    echo "[cache-stage] refusing destination outside /dev/shm/drivedreamer-navsim-cache: $destination_root" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a requested <<< "$requested_components"
IFS=',' read -r -a staged <<< "$stage_components"
declare -A should_stage=()
declare -A manifest_hashes=()
for component in "${staged[@]}"; do
  component="${component//[[:space:]]/}"
  case "$component" in qwen|wan|ppd) should_stage["$component"]=1 ;; "") ;; *) echo "[cache-stage] invalid stage component: $component" >&2; exit 2 ;; esac
done

mkdir -p "$destination_root"
required_bytes=0
for component in "${requested[@]}"; do
  component="${component//[[:space:]]/}"
  case "$component" in qwen|wan|ppd) ;; *) echo "[cache-stage] invalid requested component: $component" >&2; exit 2 ;; esac
  if [ ! -f "$source_root/$component/manifest.json" ]; then
    echo "[cache-stage] missing source manifest: $source_root/$component/manifest.json" >&2
    exit 2
  fi
  manifest_hashes["$component"]="$(sha256sum "$source_root/$component/manifest.json" | awk '{print $1}')"
  if [ "${should_stage[$component]:-0}" = "1" ]; then
    target="$destination_root/$component"
    marker="$target/.stage-source-manifest.sha256"
    if [ -f "$marker" ] && [ "$(tr -d '[:space:]' < "$marker")" = "${manifest_hashes[$component]}" ]; then
      continue
    fi
    # Partial or mismatched targets are disposable node-local state. Removing
    # them before df avoids double-counting stale bytes on a DLC restart.
    rm -rf -- "$destination_root/.${component}.partial" "$target"
    component_bytes="$(du -s --block-size=1 "$source_root/$component" | awk '{print $1}')"
    required_bytes=$((required_bytes + component_bytes))
  fi
done

available_bytes="$(df -B1 --output=avail "$(dirname "$destination_root")" | tail -n 1 | tr -d ' ')"
reserve_bytes=$((reserve_gb * 1024 * 1024 * 1024))
if (( required_bytes + reserve_bytes > available_bytes )); then
  echo "[cache-stage] tmpfs capacity is insufficient: required=$required_bytes reserve=$reserve_bytes available=$available_bytes" >&2
  if [ "$stage_mode" = "auto" ]; then
    echo "[cache-stage] auto mode: falling back to shared CPFS cache" >&2
    printf '%s\n' "$source_root"
    exit 0
  fi
  exit 2
fi

echo "[cache-stage] source=$source_root destination=$destination_root required_bytes=$required_bytes available_bytes=$available_bytes reserve_gb=$reserve_gb copy_workers=$copy_workers" >&2

for component in "${requested[@]}"; do
  component="${component//[[:space:]]/}"
  target="$destination_root/$component"
  manifest_hash="${manifest_hashes[$component]}"
  if [ "${should_stage[$component]:-0}" = "1" ]; then
    marker="$target/.stage-source-manifest.sha256"
    if [ -f "$marker" ] && [ "$(tr -d '[:space:]' < "$marker")" = "$manifest_hash" ]; then
      echo "[cache-stage] reuse component=$component target=$target" >&2
      continue
    fi
    partial="$destination_root/.${component}.partial"
    rm -rf -- "$partial" "$target"
    echo "[cache-stage] copying component=$component to tmpfs workers=$copy_workers" >&2
    mkdir -p "$partial"
    # Each component contains independent rank LMDB directories. Multiple CPFS
    # read streams fill tmpfs substantially faster than one recursive cp while
    # preserving each file's metadata and sparse layout.
    find "$source_root/$component" -mindepth 1 -maxdepth 1 -print0 \
      | xargs -0 -r -P "$copy_workers" cp -a --sparse=always -t "$partial"
    printf '%s\n' "$manifest_hash" > "$partial/.stage-source-manifest.sha256"
    mv "$partial" "$target"
    echo "[cache-stage] complete component=$component bytes=$(du -s --block-size=1 "$target" | awk '{print $1}')" >&2
  else
    if [ -e "$target" ] || [ -L "$target" ]; then
      rm -rf -- "$target"
    fi
    ln -s "$source_root/$component" "$target"
    echo "[cache-stage] shared component=$component target=$target" >&2
  fi
done

printf '%s\n' "$destination_root"
