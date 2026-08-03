#!/usr/bin/env bash
# Extract the complete OpenScene trainval camera stream needed by DriveDreamer.
#
# NAVSIM's regular sensor_blobs package is intentionally sparse: it contains
# enough history for planning, but not every future frame used by the 9-frame
# world-model target.  The local NAVSIM mirror also contains the 200 complete
# OpenScene camera archives.  Extract them to a separate directory so the
# shared, sparse dataset is never overwritten.

set -euo pipefail

archive_root="${ARCHIVE_ROOT:-/mnt/data_and_weight/Public_Space/navsim/trainval_all}"
target_root="${TARGET_ROOT:-/mnt/data_and_weight/Public_Space/navsim/trainval_full_sensor_blobs}"
workers="${WORKERS:-6}"
marker_root="${target_root}/.camera_extract_done"

mkdir -p "${target_root}" "${marker_root}"

export target_root marker_root
find "${archive_root}" -maxdepth 1 -type f \
  -name 'openscene_sensor_trainval_camera_*.tgz' -print0 \
  | sort -z \
  | xargs -0 -r -n 1 -P "${workers}" bash -c '
      archive="$1"
      archive_name="${archive##*/}"
      marker="${marker_root}/${archive_name}.done"
      if [[ -f "${marker}" ]]; then
        printf "[SKIP] %s\n" "${archive_name}"
        exit 0
      fi

      printf "[START] %s\n" "${archive_name}"
      tar -xzf "${archive}" \
        -C "${target_root}" \
        --strip-components=2 \
        --overwrite
      touch "${marker}"
      printf "[OK] %s\n" "${archive_name}"
    ' _

expected=200
actual="$(find "${marker_root}" -maxdepth 1 -type f -name '*.done' | wc -l)"
if [[ "${actual}" -ne "${expected}" ]]; then
  printf "Expected %d completed archives, found %d\n" "${expected}" "${actual}" >&2
  exit 1
fi

printf "[OK] all %d camera archives extracted under %s/trainval\n" \
  "${actual}" "${target_root}"
