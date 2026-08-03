#!/usr/bin/env bash
# Stream the complete NAVSIM camera archives through local temporary storage
# and keep only the three DriveDreamer MP4 targets plus first-frame stills.

set -euo pipefail

: "${DRIVEDREAMER_ROOT:?Run 'source env.sh' first}"
: "${DATA_ROOT:?Run 'source env.sh' first}"

archive_root="${ARCHIVE_ROOT:-/mnt/data_and_weight/Public_Space/navsim/trainval_all}"
archive_workers="${ARCHIVE_WORKERS:-4}"
video_workers="${VIDEO_WORKERS_PER_ARCHIVE:-2}"
encoder_preset="${VIDEO_ENCODER_PRESET:-medium}"
index_workers="${INDEX_WORKERS:-16}"
staging_root="${STAGING_ROOT:-/tmp/drivedreamer_policy_camera_stage}"
meta_dir="${DATA_ROOT}/meta/train"
video_dir="${DATA_ROOT}/navsim_video/train"
index_path="${DATA_ROOT}/meta/train_video_log_index.json"
marker_dir="${video_dir}/.camera_archive_done"
datalist="${DRIVEDREAMER_ROOT}/train_meta.json"

if [[ ! -s "$index_path" ]]; then
    python tools/generate_videos_from_camera_archive.py \
        --build-index \
        --datalist "$datalist" \
        --meta-dir "$meta_dir" \
        --video-dir "$video_dir" \
        --index-path "$index_path" \
        --marker-dir "$marker_dir" \
        --staging-root "$staging_root" \
        --index-workers "$index_workers"
else
    printf "[SKIP] video log index already exists: %s\n" "$index_path"
fi

export DRIVEDREAMER_ROOT datalist meta_dir video_dir index_path marker_dir
export staging_root video_workers encoder_preset

find "$archive_root" -maxdepth 1 -type f \
    -name 'openscene_sensor_trainval_camera_*.tgz' -print0 \
    | sort -z \
    | xargs -0 -r -n 1 -P "$archive_workers" bash -c '
        python "$DRIVEDREAMER_ROOT/tools/generate_videos_from_camera_archive.py" \
            --archive "$1" \
            --datalist "$datalist" \
            --meta-dir "$meta_dir" \
            --video-dir "$video_dir" \
            --index-path "$index_path" \
            --marker-dir "$marker_dir" \
            --staging-root "$staging_root" \
            --video-workers "$video_workers" \
            --encoder-preset "$encoder_preset"
    ' _

expected="$(
    find "$archive_root" -maxdepth 1 -type f \
        -name 'openscene_sensor_trainval_camera_*.tgz' | wc -l
)"
actual="$(
    find "$marker_dir" -maxdepth 1 -type f -name '*.done' | wc -l
)"
if [[ "$actual" -ne "$expected" ]]; then
    printf "Expected %d completed video archives, found %d\n" \
        "$expected" "$actual" >&2
    exit 1
fi

printf "[OK] videos generated from all %d camera archives\n" "$actual"
