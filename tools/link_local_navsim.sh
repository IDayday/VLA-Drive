#!/usr/bin/env bash
# Link the local NAVSIM mirror into the layout expected by the devkit.

set -euo pipefail

: "${DRIVEDREAMER_ROOT:?Run 'source env.sh' first}"

source_root="${NAVSIM_SOURCE_ROOT:-/mnt/data_and_weight/Public_Space/navsim}"
raw_root="${DRIVEDREAMER_ROOT}/navsim_dataset_raw"
full_trainval="${source_root}/trainval_all/trainval_sensor_blobs/trainval"
sparse_trainval="${source_root}/trainval_sensor_blobs/trainval"

if [[ -d "$full_trainval" ]]; then
    trainval_sensor_root="$full_trainval"
else
    trainval_sensor_root="$sparse_trainval"
fi

required_paths=(
    "$source_root/maps"
    "$source_root/mini_navsim_logs/mini"
    "$source_root/test_navsim_logs/test"
    "$source_root/trainval_navsim_logs/trainval"
    "$source_root/mini_sensor_blobs/mini"
    "$source_root/test_sensor_blobs/test"
    "$trainval_sensor_root"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        printf "Missing required NAVSIM path: %s\n" "$path" >&2
        exit 1
    fi
done

mkdir -p "$raw_root/navsim_logs" "$raw_root/sensor_blobs"
ln -sfnT "$source_root/maps" "$raw_root/maps"
ln -sfnT "$source_root/mini_navsim_logs/mini" "$raw_root/navsim_logs/mini"
ln -sfnT "$source_root/test_navsim_logs/test" "$raw_root/navsim_logs/test"
ln -sfnT \
    "$source_root/trainval_navsim_logs/trainval" \
    "$raw_root/navsim_logs/trainval"
ln -sfnT "$source_root/mini_sensor_blobs/mini" "$raw_root/sensor_blobs/mini"
ln -sfnT "$source_root/test_sensor_blobs/test" "$raw_root/sensor_blobs/test"
ln -sfnT "$trainval_sensor_root" "$raw_root/sensor_blobs/trainval"

for optional_split in \
    navhard_two_stage \
    private_test_hard_two_stage \
    warmup_two_stage; do
    if [[ -e "$source_root/$optional_split" ]]; then
        ln -sfnT "$source_root/$optional_split" "$raw_root/$optional_split"
    fi
done

printf "[OK] NAVSIM links created under %s\n" "$raw_root"
printf "[OK] trainval sensors: %s\n" "$trainval_sensor_root"
