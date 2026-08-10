#!/usr/bin/env bash
# Convert local NAVSIM-v2 navhard assets to the metadata format consumed by infer.py.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
source "$project_root/env.sh"

sensor_root="$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs"
scene_root="$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles"
processed_root="$OPENSCENE_DATA_ROOT/meta/navhard_two_stage"
datalist="${NAVHARD_DATALIST:-$project_root/navhard_two_stage_meta.json}"

for path in "$sensor_root" "$scene_root"; do
  if [ ! -d "$path" ]; then
    echo "[groundedworld-navhard] missing downloaded NAVSIM-v2 asset: $path" >&2
    exit 2
  fi
done
if [ -e "$datalist" ] && [ "${OVERWRITE:-0}" != "1" ]; then
  echo "[groundedworld-navhard] refusing to overwrite datalist: $datalist" >&2
  exit 2
fi

(cd "$project_root/navsim_data_process" && \
  PYTHONPATH="$project_root/navsim:$project_root/navsim_data_process:${PYTHONPATH:-}" \
  python "$project_root/navsim_data_process/make_data.py" \
    --split navhard_two_stage --data_root "$OPENSCENE_DATA_ROOT")

python tools/grounded_world/build_datalist.py \
  --meta-dir "$processed_root" \
  --output "$datalist"

processed_count="$(find "$processed_root" -maxdepth 1 -type f -name '*.pkl' ! -name '*-depth.pkl' | wc -l)"
datalist_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$datalist")"
if [ "$processed_count" != "$datalist_count" ] || [ "$datalist_count" = "0" ]; then
  echo "[groundedworld-navhard] metadata count mismatch: pkl=$processed_count datalist=$datalist_count" >&2
  exit 3
fi
echo "[groundedworld-navhard] prepared $datalist_count samples at $processed_root"
