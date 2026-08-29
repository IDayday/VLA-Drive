#!/usr/bin/env bash

set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/_oracle_effect_common.sh"
gate2o_parse_args "$@"
gate2o_require_common

cache_one() {
  local tokens="$1" output="$2" split="$3"
  if [[ ! -f "$output/manifest.json" ]]; then
    gate2o_run_python -m research.cf_effect_gate_wote.src.cache_wote_features cache \
      --wote-root "$gate2o_wote_root" --release-root "$gate2o_release_root" \
      --data-root "$gate2o_data_root" --tokens "$tokens" --output "$output" \
      --run-id "$gate2o_run_id-$split" --split "$split" --device "$gate2o_device" \
      --shard-scenes 16 --label-source none
  fi
}

cache_one "$gate2o_determinism_tokens" "$gate2o_output_root/features-determinism16" smoke
cache_one "$gate2o_train_tokens" "$gate2o_output_root/features-train" train
cache_one "$gate2o_val_tokens" "$gate2o_output_root/features-val" val
cache_one "$gate2o_test_tokens" "$gate2o_output_root/features-test" test
printf '%s\n' '[gate2o] label-free frozen WoTE feature caches complete'

