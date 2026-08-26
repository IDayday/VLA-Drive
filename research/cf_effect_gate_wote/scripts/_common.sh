#!/usr/bin/env bash

set -Eeuo pipefail

cf_gate_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cf_gate_project_root="$(cd -- "$cf_gate_script_dir/../../.." && pwd)"
cf_gate_pythonpath="$cf_gate_project_root${PYTHONPATH:+:$PYTHONPATH}"

cf_gate_print_command() {
  printf 'COMMAND'
  printf ' %q' "$@"
  printf '\n'
}

cf_gate_require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    printf '[cf-effect-gate] missing %s: %s\n' "$label" "$path" >&2
    return 1
  fi
}

cf_gate_require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    printf '[cf-effect-gate] missing %s: %s\n' "$label" "$path" >&2
    return 1
  fi
}

cf_gate_run_python() {
  PYTHONPATH="$cf_gate_pythonpath" python "$@"
}
