#!/usr/bin/env bash
# Source the shared repository environment from any top-level entrypoint.

_vla_env_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_vla_repo_root="$(cd -- "${_vla_env_script_dir}/.." && pwd)"

source "${_vla_repo_root}/env.sh"
cd "${DRIVEDREAMER_ROOT}"

unset _vla_env_script_dir _vla_repo_root
