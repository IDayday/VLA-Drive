#!/usr/bin/env bash
# Source the shared repository environment from any top-level entrypoint.

_vla_env_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${_vla_env_script_dir}/env.sh" ]; then
  _vla_repo_root="${_vla_env_script_dir}"
elif [ -f "${_vla_env_script_dir}/../env.sh" ]; then
  _vla_repo_root="$(cd -- "${_vla_env_script_dir}/.." && pwd)"
else
  echo "load_env.sh: env.sh not found next to load_env.sh or in its parent directory" >&2
  return 1 2>/dev/null || exit 1
fi

source "${_vla_repo_root}/env.sh"
cd "${DRIVEDREAMER_ROOT}"

unset _vla_env_script_dir _vla_repo_root
