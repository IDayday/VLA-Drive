#!/usr/bin/env bash
# Source repository defaults from any non-interactive entrypoint.

_vla_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_vla_repo_root}/env.sh"
cd "${DRIVEDREAMER_ROOT}"
unset _vla_repo_root
