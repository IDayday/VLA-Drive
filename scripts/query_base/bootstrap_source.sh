#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'Usage: bash scripts/query_base/bootstrap_source.sh DESTINATION' >&2
  exit 2
fi
destination=$1
revision=1493deda247107efe076b9294863a6753391298a
if [[ -e "$destination" ]]; then
  test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
  git -C "$destination" diff --quiet HEAD
  exit
fi
git init "$destination"
git -C "$destination" remote add origin https://github.com/IDayday/VLA-Drive.git
git -C "$destination" fetch --depth 1 origin "$revision"
git -C "$destination" checkout --detach FETCH_HEAD
test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
