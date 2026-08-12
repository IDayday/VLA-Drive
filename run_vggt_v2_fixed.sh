#!/usr/bin/env bash
# Restart the full V2 run after the DeepSpeed BF16 diagnostic-path fix.
#
# This launcher never generates the VGGT cache. It reuses a completed atomic
# manifest, runs a 2-step intervention smoke at interval 1, then starts formal
# training with intervention diagnostics every 500 steps by default.

set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

if [[ "${VGGT_FIXED_CONFIRM_OLD_JOB_STOPPED:-0}" != "1" ]]; then
  echo "Refusing to start a second full V2 job." >&2
  echo "Stop/cancel the old V2 DLC job, then set VGGT_FIXED_CONFIRM_OLD_JOB_STOPPED=1." >&2
  exit 2
fi

cache_manifest="$NAVSIM_VGGT_CACHE_ROOT/vggt_query/manifest.json"
if [[ ! -f "$cache_manifest" ]]; then
  echo "Missing completed VGGT cache manifest: $cache_manifest" >&2
  echo "This fixed restart launcher will not generate or repair the cache." >&2
  exit 2
fi

# Fail before allocating 16 PPUs if this checkout does not contain the dtype
# fix. The functional interval-1 smoke below remains the authoritative runtime
# check on the PPU/DeepSpeed stack.
python - "$DRIVEDREAMER_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = {
    root / "starVLA/model/modules/vggt_query/geometry_memory.py": (
        "combined.to(dtype=compute_dtype)",
    ),
    root / "starVLA/model/modules/vggt_query/alignment.py": (
        "student_queries.to(dtype=compute_dtype)",
    ),
    root / "starVLA/model/modules/vggt_query/planning_heads.py": (
        "geometry_memory.to(dtype=self.memory_norm.weight.dtype)",
    ),
    root / "8-train_vggt_action.sh": (
        "VGGT_INTERVENTION_INTERVAL",
    ),
}
for path, snippets in required.items():
    text = path.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in text:
            raise RuntimeError(
                f"Checkout is missing the V2 BF16 fix marker {snippet!r} in {path}"
            )
print("[vggt-fixed] source dtype-fix contract PASS")
PY

# A formal fixed restart is always the full V2 treatment, never a control or
# debug run. One-shot path and optimization overrides remain available through
# the normal environment precedence.
unset VGGT_EXPERIMENT_OVERLAY
unset TRAINING_SKIP_FINAL_SAVE
export VGGT_DEBUG=0
export VGGT_REQUIRE_TEACHER_CACHE=1
export VGGT_PIPELINE_SKIP_TRAIN=0
export VGGT_CACHE_FULL_VALIDATE="${VGGT_CACHE_FULL_VALIDATE:-0}"
export VGGT_RUN_SMOKE_BEFORE_FORMAL=1
export VGGT_INTERVENTION_INTERVAL="${VGGT_INTERVENTION_INTERVAL:-500}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export RUN_ID="${RUN_ID:-vggt-v2-layer11-global-m195-dtypefix-${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}}"

if ! [[ "$VGGT_INTERVENTION_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "VGGT_INTERVENTION_INTERVAL must be a positive integer, got: $VGGT_INTERVENTION_INTERVAL" >&2
  exit 2
fi

run_dir="$NAVSIM_EXP_ROOT/$RUN_ID"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing V2 output: $run_dir" >&2
  echo "Use a new RUN_ID." >&2
  exit 2
fi

echo "[vggt-fixed] run_id=$RUN_ID"
echo "[vggt-fixed] cache_policy=existing-manifest-only full_validate=$VGGT_CACHE_FULL_VALIDATE"
echo "[vggt-fixed] smoke_intervention_interval=1 formal_intervention_interval=$VGGT_INTERVENTION_INTERVAL"

exec bash "$DRIVEDREAMER_ROOT/run_vggt_pipeline.sh"
