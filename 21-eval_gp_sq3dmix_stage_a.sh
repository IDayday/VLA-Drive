#!/usr/bin/env bash
# Select on an independent split, then enforce all Stage-A gates on final split.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"

run_dir="${GP_STAGE_A_RUN_DIR:-}"
checkpoint=""
cache_root="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
stats_root="${GP_SQ3DMIX_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_slot_stats}"
source_datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
selection_datalist="${GP_STAGE_A_SELECTION_DATALIST:-$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_model_selection.json}"
final_datalist="${GP_STAGE_A_FINAL_DATALIST:-$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_final_gate.json}"
output_root=""
result_csv="${GP_STAGE_A_RESULT_CSV:-$project_root/docs/experiments/results/gp_sq3dmix_stage_a.csv}"
dry_run=0
while (( $# )); do
  case "$1" in
    --run-dir) run_dir="${2:?}"; shift 2 ;;
    --checkpoint) checkpoint="${2:?}"; shift 2 ;;
    --cache-root) cache_root="${2:?}"; shift 2 ;;
    --stats-root) stats_root="${2:?}"; shift 2 ;;
    --selection-datalist) selection_datalist="${2:?}"; shift 2 ;;
    --final-datalist) final_datalist="${2:?}"; shift 2 ;;
    --output-root) output_root="${2:?}"; shift 2 ;;
    --result-csv) result_csv="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo "Usage: bash $project_root/21-eval_gp_sq3dmix_stage_a.sh --run-dir DIR [--checkpoint FILE] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$run_dir" && -f "$run_dir/config.yaml" ]] || { echo "Valid --run-dir is required" >&2; exit 2; }
[[ "$(git -C "$project_root" branch --show-current)" == "feature/gp-sq-3d-mix" ]] || { echo "Wrong DLC-visible branch" >&2; exit 2; }
for path in "$cache_root/vggt_dense/manifest.json" "$stats_root/manifest.json" "$stats_root/gp_sq3dmix_pooled_stats.pt" "$source_datalist" "$selection_datalist" "$final_datalist"; do
  [[ -e "$path" ]] || { echo "Missing Stage-A evaluation input: $path" >&2; exit 2; }
done
output_root="${output_root:-$run_dir/stage_a_gate}"
if [[ -n "$checkpoint" ]]; then
  candidates=("$checkpoint")
else
  mapfile -t candidates < <(find "$run_dir/checkpoints" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' | sort -V)
fi
(( ${#candidates[@]} )) || { echo "No Stage-A checkpoints found" >&2; exit 2; }
for path in "${candidates[@]}"; do [[ -f "$path" ]] || { echo "Missing checkpoint: $path" >&2; exit 2; }; done
echo "[stage-a-eval] code=$project_root run=$run_dir candidates=${#candidates[@]} output=$output_root"
if (( dry_run )); then
  printf 'python %q --run-dir %q --checkpoint %q --datalist %q --output %q\n' "$project_root/tools/evaluate_gp_sq3dmix_stage_a.py" "$run_dir" "${candidates[0]}" "$selection_datalist" "$output_root/model_selection/example.json"
  printf 'python %q --run-dir %q --checkpoint SELECTED --datalist %q --output %q\n' "$project_root/tools/evaluate_gp_sq3dmix_stage_a.py" "$run_dir" "$final_datalist" "$output_root/final_gate.json"
  printf 'write immutable result CSV %q\n' "$result_csv"
  exit 0
fi
[[ ! -e "$output_root" ]] || { echo "Refusing to overwrite gate output: $output_root" >&2; exit 2; }
if [[ -e "$result_csv" ]]; then
  python - "$result_csv" <<'PY'
import csv,sys
with open(sys.argv[1],newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
if len(rows) != 1 or rows[0].get("status") != "not_run":
    raise SystemExit(f"Refusing to overwrite completed result CSV: {sys.argv[1]}")
PY
fi
mkdir -p "$output_root/model_selection"
reports=()
for candidate in "${candidates[@]}"; do
  name="$(basename -- "$candidate" .pt)"
  report="$output_root/model_selection/$name.json"
  BASE_VLM="$BASE_VLM" python "$project_root/tools/evaluate_gp_sq3dmix_stage_a.py" \
    --run-dir "$run_dir" --checkpoint "$candidate" --datalist "$selection_datalist" \
    --cache-root "$cache_root" --stats-root "$stats_root" --source-datalist "$source_datalist" \
    --output "$report" --batch-size "${GP_STAGE_A_EVAL_BATCH_SIZE:-4}" \
    --num-workers "${GP_STAGE_A_EVAL_WORKERS:-2}"
  reports+=("$report")
done
selected="$(python "$project_root/tools/select_gp_sq3dmix_stage_a_checkpoint.py" "${reports[@]}")"
BASE_VLM="$BASE_VLM" python "$project_root/tools/evaluate_gp_sq3dmix_stage_a.py" \
  --run-dir "$run_dir" --checkpoint "$selected" --datalist "$final_datalist" \
  --cache-root "$cache_root" --stats-root "$stats_root" --source-datalist "$source_datalist" \
  --output "$output_root/final_gate.json" --batch-size "${GP_STAGE_A_EVAL_BATCH_SIZE:-4}" \
  --num-workers "${GP_STAGE_A_EVAL_WORKERS:-2}"
python - "$output_root/final_gate.json" "$result_csv" <<'PY'
import csv,json,os,sys
from pathlib import Path
report=json.load(open(sys.argv[1])); output=Path(sys.argv[2])
if output.exists():
 with output.open(newline="",encoding="utf-8") as f: existing=list(csv.DictReader(f))
 if len(existing) != 1 or existing[0].get("status") != "not_run": raise SystemExit(f"Refusing to overwrite {output}")
output.parent.mkdir(parents=True,exist_ok=True); tmp=output.with_name(output.name+f".tmp-{os.getpid()}")
with tmp.open("w",newline="") as f:
 w=csv.writer(f); w.writerow(["phase","status","reason","checkpoint","sample_count","mean_real_minus_base","utility_ci_lower","utility_ci_upper","relative_shuffled_real_gap","gap_ci_lower","gap_ci_upper","all_passed"])
 w.writerow(["stage_a","complete","",report["checkpoint"],report["sample_count"],report["mean_real_minus_base"],*report["real_minus_base_bootstrap_ci"],report["relative_shuffled_real_flow_gap"],*report["relative_gap_bootstrap_ci"],report["all_passed"]])
os.replace(tmp,output)
PY
if python -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["all_passed"] else 1)' "$output_root/final_gate.json"; then
  echo "[stage-a-eval] GO for matched Stage B: selected=$selected gate=$output_root/final_gate.json"
else
  echo "[stage-a-eval] NO-GO: one or more immutable Stage-A gates failed; Stage B is forbidden"
  exit 3
fi
