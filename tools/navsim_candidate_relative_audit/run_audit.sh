#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SPLIT="train"
MAX_SCENES=500
NUM_CANDIDATES=12
OUTPUT_DIR="reports/navsim_candidate_relative_audit"
TRAFFIC_POLICY="non_reactive"
MODE="smoke"

usage() {
  printf '%s\n' \
    "Usage: bash tools/navsim_candidate_relative_audit/run_audit.sh [options]" \
    "  --mode smoke|full" \
    "  --split mini|train|trainval" \
    "  --max-scenes N" \
    "  --num-candidates 8..16" \
    "  --output-dir PATH" \
    "  --traffic-policy non_reactive|reactive"
}

while (($#)); do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --max-scenes) MAX_SCENES="$2"; shift 2 ;;
    --num-candidates) NUM_CANDIDATES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --traffic-policy) TRAFFIC_POLICY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${MODE}" == "smoke" || "${MODE}" == "full" ]] || { printf 'Invalid --mode\n' >&2; exit 2; }
[[ "${SPLIT}" == "mini" || "${SPLIT}" == "train" || "${SPLIT}" == "trainval" ]] || { printf 'Invalid --split\n' >&2; exit 2; }
[[ "${TRAFFIC_POLICY}" == "non_reactive" || "${TRAFFIC_POLICY}" == "reactive" ]] || { printf 'Invalid --traffic-policy\n' >&2; exit 2; }
((NUM_CANDIDATES >= 8 && NUM_CANDIDATES <= 16)) || { printf '--num-candidates must be 8..16\n' >&2; exit 2; }
((MAX_SCENES > 0)) || { printf '--max-scenes must be positive\n' >&2; exit 2; }

mkdir -p "${OUTPUT_DIR}"
COMMAND_LOG="${OUTPUT_DIR}/COMMANDS.sh"
if [[ ! -e "${COMMAND_LOG}" ]]; then
  printf '#!/usr/bin/env bash\nset -euo pipefail\n\n' >"${COMMAND_LOG}"
fi

run() {
  local rendered
  printf -v rendered '%q ' "$@"
  printf '%s\n' "${rendered% }" | tee -a "${COMMAND_LOG}"
  "$@"
}

python_module() {
  run env PYTHONPATH=. python -m "$@"
}

assert_gate() {
  local path="$1"
  local key="$2"
  run python -c "import json; x=json.load(open('${path}')); assert x['${key}']=='PASS', x"
}

run_core() {
  local scenes="$1"
  local out="$2"
  local scene_mode="$3"
  python_module tools.navsim_candidate_relative_audit.inspect_environment \
    --split "${SPLIT}" --max-scenes "${scenes}" --output-dir "${out}"
  python_module tools.navsim_candidate_relative_audit.inspect_scenes \
    --mode "${scene_mode}" --split "${SPLIT}" --max-scenes "${scenes}" --output-dir "${out}"
  python_module tools.navsim_candidate_relative_audit.validate_alignment \
    --split "${SPLIT}" --max-scenes "$((scenes < 64 ? scenes : 64))" --output-dir "${out}"
  assert_gate "${out}/gate_a.json" gate_a
  python_module tools.navsim_candidate_relative_audit.candidate_generator \
    --split "${SPLIT}" --max-scenes "${scenes}" --num-candidates "${NUM_CANDIDATES}" \
    --traffic-policy "${TRAFFIC_POLICY}" --output-dir "${out}"
  python_module tools.navsim_candidate_relative_audit.score_candidates \
    --split "${SPLIT}" --max-scenes "${scenes}" --traffic-policy "${TRAFFIC_POLICY}" \
    --output-dir "${out}"
  assert_gate "${out}/gate_b.json" gate_b
  if [[ "${TRAFFIC_POLICY}" == "non_reactive" ]]; then
    python_module tools.navsim_candidate_relative_audit.build_candidate_relative_targets \
      --split "${SPLIT}" --max-scenes "${scenes}" --traffic-policy non_reactive --output-dir "${out}"
    python_module tools.navsim_candidate_relative_audit.analyze_target_diversity \
      --split "${SPLIT}" --max-scenes "${scenes}" --output-dir "${out}"
    python_module tools.navsim_candidate_relative_audit.build_soft_contrastive_labels \
      --split "${SPLIT}" --max-scenes "${scenes}" --output-dir "${out}"
  fi
}

if [[ "${MODE}" == "smoke" ]]; then
  SMOKE_SCENES=$((MAX_SCENES < 8 ? MAX_SCENES : 8))
  run_core "${SMOKE_SCENES}" "${OUTPUT_DIR}" smoke
  EFFECTIVE_SCENES="${SMOKE_SCENES}"
else
  # Full mode is gated by an isolated 8-scene preflight. The preflight never
  # mutates source data/caches and its outputs remain available for diagnosis.
  PREFLIGHT_DIR="${OUTPUT_DIR}/smoke_preflight"
  run_core 8 "${PREFLIGHT_DIR}" smoke
  run_core "${MAX_SCENES}" "${OUTPUT_DIR}" statistics
  EFFECTIVE_SCENES="${MAX_SCENES}"
fi

if [[ "${TRAFFIC_POLICY}" == "non_reactive" ]]; then
  ORACLE_SCENES=$((EFFECTIVE_SCENES < 500 ? EFFECTIVE_SCENES : 500))
  if ((ORACLE_SCENES < 8)); then ORACLE_SCENES=8; fi
  python_module tools.navsim_candidate_relative_audit.run_oracle_probe \
    --split "${SPLIT}" --max-scenes "${ORACLE_SCENES}" --num-candidates "${NUM_CANDIDATES}" \
    --max-scenes-per-log 8 --output-dir "${OUTPUT_DIR}"
  VISUAL_SCENES=$((EFFECTIVE_SCENES < 12 ? EFFECTIVE_SCENES : 12))
  python_module tools.navsim_candidate_relative_audit.audit_future_visual_anchor \
    --split "${SPLIT}" --max-scenes "${VISUAL_SCENES}" --output-dir "${OUTPUT_DIR}"
fi

V2_SCENES=$((EFFECTIVE_SCENES < 32 ? EFFECTIVE_SCENES : 32))
if [[ "${MODE}" == "smoke" ]]; then
  python_module tools.navsim_candidate_relative_audit.audit_v2_extensions \
    --split "${SPLIT}" --max-scenes "${V2_SCENES}" --skip-track-rerun \
    --synthetic-metadata-samples 16 --output-dir "${OUTPUT_DIR}"
else
  python_module tools.navsim_candidate_relative_audit.audit_v2_extensions \
    --split "${SPLIT}" --max-scenes "${V2_SCENES}" --output-dir "${OUTPUT_DIR}"
fi

if [[ "${TRAFFIC_POLICY}" == "non_reactive" ]]; then
  run env MPLBACKEND=Agg PYTHONPATH=. python -m tools.navsim_candidate_relative_audit.visualize_audit \
    --split "${SPLIT}" --max-scenes "${EFFECTIVE_SCENES}" --output-dir "${OUTPUT_DIR}"
  python_module tools.navsim_candidate_relative_audit.generate_report \
    --split "${SPLIT}" --max-scenes "${EFFECTIVE_SCENES}" --output-dir "${OUTPUT_DIR}"
fi

run env PYTHONPATH=. pytest -q tests/test_navsim_candidate_relative_audit.py

if [[ "${TRAFFIC_POLICY}" == "non_reactive" ]]; then
  run python - "${OUTPUT_DIR}" "${SPLIT}" <<'PY'
import json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
split = sys.argv[2]
env = json.loads((root / "environment.json").read_text())
field = json.loads((root / "field_inventory.json").read_text())
candidate = json.loads((root / "candidate_generation_summary.json").read_text())
score = json.loads((root / "candidate_scoring_summary.json").read_text())
target = json.loads((root / "candidate_relative_target_summary.json").read_text())
v2 = json.loads((root / "v2_extension_results.json").read_text())
visual = json.loads((root / "future_visual_anchor_summary.json").read_text())
final = json.loads((root / "final_summary.json").read_text())
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print(f"""Branch: {env['repository']['branch']}
Commit: {commit}
NAVSIM version: {env['navsim_runtime']['setup_version']}
Dataset split: {split}
Number of audited scenes: {field['audited_scene_count']}
Number of candidates per scene: {candidate['candidates_per_scene']}
Future camera coverage: {visual['future_front_camera_coverage']:.3%}
Future track coverage: {field['field_coverage']['future_actor_track_token_coverage']:.3%}
Candidate scoring success rate: {score['success_rate']:.3%}
Candidate-relative target coverage: {target['candidate_relative_target_coverage']:.3%}
Reactive policy available: {v2['reactive']['available']}
Synthetic scenes available: {v2['synthetic']['available']}

F1 candidate-relative structured consequence: {final['judgements']['F1 candidate-relative structured consequence']}
F2 GT visual anchor: {final['judgements']['F2 GT visual anchor']}
F3 soft contrastive supervision: {final['judgements']['F3 soft contrastive supervision']}
F4 inverse verifier supervision: {final['judgements']['F4 inverse verifier supervision']}
F5 non-GT future image supervision: {final['judgements']['F5 non-GT future image supervision']}

Primary blocker: {final['primary_blocker']}
Recommended next plan: {final['recommended_plan']}
Report path: {root / 'FINAL_FEASIBILITY_REPORT.md'}""")
PY
fi
