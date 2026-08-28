#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

split="trainval"
max_scenes=500
num_candidates=12
output_dir="${REPO_ROOT}/reports/navsim_candidate_relative_audit"
traffic_policy="non_reactive"
mode="smoke"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) split="$2"; shift 2 ;;
    --max-scenes) max_scenes="$2"; shift 2 ;;
    --num-candidates) num_candidates="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --traffic-policy) traffic_policy="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --mode smoke|full --split trainval|mini|navtrain --max-scenes N --num-candidates K --output-dir DIR --traffic-policy non_reactive|reactive"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${split}" in
  mini|trainval|navtrain) ;;
  *) echo "Refusing disallowed split '${split}'; test/navtest/navhard/private labels are outside scope." >&2; exit 2 ;;
esac
case "${mode}" in smoke|full) ;; *) echo "--mode must be smoke or full" >&2; exit 2 ;; esac
case "${traffic_policy}" in non_reactive|reactive) ;; *) echo "Unknown traffic policy '${traffic_policy}'" >&2; exit 2 ;; esac
if ! [[ "${max_scenes}" =~ ^[1-9][0-9]*$ && "${num_candidates}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-scenes and --num-candidates must be positive integers" >&2
  exit 2
fi

mkdir -p "${output_dir}"
cd "${REPO_ROOT}"

python - "${output_dir}/COMMANDS.sh" "${mode}" "${split}" "${max_scenes}" "${num_candidates}" "${output_dir}" "${traffic_policy}" <<'PY'
from pathlib import Path
import shlex
import sys

path = Path(sys.argv[1])
values = sys.argv[2:]
command = [
    "bash", "tools/navsim_candidate_relative_audit/run_audit.sh",
    "--mode", values[0], "--split", values[1], "--max-scenes", values[2],
    "--num-candidates", values[3], "--output-dir", values[4],
    "--traffic-policy", values[5],
]
if not path.exists():
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n", encoding="utf-8")
with path.open("a", encoding="utf-8") as stream:
    stream.write(" ".join(shlex.quote(item) for item in command) + "\n")
PY

common=(--split "${split}" --max-scenes "${max_scenes}" --num-candidates "${num_candidates}" --output-dir "${output_dir}" --mode "${mode}")
plain=(--split "${split}" --max-scenes "${max_scenes}" --output-dir "${output_dir}" --mode "${mode}")

if [[ "${traffic_policy}" == "reactive" ]]; then
  python -m tools.navsim_candidate_relative_audit.inspect_environment "${plain[@]}"
  python -m tools.navsim_candidate_relative_audit.audit_v2_extensions "${plain[@]}"
  echo "Reactive code was audited, but no eligible mini/trainval v2 metric cache is configured. Results were not mixed with non-reactive labels." >&2
  exit 3
fi

if [[ "${mode}" == "full" ]]; then
  if ! python - "${output_dir}/gate_status.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text()) if path.is_file() else {}
raise SystemExit(0 if data.get("gate_a", {}).get("passed") and data.get("gate_b", {}).get("passed") else 1)
PY
  then
    echo "Full audit requires a successful smoke Gate A/B in the same output directory. Run --mode smoke first." >&2
    exit 4
  fi
fi

python -m tools.navsim_candidate_relative_audit.inspect_environment "${plain[@]}"
python -m tools.navsim_candidate_relative_audit.inspect_scenes "${plain[@]}"

# Gate A is deliberately bounded at 64 cache-matched scenes even in a 500-scene
# statistics run; subsequent candidate stages remain blocked on its result.
alignment_scenes="${max_scenes}"
alignment_mode="${mode}"
if [[ "${mode}" == "smoke" ]]; then
  alignment_scenes=8
elif [[ "${alignment_scenes}" -gt 64 ]]; then
  alignment_scenes=64
fi
python -m tools.navsim_candidate_relative_audit.validate_alignment \
  --split "${split}" --max-scenes "${alignment_scenes}" --output-dir "${output_dir}" --mode "${alignment_mode}"

python -m tools.navsim_candidate_relative_audit.candidate_generator "${common[@]}"
python -m tools.navsim_candidate_relative_audit.score_candidates "${common[@]}" --traffic-policy non_reactive --sanity-scenes 2
python -m tools.navsim_candidate_relative_audit.build_candidate_relative_targets "${common[@]}" --max-actors 16 --max-shared-actors 64
python -m tools.navsim_candidate_relative_audit.analyze_target_diversity "${plain[@]}"
python -m tools.navsim_candidate_relative_audit.build_soft_contrastive_labels "${plain[@]}"
python -m tools.navsim_candidate_relative_audit.run_oracle_probe "${plain[@]}"
python -m tools.navsim_candidate_relative_audit.audit_future_visual_anchor "${plain[@]}" --num-figures 12
python -m tools.navsim_candidate_relative_audit.audit_v2_extensions "${plain[@]}"
python -m tools.navsim_candidate_relative_audit.visualize_audit "${plain[@]}"
pytest -q tests/test_navsim_candidate_relative_audit.py | tee "${output_dir}/TEST_RESULTS.txt"
python -m tools.navsim_candidate_relative_audit.generate_final_report "${plain[@]}"
