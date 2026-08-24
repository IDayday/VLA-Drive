#!/usr/bin/env bash
# Internal fail-closed launcher shared by Stage A and matched Stage B wrappers.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/load_env.sh"

stage=""
variant="gp"
dry_run=0
action_checkpoint="${ACTION_ONLY_CHECKPOINT:-}"
stage_a_checkpoint="${GP_STAGE_A_CHECKPOINT:-}"
gate_report="${GP_STAGE_A_GATE_REPORT:-}"
cache_root="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
stats_root="${GP_SQ3DMIX_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_slot_stats}"
source_datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
datalist=""
run_root="${GP_SQ3DMIX_RUN_ROOT:-$NAVSIM_EXP_ROOT}"
run_id=""
devices="${GP_SQ3DMIX_DEVICES:-8}"
batch_size="${GP_SQ3DMIX_BATCH_SIZE:-4}"
gradient_accumulation="${GP_SQ3DMIX_GRADIENT_ACCUMULATION:-1}"
extra_args=()

while (( $# )); do
  case "$1" in
    --stage) stage="${2:?}"; shift 2 ;;
    --variant) variant="${2:?}"; shift 2 ;;
    --action-checkpoint) action_checkpoint="${2:?}"; shift 2 ;;
    --stage-a-checkpoint) stage_a_checkpoint="${2:?}"; shift 2 ;;
    --gate-report) gate_report="${2:?}"; shift 2 ;;
    --cache-root) cache_root="${2:?}"; shift 2 ;;
    --stats-root) stats_root="${2:?}"; shift 2 ;;
    --source-datalist) source_datalist="${2:?}"; shift 2 ;;
    --datalist) datalist="${2:?}"; shift 2 ;;
    --run-root) run_root="${2:?}"; shift 2 ;;
    --run-id) run_id="${2:?}"; shift 2 ;;
    --devices) devices="${2:?}"; shift 2 ;;
    --batch-size) batch_size="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --) shift; extra_args=("$@"); break ;;
    *) echo "Unknown launcher option: $1" >&2; exit 2 ;;
  esac
done

[[ "$stage" == "stage_a" || "$stage" == "stage_b" ]] || { echo "--stage must be stage_a or stage_b" >&2; exit 2; }
[[ "$variant" == "gp" || ( "$stage" == "stage_b" && "$variant" == "control" ) ]] || { echo "Invalid stage/variant" >&2; exit 2; }
for value in "$devices" "$batch_size" "$gradient_accumulation"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Topology values must be positive integers" >&2; exit 2; }
done
effective_batch=$((devices * batch_size * gradient_accumulation))
[[ "$effective_batch" == "32" ]] || { echo "Effective global batch must remain 32, found $effective_batch" >&2; exit 2; }

branch="$(git -C "$project_root" branch --show-current)"
commit="$(git -C "$project_root" rev-parse HEAD)"
[[ "$branch" == "feature/gp-sq-3d-mix" ]] || { echo "Wrong DLC-visible branch: $branch" >&2; exit 2; }
if (( ! dry_run )) && [[ -n "$(git -C "$project_root" status --porcelain)" ]]; then
  echo "Formal GP training requires a clean DLC-visible worktree" >&2
  exit 2
fi
[[ -n "$action_checkpoint" ]] || { echo "ACTION_ONLY_CHECKPOINT is required (fail closed)" >&2; exit 2; }
[[ -f "$action_checkpoint" ]] || { echo "Missing action-only checkpoint: $action_checkpoint" >&2; exit 2; }
[[ -d "$BASE_VLM" ]] || { echo "Missing base VLM: $BASE_VLM" >&2; exit 2; }

if [[ "$stage" == "stage_a" ]]; then
  datalist="${datalist:-$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_train.json}"
  max_steps="${GP_STAGE_A_STEPS:-2000}"
  expected_count=2000
  save_interval="${GP_STAGE_A_SAVE_INTERVAL:-500}"
  mode=gated_residual
  cache_enabled=true
  run_id="${run_id:-gp-sq3dmix-stage-a-${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}}"
else
  datalist="${datalist:-$source_datalist}"
  max_steps="${GP_STAGE_B_STEPS:-10000}"
  expected_count="${GP_STAGE_B_EXPECTED_SAMPLES:-103288}"
  save_interval=2000
  [[ -n "$gate_report" && -f "$gate_report" ]] || { echo "Stage B requires GP_STAGE_A_GATE_REPORT" >&2; exit 2; }
  for path in "$stats_root/manifest.json" "$cache_root/vggt_dense/manifest.json" "$source_datalist"; do
    [[ -f "$path" ]] || { echo "Stage B gate-binding input is missing: $path" >&2; exit 2; }
  done
  [[ -n "$stage_a_checkpoint" && -f "$stage_a_checkpoint" ]] || {
    echo "Matched Stage B requires GP_STAGE_A_CHECKPOINT before either arm may start" >&2
    exit 2
  }
  python - "$gate_report" "$(git -C "$project_root" rev-parse HEAD)" \
    "$stats_root/manifest.json" "$cache_root/vggt_dense/manifest.json" \
    "$source_datalist" "$action_checkpoint" "$dry_run" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()
report=json.load(open(sys.argv[1]))
if report.get("all_passed") is not True:
    raise SystemExit("Stage A final gates did not all pass; Stage B is forbidden")
if report.get("code_commit") != sys.argv[2]:
    raise SystemExit("Stage A gate report was produced by a different code commit")
for key,path in (
    ("stats_manifest_sha256",sys.argv[3]),
    ("cache_manifest_sha256",sys.argv[4]),
    ("source_datalist_sha256",sys.argv[5]),
):
    if report.get(key) != sha(path):
        raise SystemExit(f"Stage A gate report {key} does not match the Stage B input")
if int(sys.argv[7]):
    if not report.get("action_only_checkpoint_sha256"):
        raise SystemExit("Stage A gate report has no action-only checkpoint SHA256")
elif report.get("action_only_checkpoint_sha256") != sha(sys.argv[6]):
    raise SystemExit("Stage B action-only checkpoint differs from Stage A")
PY
  python - "$gate_report" "$stage_a_checkpoint" "$dry_run" <<'PY'
import hashlib,json,sys
from pathlib import Path
report=json.load(open(sys.argv[1])); checkpoint=Path(sys.argv[2]).resolve()
if Path(report.get("checkpoint", "")).resolve() != checkpoint:
    raise SystemExit("Stage-A checkpoint does not match the passed final-gate report")
if int(sys.argv[3]):
    if not report.get("checkpoint_sha256"):
        raise SystemExit("Stage-A gate report has no checkpoint SHA256")
    raise SystemExit(0)
digest=hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(1024*1024), b""): digest.update(block)
if report.get("checkpoint_sha256") != digest.hexdigest():
    raise SystemExit("Stage-A checkpoint SHA256 does not match the final-gate report")
PY
  if [[ "$variant" == "control" ]]; then
    mode=disabled
    cache_enabled=false
  else
    mode=gated_residual
    cache_enabled=true
  fi
  run_id="${run_id:-gp-sq3dmix-stage-b-${variant}-${PAI_JOB_ID:-$(date +'%Y%m%d_%H%M%S')}}"
fi
[[ "$max_steps" =~ ^[1-9][0-9]*$ ]] || { echo "Training steps must be positive" >&2; exit 2; }
[[ -f "$datalist" ]] || { echo "Missing training datalist: $datalist" >&2; exit 2; }
actual_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$datalist")"
[[ "$actual_count" == "$expected_count" ]] || { echo "Datalist count $actual_count != $expected_count" >&2; exit 2; }

if [[ "$cache_enabled" == true ]]; then
  [[ -f "$cache_root/vggt_dense/manifest.json" ]] || { echo "Missing full train dense cache" >&2; exit 2; }
  [[ -f "$stats_root/gp_sq3dmix_pooled_stats.pt" && -f "$stats_root/manifest.json" ]] || { echo "Missing GP slot statistics; run script 19" >&2; exit 2; }
  [[ -f "$source_datalist" ]] || { echo "Missing full-train source datalist" >&2; exit 2; }
fi

run_dir="$run_root/$run_id"
[[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite experiment: $run_dir" >&2; exit 2; }
base_config="${TRAIN_CONFIG_YAML:-$project_root/starVLA/config/training/cfg_yaw_1225.yaml}"
overlay="$project_root/starVLA/config/training/gp_sq_3d_mix.yaml"
accelerate_config="${TRAIN_ACCELERATE_CONFIG:-$project_root/starVLA/config/deepseeds/deepspeed_zero2.yaml}"
for path in "$base_config" "$overlay" "$accelerate_config"; do [[ -f "$path" ]] || { echo "Missing config: $path" >&2; exit 2; }; done

visible_devices="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$visible_devices" ]]; then visible_devices="$(seq -s, 0 $((devices - 1)))"; fi
IFS=',' read -r -a visible_array <<< "$visible_devices"
[[ "${#visible_array[@]}" == "$devices" ]] || { echo "CUDA_VISIBLE_DEVICES does not match --devices" >&2; exit 2; }

launch=(accelerate launch --main_process_port "${MAIN_PROCESS_PORT:-29695}" --config_file "$accelerate_config" --num_processes "$devices" --num_machines 1 --machine_rank 0 --mixed_precision bf16)
train=(starVLA/training/train_starvla.py --config_yaml "$base_config" --config_overlay "$overlay"
  --run_root_dir "$run_root" --run_id "$run_id" --seed 20260824
  --framework.qwenvl.base_vlm "$BASE_VLM" --framework.qwenvl.attn_implementation sdpa
  --framework.gp_sq_3d_mix.mode "$mode" --framework.gp_sq_3d_mix.training.stage "$stage"
  --framework.gp_sq_3d_mix.cache.enabled "$cache_enabled"
  --datasets.vla_data.datalist_path "$datalist" --datasets.vla_data.data_root "$DATA_ROOT"
  --datasets.vla_data.split train --datasets.vla_data.expected_sample_count "$expected_count"
  --datasets.vla_data.per_device_batch_size "$batch_size" --datasets.vla_data.load_act_data 1
  --datasets.video_data.load_2d_data 0 --datasets.gs_data.load_3d_data 0
  --datasets.reward_data.load_reward_data 0 --w_depth 0 --rgb_query_loss 0 --gs_query_loss 0
  --trainer.pretrained_checkpoint "$action_checkpoint"
  --trainer.gradient_accumulation_steps "$gradient_accumulation"
  --trainer.max_train_steps "$max_steps" --trainer.num_warmup_steps 0
  --trainer.save_interval "$save_interval" --trainer.logging_frequency 20
  --trainer.optimizer.weight_decay 1e-3 --framework.action_model.repeated_diffusion_steps 1
  --trainer.learning_rate.action_model 1e-6)
if [[ "$cache_enabled" == true ]]; then
  train+=(--framework.gp_sq_3d_mix.cache.root "$cache_root"
    --framework.gp_sq_3d_mix.stats.root "$stats_root"
    --framework.gp_sq_3d_mix.stats.source_datalist "$source_datalist"
    --framework.gp_sq_3d_mix.stats.source_cache_manifest "$cache_root/vggt_dense/manifest.json")
fi
if [[ "$stage" == "stage_b" && "$variant" == "gp" ]]; then
  train+=(--trainer.gp_stage_a_checkpoint "$stage_a_checkpoint" --trainer.loss_weights.geometry_rank 0.05)
fi
train+=("${extra_args[@]}")

echo "[gp-train] code=$project_root branch=$branch commit=$commit"
echo "[gp-train] stage=$stage variant=$variant mode=$mode run=$run_dir"
echo "[gp-train] topology=1x$devices batch=$batch_size accumulation=$gradient_accumulation effective_batch=$effective_batch"
echo "[gp-train] action_checkpoint=$action_checkpoint"
echo "[gp-train] datalist=$datalist cache=$cache_root stats=$stats_root"
printf '[gp-train] command: CUDA_VISIBLE_DEVICES=%q ' "$visible_devices"
printf '%q ' "${launch[@]}" "${train[@]}"
printf '\n'
(( dry_run )) && exit 0

mkdir -p "$run_dir"
{
  printf 'branch=%s\ncommit=%s\nstage=%s\nvariant=%s\nmode=%s\n' "$branch" "$commit" "$stage" "$variant" "$mode"
  printf 'datalist_sha256=%s\ncache_manifest_sha256=%s\nslot_stats_manifest_sha256=%s\naction_checkpoint_sha256=%s\n' \
    "$(sha256sum "$datalist" | cut -d' ' -f1)" \
    "$([[ "$cache_enabled" == true ]] && sha256sum "$cache_root/vggt_dense/manifest.json" | cut -d' ' -f1 || printf disabled)" \
    "$([[ "$cache_enabled" == true ]] && sha256sum "$stats_root/manifest.json" | cut -d' ' -f1 || printf disabled)" \
    "$(sha256sum "$action_checkpoint" | cut -d' ' -f1)"
  printf 'resolved_command='
  printf '%q ' CUDA_VISIBLE_DEVICES="$visible_devices" "${launch[@]}" "${train[@]}"
  printf '\n'
} > "$run_dir/launcher_manifest.txt"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}/${run_id}/node0"
mkdir -p "$TRITON_CACHE_DIR"
CUDA_VISIBLE_DEVICES="$visible_devices" "${launch[@]}" "${train[@]}"
