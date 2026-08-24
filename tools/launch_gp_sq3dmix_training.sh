#!/usr/bin/env bash
# Fail-closed unified launcher for GP-SQ3D-Mix Stage-A-v2 and matched continuations.

set -Eeuo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/load_env.sh"

phase=""
variant=""
seed=20260824
dry_run=0
preflight_only=0
active_phase=argument_validation
trap 'status=$?; echo "[gp-train-v2] FAILED phase=$active_phase status=$status" >&2; exit "$status"' ERR
action_checkpoint="${ACTION_ONLY_CHECKPOINT:-}"
stage_a_checkpoint="${GP_STAGE_A_V2_CHECKPOINT:-}"
decision_report="${GP_STAGE_A_V2_DECISION:-}"
permission_report="${GP_FORMAL_PERMISSION_REPORT:-}"
cache_root="${GP_SQ3DMIX_TRAIN_CACHE_ROOT:-$NAVSIM_VGGT_DENSE_CACHE_ROOT}"
stats_root="${GP_SQ3DMIX_V2_STATS_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_slot_stats}"
negative_root="${GP_SQ3DMIX_V2_NEGATIVE_MAP_ROOT:-$DRIVEDREAMER_SHARED_ROOT/navsim_feature_cache/gp_sq3dmix_stage_a_v2_negative_maps}"
source_datalist="${GP_SQ3DMIX_TRAIN_DATALIST:-$NAVSIM_DATALIST_PATH}"
datalist=""
negative_map_dir=""
run_root="${GP_SQ3DMIX_V2_RUN_ROOT:-$NAVSIM_EXP_ROOT/gp_sq3dmix_stage_a_v2}"
run_id=""
devices="${GP_SQ3DMIX_DEVICES:-1}"
per_device_batch="${GP_SQ3DMIX_PER_DEVICE_BATCH:-2}"
gradient_accumulation="${GP_SQ3DMIX_GRADIENT_ACCUMULATION:-}"
max_steps=""
save_interval=""
warmup_steps=""
resume_checkpoint="none"
resume_step=0
extra_args=()

while (( $# )); do
  case "$1" in
    --phase|--stage) phase="${2:?}"; shift 2 ;;
    --variant) variant="${2:?}"; shift 2 ;;
    --seed) seed="${2:?}"; shift 2 ;;
    --action-checkpoint) action_checkpoint="${2:?}"; shift 2 ;;
    --stage-a-checkpoint) stage_a_checkpoint="${2:?}"; shift 2 ;;
    --decision-report|--gate-report) decision_report="${2:?}"; shift 2 ;;
    --permission-report) permission_report="${2:?}"; shift 2 ;;
    --cache-root) cache_root="${2:?}"; shift 2 ;;
    --stats-root) stats_root="${2:?}"; shift 2 ;;
    --negative-root) negative_root="${2:?}"; shift 2 ;;
    --negative-map-dir) negative_map_dir="${2:?}"; shift 2 ;;
    --source-datalist) source_datalist="${2:?}"; shift 2 ;;
    --datalist) datalist="${2:?}"; shift 2 ;;
    --run-root) run_root="${2:?}"; shift 2 ;;
    --run-id) run_id="${2:?}"; shift 2 ;;
    --devices) devices="${2:?}"; shift 2 ;;
    --per-device-batch|--batch-size) per_device_batch="${2:?}"; shift 2 ;;
    --gradient-accumulation) gradient_accumulation="${2:?}"; shift 2 ;;
    --max-steps) max_steps="${2:?}"; shift 2 ;;
    --save-interval) save_interval="${2:?}"; shift 2 ;;
    --warmup-steps) warmup_steps="${2:?}"; shift 2 ;;
    --resume-checkpoint) resume_checkpoint="${2:?}"; shift 2 ;;
    --resume-step) resume_step="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --preflight-only) preflight_only=1; shift ;;
    --) shift; extra_args=("$@"); break ;;
    *) echo "Unknown launcher option: $1" >&2; exit 2 ;;
  esac
done

case "$phase" in
  smoke|stage_a_v2|stage_b|stage_c_30k|stage_c_100k) ;;
  stage_a) phase=stage_a_v2 ;;
  *) echo "--phase must be smoke, stage_a_v2, stage_b, stage_c_30k, or stage_c_100k" >&2; exit 2 ;;
esac
case "$variant" in
  projected_residual|gated_residual|control) ;;
  gp) variant=gated_residual ;;
  *) echo "--variant must be projected_residual, gated_residual, or control" >&2; exit 2 ;;
esac
if [[ "$phase" == "smoke" || "$phase" == "stage_a_v2" ]]; then
  [[ "$variant" != control ]] || { echo "Stage A cannot use control mode" >&2; exit 2; }
fi
for value in "$devices" "$per_device_batch" "$seed"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Topology and seed values must be positive integers" >&2; exit 2; }
done
topology_product=$((devices * per_device_batch))
if [[ -z "$gradient_accumulation" ]]; then
  (( 32 % topology_product == 0 )) || { echo "devices*batch must divide global batch 32" >&2; exit 2; }
  gradient_accumulation=$((32 / topology_product))
fi
[[ "$gradient_accumulation" =~ ^[1-9][0-9]*$ ]] || { echo "Gradient accumulation must be positive" >&2; exit 2; }
effective_batch=$((topology_product * gradient_accumulation))
[[ "$effective_batch" == 32 ]] || { echo "Effective global batch must be 32, found $effective_batch" >&2; exit 2; }
# Canonical topology/batch names are exported for the runtime manifest and for
# delegated DLC launchers.  This remains a single-node protocol.
NUM_MACHINES=1
MACHINE_RANK=0
LOCAL_NUM_PROCESSES="$devices"
NUM_PROCESSES="$devices"
PER_DEVICE_BATCH_SIZE="$per_device_batch"
GRADIENT_ACCUMULATION_STEPS="$gradient_accumulation"
TARGET_EFFECTIVE_BATCH_SIZE=32
DRY_RUN="$dry_run"
PREFLIGHT_ONLY="$preflight_only"

branch="$(git -C "$project_root" branch --show-current)"
commit="$(git -C "$project_root" rev-parse HEAD)"
[[ "$branch" == "feature/gp-sq-3d-mix-stage-a-v2" ]] || { echo "Wrong DLC-visible branch: $branch" >&2; exit 2; }
[[ -z "$(git -C "$project_root" status --porcelain)" ]] || { echo "GP Stage-A-v2 runs require a clean DLC-visible worktree" >&2; exit 2; }
[[ -n "$action_checkpoint" && -f "$action_checkpoint" ]] || { echo "ACTION_ONLY_CHECKPOINT is required and must exist" >&2; exit 2; }
[[ -d "$BASE_VLM" ]] || { echo "Missing base VLM: $BASE_VLM" >&2; exit 2; }
for path in "$source_datalist" "$cache_root/vggt_dense/manifest.json" "$stats_root/manifest.json" "$stats_root/gp_sq3dmix_pooled_stats.pt"; do
  [[ -f "$path" ]] || { echo "Missing bound GP input: $path" >&2; exit 2; }
done

training_stage=stage_a
mode="$variant"
logging_frequency=20
case "$phase" in
  smoke)
    datalist="${datalist:-$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_v2_smoke_train_256.json}"
    negative_map_dir="${negative_map_dir:-$negative_root/smoke_train_256}"
    expected_count=256
    max_steps="${max_steps:-100}"
    save_interval="${save_interval:-50}"
    warmup_steps="${warmup_steps:-10}"
    logging_frequency=1
    run_id="${run_id:-gp-sq3dmix-stage-a-v2-smoke-${variant}-${seed}}"
    ;;
  stage_a_v2)
    datalist="${datalist:-$project_root/docs/experiments/splits/gp_sq3dmix_stage_a_v2_train.json}"
    negative_map_dir="${negative_map_dir:-$negative_root/stage_a_v2_train}"
    expected_count=8000
    max_steps="${max_steps:-2000}"
    save_interval="${save_interval:-250}"
    warmup_steps="${warmup_steps:-100}"
    run_id="${run_id:-gp-sq3dmix-stage-a-v2-${variant}-${seed}}"
    ;;
  stage_b)
    training_stage=stage_b
    datalist="${datalist:-$source_datalist}"
    negative_map_dir="${negative_map_dir:-$negative_root/train_full}"
    expected_count=103288
    max_steps="${max_steps:-10000}"
    save_interval="${save_interval:-2000}"
    warmup_steps="${warmup_steps:-0}"
    run_id="${run_id:-gp-sq3dmix-stage-b-${variant}-${seed}}"
    ;;
  stage_c_30k)
    training_stage=stage_b
    datalist="${datalist:-$source_datalist}"
    negative_map_dir="${negative_map_dir:-$negative_root/train_full}"
    expected_count=103288
    max_steps="${max_steps:-30000}"
    save_interval="${save_interval:-5000}"
    warmup_steps="${warmup_steps:-0}"
    run_id="${run_id:-gp-sq3dmix-stage-c-30k-${variant}-${seed}}"
    ;;
  stage_c_100k)
    training_stage=stage_b
    datalist="${datalist:-$source_datalist}"
    negative_map_dir="${negative_map_dir:-$negative_root/train_full}"
    expected_count=103288
    max_steps="${max_steps:-100000}"
    save_interval="${save_interval:-10000}"
    warmup_steps="${warmup_steps:-0}"
    run_id="${run_id:-gp-sq3dmix-stage-c-100k-${variant}-${seed}}"
    ;;
esac
for value in "$max_steps" "$save_interval" "$warmup_steps" "$resume_step"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Step arguments must be non-negative integers" >&2; exit 2; }
done
RUN_ID="$run_id"
MAX_TRAIN_STEPS="$max_steps"
SAVE_INTERVAL="$save_interval"
(( max_steps > 0 && save_interval > 0 )) || { echo "max/save steps must be positive" >&2; exit 2; }
[[ -f "$datalist" ]] || { echo "Missing training datalist: $datalist" >&2; exit 2; }
actual_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$datalist")"
[[ "$actual_count" == "$expected_count" ]] || { echo "Datalist count $actual_count != $expected_count" >&2; exit 2; }
for path in "$negative_map_dir/hard_negative_map.json" "$negative_map_dir/manifest.json"; do
  [[ -f "$path" ]] || { echo "Missing fixed negative-map input: $path" >&2; exit 2; }
done

python - "$negative_map_dir" "$datalist" "$cache_root/vggt_dense/manifest.json" "$source_datalist" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): d.update(block)
 return d.hexdigest()
root=Path(sys.argv[1]); manifest=json.load(open(root/'manifest.json'))
checks={
 'complete': manifest.get('complete') is True,
 'target': manifest.get('target_split_sha256') == sha(sys.argv[2]),
 'cache': manifest.get('dense_cache_manifest_sha256') == sha(sys.argv[3]),
 'source': manifest.get('source_datalist_sha256') == sha(sys.argv[4]),
 'map': manifest.get('map_file_sha256') == sha(root/'hard_negative_map.json'),
 'fallback': float(manifest.get('fallback_rate',1)) <= .01,
 'self': int(manifest.get('self_donor_count',-1)) == 0,
 'same_log': int(manifest.get('same_log_violation_count',-1)) == 0,
}
if not all(checks.values()): raise SystemExit(f'negative-map binding failed: {checks}')
PY
map_commit="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_commit"])' "$negative_map_dir/manifest.json")"
python "$project_root/tools/validate_gp_sq3dmix_code_commit.py" --repo "$project_root" --bound "$map_commit" --current "$commit"

if [[ "$phase" == stage_b || "$phase" == stage_c_30k || "$phase" == stage_c_100k ]]; then
  [[ -n "$decision_report" && -f "$decision_report" ]] || { echo "Matched continuation requires Stage-A-v2 decision report" >&2; exit 2; }
  python - "$decision_report" "$variant" <<'PY'
import json,sys
report=json.load(open(sys.argv[1])); variant=sys.argv[2]
if report.get('all_passed') is not True: raise SystemExit('Stage A all_passed is not true')
selected=report.get('selected_variant')
if variant != 'control' and variant != selected: raise SystemExit(f'variant {variant} is not selected variant {selected}')
PY
  decision_commit="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_commit"])' "$decision_report")"
  python "$project_root/tools/validate_gp_sq3dmix_code_commit.py" --repo "$project_root" --bound "$decision_commit" --current "$commit"
  if [[ "$variant" != control ]]; then
    [[ -n "$stage_a_checkpoint" && -f "$stage_a_checkpoint" ]] || { echo "GP continuation requires the selected Stage-A checkpoint" >&2; exit 2; }
  fi
fi
if [[ "$phase" == stage_c_30k || "$phase" == stage_c_100k ]]; then
  [[ -n "$permission_report" && -f "$permission_report" ]] || { echo "Formal training permission JSON is required" >&2; exit 2; }
  permission_key=formal_30k_allowed
  [[ "$phase" != stage_c_100k ]] || permission_key=formal_100k_allowed
  python - "$permission_report" "$permission_key" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
if report.get(sys.argv[2]) is not True: raise SystemExit(f'{sys.argv[2]} is not true; formal launch forbidden')
PY
  permission_commit="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("code_commit", ""))' "$permission_report")"
  [[ -n "$permission_commit" ]] || { echo "Formal permission has no code_commit" >&2; exit 2; }
  python "$project_root/tools/validate_gp_sq3dmix_code_commit.py" --repo "$project_root" --bound "$permission_commit" --current "$commit"
fi
if [[ "$phase" == stage_c_100k ]]; then
  [[ "$resume_checkpoint" != none && -f "$resume_checkpoint" ]] || { echo "100k extension requires a verified 30k/50k resume checkpoint" >&2; exit 2; }
  (( resume_step == 30000 || resume_step == 50000 )) || { echo "100k extension must resume at step 30000 or 50000" >&2; exit 2; }
  (( max_steps > resume_step )) || { echo "100k extension max steps must exceed resume step" >&2; exit 2; }
fi

cache_enabled=true
if [[ "$variant" == control ]]; then mode=disabled; cache_enabled=false; fi
run_dir="$run_root/$run_id"
[[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite experiment: $run_dir" >&2; exit 2; }
base_config="${TRAIN_CONFIG_YAML:-$project_root/starVLA/config/training/cfg_yaw_1225.yaml}"
overlay="$project_root/starVLA/config/training/gp_sq_3d_mix.yaml"
accelerate_config="${TRAIN_ACCELERATE_CONFIG:-$project_root/starVLA/config/deepseeds/deepspeed_zero2.yaml}"
for path in "$base_config" "$overlay" "$accelerate_config"; do [[ -f "$path" ]] || { echo "Missing config: $path" >&2; exit 2; }; done

visible_devices="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((devices - 1)))}"
IFS=',' read -r -a visible_array <<< "$visible_devices"
[[ "${#visible_array[@]}" == "$devices" ]] || { echo "CUDA_VISIBLE_DEVICES count does not match --devices" >&2; exit 2; }
launch=(accelerate launch --main_process_port "${MAIN_PROCESS_PORT:-29695}" --config_file "$accelerate_config" --num_processes "$devices" --num_machines 1 --machine_rank 0 --mixed_precision bf16)
train=(starVLA/training/train_starvla.py --config_yaml "$base_config" --config_overlay "$overlay"
  --run_root_dir "$run_root" --run_id "$run_id" --seed "$seed"
  --framework.qwenvl.base_vlm "$BASE_VLM" --framework.qwenvl.attn_implementation sdpa
  --framework.gp_sq_3d_mix.mode "$mode" --framework.gp_sq_3d_mix.training.stage "$training_stage"
  --framework.gp_sq_3d_mix.cache.enabled "$cache_enabled"
  --datasets.vla_data.datalist_path "$datalist" --datasets.vla_data.data_root "$DATA_ROOT"
  --datasets.vla_data.split train --datasets.vla_data.expected_sample_count "$expected_count"
  --datasets.vla_data.per_device_batch_size "$per_device_batch" --datasets.vla_data.load_act_data 1
  --datasets.video_data.load_2d_data 0 --datasets.gs_data.load_3d_data 0
  --datasets.reward_data.load_reward_data 0 --w_depth 0 --rgb_query_loss 0 --gs_query_loss 0
  --trainer.pretrained_checkpoint "$action_checkpoint"
  --trainer.gradient_accumulation_steps "$gradient_accumulation"
  --trainer.max_train_steps "$max_steps" --trainer.num_warmup_steps "$warmup_steps"
  --trainer.save_interval "$save_interval" --trainer.logging_frequency "$logging_frequency"
  --trainer.optimizer.weight_decay 1e-3 --framework.action_model.repeated_diffusion_steps 1
  --trainer.resume_ckpt "$resume_checkpoint" --trainer.resume_step "$resume_step"
  --trainer.learning_rate.action_model 1e-6)
if [[ "$cache_enabled" == true ]]; then
  train+=(
    --framework.gp_sq_3d_mix.cache.root "$cache_root"
    --framework.gp_sq_3d_mix.stats.root "$stats_root"
    --framework.gp_sq_3d_mix.stats.source_datalist "$source_datalist"
    --framework.gp_sq_3d_mix.stats.source_cache_manifest "$cache_root/vggt_dense/manifest.json"
    --framework.gp_sq_3d_mix.negative_map.path "$negative_map_dir/hard_negative_map.json"
    --framework.gp_sq_3d_mix.negative_map.manifest "$negative_map_dir/manifest.json"
    --framework.gp_sq_3d_mix.negative_map.source_datalist "$source_datalist"
    --framework.gp_sq_3d_mix.negative_map.source_cache_root "$cache_root"
  )
fi
if [[ "$training_stage" == stage_b ]]; then
  train+=(--trainer.loss_weights.geometry_rank_hard 0.025 --trainer.loss_weights.geometry_rank_spatial 0.025 --trainer.loss_weights.baseline_fidelity 0)
  [[ "$variant" == control ]] || train+=(--trainer.gp_stage_a_checkpoint "$stage_a_checkpoint")
fi
train+=("${extra_args[@]}")

echo "[gp-train-v2] code=$project_root branch=$branch commit=$commit"
echo "[gp-train-v2] phase=$phase variant=$variant mode=$mode seed=$seed run=$run_dir"
echo "[gp-train-v2] topology=1x$devices batch=$per_device_batch accumulation=$gradient_accumulation effective_batch=$effective_batch"
echo "[gp-train-v2] datalist=$datalist negative_map=$negative_map_dir stats=$stats_root"
echo "[gp-train-v2] dry_run=$DRY_RUN preflight_only=$PREFLIGHT_ONLY"
printf '[gp-train-v2] command: CUDA_VISIBLE_DEVICES=%q ' "$visible_devices"
printf '%q ' "${launch[@]}" "${train[@]}"
printf '\n'
(( dry_run )) && exit 0

active_phase=ppu_runtime_preflight
python - "$devices" <<'PY'
import sys,torch
requested=int(sys.argv[1]); available=torch.cuda.device_count()
if available < requested: raise SystemExit(f'requested {requested} accelerator devices, found {available}')
PY
CUDA_VISIBLE_DEVICES="$visible_devices" torchrun --standalone --nnodes="$NUM_MACHINES" \
  --nproc-per-node="$LOCAL_NUM_PROCESSES" "$project_root/tools/check_ppu_runtime.py"
(( preflight_only )) && exit 0

active_phase=output_initialization
mkdir -p "$run_dir"
{
  printf 'branch=%s\ncommit=%s\nphase=%s\nvariant=%s\nmode=%s\nseed=%s\n' "$branch" "$commit" "$phase" "$variant" "$mode" "$seed"
  printf 'devices=%s\nper_device_batch=%s\ngradient_accumulation=%s\neffective_global_batch=%s\n' "$devices" "$per_device_batch" "$gradient_accumulation" "$effective_batch"
  for binding in \
    "datalist:$datalist" \
    "cache_manifest:$cache_root/vggt_dense/manifest.json" \
    "slot_stats_manifest:$stats_root/manifest.json" \
    "negative_map:$negative_map_dir/hard_negative_map.json" \
    "negative_map_manifest:$negative_map_dir/manifest.json" \
    "action_checkpoint:$action_checkpoint"; do
    name="${binding%%:*}"; path="${binding#*:}"
    printf '%s_path=%s\n%s_sha256=%s\n' "$name" "$path" "$name" "$(sha256sum "$path" | cut -d' ' -f1)"
  done
  for optional_binding in \
    "stage_a_checkpoint:$stage_a_checkpoint" \
    "stage_a_decision:$decision_report" \
    "formal_permission:$permission_report" \
    "resume_checkpoint:$resume_checkpoint"; do
    name="${optional_binding%%:*}"; path="${optional_binding#*:}"
    if [[ -n "$path" && "$path" != none ]]; then
      [[ -f "$path" ]] || { echo "Missing optional manifest binding: $path" >&2; exit 2; }
      printf '%s_path=%s\n%s_sha256=%s\n' "$name" "$path" "$name" "$(sha256sum "$path" | cut -d' ' -f1)"
    fi
  done
  printf 'resolved_command='
  printf '%q ' CUDA_VISIBLE_DEVICES="$visible_devices" "${launch[@]}" "${train[@]}"
  printf '\n'
} > "$run_dir/launcher_manifest.txt"
export TRITON_CACHE_DIR="${TRITON_LOCAL_CACHE_ROOT:-/tmp/drivedreamer-triton}/${run_id}/node0"
mkdir -p "$TRITON_CACHE_DIR"
launcher_log="$run_dir/launcher.log"
active_phase=training
CUDA_VISIBLE_DEVICES="$visible_devices" "${launch[@]}" "${train[@]}" 2>&1 | tee "$launcher_log"
