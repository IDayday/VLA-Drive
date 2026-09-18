# Commands used by the bounded F-only investigation

Run from `/mnt/project/DriveDreamer-Policy-paired`. Use this installed environment;
do not install a new stack. Output directories are transactional: choose fresh
names for a new experiment, retain failures, never overwrite a complete checkpoint.
Only bounded diagnostics are authorized by these configs. They do not release a
100/2000-update production run.

```bash
cd /mnt/project/DriveDreamer-Policy-paired
export PATH=/root/miniconda3/envs/ddp/bin:$PATH
export PYTHONPATH=$PWD/navsim:$PWD
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
git status --short
git log -3 --oneline
nvidia-smi
python -m starVLA.rl.flow_grpo.cli preflight \
  --config configs/flow_grpo/frozen_research_g16_global_batch.yaml
```

The preflight reads model/processor contracts. The launch receipts below explicitly
reuse an already completed full dataset check rather than silently claim a fresh
whole-corpus scan. A source/asset mismatch is a failure, not something to disable.

## CPU / CUDA utility regression

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/flow_grpo tests/cluster_flow_grpo
python -m pytest -q tests/flow_grpo/test_advantages.py \
  tests/flow_grpo/test_diversity.py tests/cluster_flow_grpo/test_pipeline.py \
  tests/cluster_flow_grpo/test_asset_reuse.py
```

## Recreate one bounded G16 arm, with idle-GPU supervision

The recorded specs are real executed specs. Clone to a new job/control/output
directory. This example needs eight idle **local** GPUs; the supervisor refuses
occupied cards. To reproduce group mode, choose its recorded spec instead. Do not
move an eight-GPU spec to rl-zt4/rl-zt2: their user limits remain6/4 respectively.

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.cluster_flow_grpo.cluster import run, write_json
spec = json.loads(Path('reports/ddp_flow_grpo_frozen_improvement/execution/f_g16_global_batch_spec_attempt2.json').read_text())
spec['job_id'] = 'f_g16_global_reproduce'
spec['nodes'][0]['host'] = 'local'
spec['master_port'] = 29810
spec['control_dir'] = str(Path('runs/f_g16_global_reproduce_control').resolve())
spec['entry'] += ['--output-dir', 'runs/f_g16_global_reproduce', '--max-updates', '8']
path = Path('runs/f_g16_global_reproduce_spec.json')
write_json(path, spec)
print(run(path))
PY
```

No untested chunk2 shortcut is used. Both shipped research YAMLs retain chunk1,
world8/accumulation2, scene batch16 and G16. Their default output directories contain
completed evidence, so do not invoke a fresh native train without overriding output.

## Exact inner-epoch resume and comparison

Actual executed specs are `execution/global_resume_first_spec.json` (update1,
inner boundary) and `execution/global_resume_rest_spec.json` (resume to4). For a
new repetition clone both control/output directories and change the latter's
`--resume` value to the new first run's `checkpoints/update_000001`. To resume the
already saved update1 into another output directory, with eight-rank supervision:

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.cluster_flow_grpo.cluster import run, write_json
spec = json.loads(Path('reports/ddp_flow_grpo_frozen_improvement/execution/global_resume_rest_spec.json').read_text())
spec['job_id'] = 'f_g16_resume_reproduce'
spec['master_port'] = 29812
spec['control_dir'] = str(Path('runs/f_g16_resume_reproduce_control').resolve())
entry = spec['entry']
entry[entry.index('--output-dir') + 1] = 'runs/f_g16_resume_reproduce'
path = Path('runs/f_g16_resume_reproduce_spec.json')
write_json(path, spec)
print(run(path))
PY
```

The supervisor preserves world8 and the checked asset receipt. The actual comparison:

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.cluster_flow_grpo.boundary_evidence \
  --continuous runs/frozen_rl_research/f_g16_global_batch/checkpoints/update_000004 \
  --resumed runs/frozen_rl_research/global_resume_rest/checkpoints/update_000004 \
  --output runs/f_g16_resume_comparison_reproduce.json --cpu-threads 8
```

For RL-only optimizer evidence, the executed `execution/global_rl_only_spec.json`
adds `runtime.diagnostic_loss_scope=rl_only` and
`runtime.diagnostic_gradient_statistics=true`, with max_updates/save_every1.
This loss isolation is diagnostic only; do not use it as the production recipe.

## Export / original-interface check

```bash
python -m starVLA.rl.flow_grpo.cli export \
  --checkpoint runs/frozen_rl_research/f_g16_global_batch/checkpoints/update_000008 \
  --output-dir runs/f_g16_global_export_reproduce
CUDA_VISIBLE_DEVICES=0 python -m starVLA.rl.flow_grpo.cli verify-export \
  --config configs/flow_grpo/frozen_research_g16_global_batch.yaml \
  --checkpoint runs/frozen_rl_research/f_g16_global_batch/checkpoints/update_000008 \
  --export-dir runs/f_g16_global_export_reproduce \
  --output-dir runs/f_g16_global_export_check_reproduce
```

## Full fixed-development evaluation, v2 followed by isolated v1 scoring

Original single-candidate ODE, original ten steps, seed42+token, whole-log shards:

```bash
python -m scripts.cluster_flow_grpo.parallel_evaluation \
  --config configs/flow_grpo/frozen_research_g16_global_batch.yaml \
  --checkpoint runs/frozen_rl_research/f_g16_global_batch/export_update8 \
  --output runs/frozen_rl_research/evaluation_g16_global_batch8 \
  --split rl_dev --tokens runs/paired_full_assets_v1/dev_tokens.json \
  --data-root runs/paired_full_assets_v1/dataset \
  --metric-cache runs/paired_full_assets_v1/metric_cache_navtrain_v2 \
  --slots reports/ddp_flow_grpo_frozen_improvement/execution/slots_g16_global_batch.json \
  --seed 42

CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD/navsim_v1.1/navsim:$PWD \
NUPLAN_MAPS_ROOT=/mnt/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
python -m scripts.analysis.paired_dev_pdms_v1 \
  --spec reports/ddp_flow_grpo_frozen_improvement/execution/v1_g16_spec.json \
  --workers 8 --output runs/f_g16_v1_reproduce
```

The first command reuses the complete evaluation if its real identity and files
still match. An identity conflict must be preserved and diagnosed. The second
reuses genuine trusted v1 metric caches, scores every scene and writes a new result.

## Candidate diversity / geometry / v1 reward-order comparison

Actual token shards are `runs/frozen_rl_research/diversity_tokens{0..3}.json`,
derived from the committed64-token manifest before scoring. A single shard:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.analysis.flow_diversity \
  --config configs/flow_grpo/frozen_research_g16_group.yaml \
  --tokens runs/frozen_rl_research/diversity_tokens0.json \
  --output runs/f_diversity_shard0_reproduce --seed 42

python -m scripts.analysis.summarize_diversity \
  --manifest reports/ddp_flow_grpo_frozen_improvement/diversity_manifest.json \
  --shards runs/frozen_rl_research/diversity_shard{0,1,2,3} \
  --output runs/f_diversity_summary_reproduce
python -m scripts.analysis.candidate_geometry \
  --manifest reports/ddp_flow_grpo_frozen_improvement/diversity_manifest.json \
  --shards runs/frozen_rl_research/diversity_shard{0,1,2,3} \
  --output runs/f_candidate_geometry_reproduce

CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD/navsim_v1.1/navsim:$PWD \
NUPLAN_MAPS_ROOT=/mnt/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
python -m scripts.analysis.diversity_pdms_v1 \
  --manifest reports/ddp_flow_grpo_frozen_improvement/diversity_manifest.json \
  --shards runs/frozen_rl_research/diversity_shard{0,1,2,3} \
  --raw-logs /mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval \
  --maps /mnt/navsim/maps --cache-root runs/frozen_rl_research/v1_behavior_cache \
  --workers 8 --output runs/f_diversity_v1_reproduce
```

All paths above are existing code entry points. Use `--help` on
`scripts.analysis.frozen_training_summary` and `scripts.analysis.frozen_paired_results`
to regenerate the training/paired plots from saved results without GPU inference.
No Navtest command was executed in this investigation; no final benchmark result
or READY record is inferred from development scores.
