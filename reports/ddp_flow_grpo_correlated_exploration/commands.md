# Actual bounded entry points

Working directory `/mnt/project/DriveDreamer-Policy-paired`. These commands use
the installed environment. Do not overwrite completed outputs. The recorded
specs in `execution/` are the actual executions; clone job/control/output names
for a new repetition. Their receipts reuse the previously completed asset check
with an explicit source rebind, not a new full-corpus scan or GPU release.

```bash
export PATH=/root/miniconda3/envs/ddp/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$PWD/navsim:$PWD
git status --short
nvidia-smi
python -m starVLA.rl.flow_grpo.cli preflight \
  --config configs/flow_grpo/frozen_research_g16_euler_correlated.yaml

CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/flow_grpo tests/cluster_flow_grpo
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/flow_grpo/test_temporal_exploration.py tests/flow_grpo/test_math.py \
  tests/flow_grpo/test_actor_replay_weighting.py tests/flow_grpo/test_inner_boundary.py
```

The native `train` rejects these experimental samplers outside diagnostic mode;
each diagnostic is capped at8 optimizer updates. Retain candidate_chunk_size1.
The zero-correlation default remains the original Flow SDE.

## Real-model oracle, sampling and training

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.analysis.correlated_oracle \
  --config configs/flow_grpo/frozen_research_g16_euler_correlated.yaml \
  --manifest reports/ddp_flow_grpo_correlated_exploration/results/manifest.json \
  --output runs/f_euler_oracle_reproduce
CUDA_VISIBLE_DEVICES=0 python -m scripts.analysis.flow_diversity \
  --config configs/flow_grpo/frozen_research_g16_group.yaml \
  --tokens runs/frozen_rl_research/diversity_tokens0.json \
  --output runs/f_exploration_shard0_reproduce \
  --noise-levels .1 --temporal-correlations .8 \
  --transition-modes flow_sde euler_gaussian
```

The executed oracle/calibration specs preserve all exact CLI arguments, scene
tokens, noise modes and allocated devices. The generic supervisor takes a **path**
to a JSON spec, not the parsed dictionary. Example fresh noisy-Euler8 execution:

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.cluster_flow_grpo.cluster import run, write_json
spec = json.loads(Path('reports/ddp_flow_grpo_correlated_exploration/execution/euler_correlated_spec.json').read_text())
spec['job_id'] = 'f_euler_reproduce'
spec['nodes'][0]['host'] = 'local'  # requires eight idle local GPUs
spec['master_addr'] = '127.0.0.1'
spec['master_port'] = 29820
spec['control_dir'] = str(Path('runs/f_euler_reproduce_control').resolve())
entry = spec['entry']
entry[entry.index('--output-dir')+1] = 'runs/f_euler_reproduce'
path = Path('runs/f_euler_reproduce_spec.json')
write_json(path, spec)
print(run(path))
PY
```

Do not reuse these eight-GPU allocations on rl-zt4/rl-zt2: their limits remain6/4.
Actual main runs used local8 and training-vla-zt2's8 idle GPUs. Existing pressure
jobs on other machines were untouched. No U or long-training job was started.

## Inner-epoch resume and original-interface export

The executed `euler_resume_first_spec.json` saves after update1, inside the
two-inner-epoch behavior batch. `euler_resume_rest_spec.json` resumes that same
chain/advantages to update4. To repeat, clone both outputs and update the latter's
`--resume` argument. Exact comparison used:

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.cluster_flow_grpo.boundary_evidence \
  --continuous runs/correlated_exploration/f_euler_correlated8/checkpoints/update_000004 \
  --resumed runs/correlated_exploration/euler_resume_rest/checkpoints/update_000004 \
  --output runs/f_euler_boundary_reproduce.json --cpu-threads 8

python -m starVLA.rl.flow_grpo.cli export \
  --checkpoint runs/correlated_exploration/f_euler_correlated8/checkpoints/update_000008 \
  --output-dir runs/f_euler_export_reproduce
CUDA_VISIBLE_DEVICES=0 python -m starVLA.rl.flow_grpo.cli verify-export \
  --config configs/flow_grpo/frozen_research_g16_euler_correlated.yaml \
  --checkpoint runs/correlated_exploration/f_euler_correlated8/checkpoints/update_000008 \
  --export-dir runs/f_euler_export_reproduce \
  --output-dir runs/f_euler_verify_reproduce
```

`euler_rl_only_spec.json` uses one actual update with official rewards, diagnostic
loss isolation and post-reduction optimizer-gradient capture. It does not replace
the RL+KL+SFT recipe of the main bounded runs.

## Development evaluation, v2 and isolated v1

```bash
python -m scripts.cluster_flow_grpo.parallel_evaluation \
  --config configs/flow_grpo/frozen_research_g16_euler_correlated.yaml \
  --checkpoint runs/correlated_exploration/f_euler_correlated8/export_update8 \
  --output runs/correlated_exploration/evaluation_euler8 \
  --split rl_dev --tokens runs/paired_full_assets_v1/dev_tokens.json \
  --data-root runs/paired_full_assets_v1/dataset \
  --metric-cache runs/paired_full_assets_v1/metric_cache_navtrain_v2 \
  --slots reports/ddp_flow_grpo_correlated_exploration/execution/euler_slots.json \
  --seed 42

CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD/navsim_v1.1/navsim:$PWD \
NUPLAN_MAPS_ROOT=/mnt/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
python -m scripts.analysis.paired_dev_pdms_v1 \
  --spec reports/ddp_flow_grpo_correlated_exploration/execution/v1_dev_spec.json \
  --workers 8 --output runs/f_correlated_v1_reproduce
```

The first command validates/reuses the complete transaction. Different inputs
at the same output path fail. v1 runs in its own Python process/import path and
rescores the identical single-candidate trajectories using genuine v1 caches.
No full Navtest/five-seed result was run or claimed in this bounded investigation.

## Saved bank integrity

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.compare_exploration \
  --manifest runs/correlated_exploration/manifest.json \
  --shards runs/frozen_rl_research/diversity_shard{0,1,2,3} \
    runs/correlated_exploration/calibration_shard{0,1,2,3} \
    runs/correlated_exploration/euler_calibration_shard{0,1,2,3} \
  --output runs/f_exploration_integrity_reproduce.json
```

This validates all saved candidates, including unfavorable ones, recomputes the
stored diversity measures and checks all overlapping banks exactly. It is not a
production acceptance gate and does not convert a historical FAIL to PASS.
