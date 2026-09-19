# Executed entry points

Run from `/mnt/project/DriveDreamer-Policy-action-batch`, in the existing `ddp`
environment. The launch manifests are in `runs/action_batch/qualified_w8_single/`.
The launcher checks GPU idleness and binds the completed asset receipt; it does
not rescan all images on every invocation. No environment upgrade is needed.

```bash
cd /mnt/project/DriveDreamer-Policy-action-batch
export PATH=/root/miniconda3/envs/ddp/bin:$PATH
export PYTHONPATH="$PWD/navsim:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Read-only source/config/parameter preflight (separate output for a new invocation):

```bash
python -m starVLA.rl.flow_grpo.cli preflight \
  --config configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml \
  --output-dir /tmp/ddp-batched-preflight
```

Actual bounded native eight-GPU diagnostic and native inner-epoch resume:

```bash
python -m scripts.cluster_flow_grpo.cluster run runs/action_batch/qualified_w8_single/pilot_spec.json
python -m scripts.cluster_flow_grpo.cluster run runs/action_batch/qualified_w8_single/resumed_spec.json
```

These two diagnostic destinations are intentionally exclusive and have already
been executed. Use fresh destinations/control ports in copied specs to reproduce,
keeping the same fixed data/order/config. Never delete completed evidence to rerun.

Zero-tolerance comparison and semantic release of completed real evidence:

```bash
python -m scripts.cluster_flow_grpo.boundary_evidence \
  --continuous runs/action_batch/qualified_w8_single/pilot/checkpoints/update_000002 \
  --resumed runs/action_batch/qualified_w8_single/resumed/checkpoints/update_000002 \
  --output runs/action_batch/qualified_w8_single/exact_resume.json \
  --cpu-threads 4 --streaming-load
python -m scripts.analysis.publish_transition_batch \
  --config configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml \
  --root runs/action_batch/qualified_w8_single
```

Full training and restart use the same controller. It selects the latest complete
matching checkpoint; the first invocation resumes the new profile's own update8.
An exclusive controller lock prevents a duplicate launch. This is also the
five-seed full-navtest evaluation entry point: inference and official v1 PDMS/v2
EPDMS scoring execute on the six allocated evaluation GPUs after the full budget.
The already completed, identity-checked SFT baseline is reused.

```bash
python -m scripts.analysis.accelerated_epoch_run \
  --spec runs/action_batch/qualified_w8_single/experiment_spec.json
```

Export and original-interface verification used in this acceptance:

```bash
python -m starVLA.rl.flow_grpo.cli export \
  --checkpoint runs/action_batch/qualified_w8_single/pilot/checkpoints/update_000002 \
  --output-dir runs/action_batch/qualified_w8_single/exported2
python -m scripts.cluster_flow_grpo.cluster run \
  runs/action_batch/qualified_w8_single/export_check_spec.json
```

Regression command (CPU control/math/real official reward tests; separate CUDA
artifacts supply production evidence):

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/flow_grpo tests/cluster_flow_grpo \
  tests/analysis/test_action_batch_probe.py \
  tests/analysis/test_cache_probe_observation.py -q
```
