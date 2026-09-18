# Reuse the next inner forward's full ratio observation

`runtime.reuse_inner_probe=true` removes the redundant no-grad forward after an
intermediate inner update. The following inner epoch already recomputes the
current policy on every saved candidate/transition at precisely those updated
weights. Its pre-update ratios are recorded as `previous_update_probe`, with
the actual optimizer update and behavior version. No current-policy condition,
gradient or old probability is cached. All G16/K10 candidates remain present.

The earlier log row explicitly says `DEFERRED_TO_NEXT_INNER_FORWARD`, without
inventing post-update values. The next row publishes the complete observation.
The final update of every invocation always performs an explicit full probe,
even if the invocation ends mid behavior batch. Resume derives the preceding
update identity from the restored update/inner cursor; it needs no invented RNG
state or lost external pending record.

Actual results:

- Targeted production-function/metrics tests:31 passed, exit0.
- Full affected CPU regression:265 passed,4 skipped, exit0. GPU tests are separate.
- F/G16/K10/global scenes16,4GPUs/accumulation4, BF16/ZeRO2 FP32 partitions,
  gamma0.6 raw, checkpointing on: two real updates, exit0.
- Same initial complete behavior bank, original advantages and reference
  statistics; every actual optimizer gradient (672 tensors at each update),
  Adam moment/master tensor and forward weight is exactly equal to the earlier
  overlap-enabled run. Zero tolerance, no relaxed historical test.
- Reused complete ratios (min/max/mean/count/clip fraction/global quantiles) are
  exactly equal to the original explicit post-update forward at update1.
- Own F visual/full reference/frozen parameters are unchanged.
- Inner-boundary update1→2 resume on rl-zt2, after continuous training on rl-zt4:
  actual exit0;14 stored checkpoint/RNG/pending files compare exactly.

The two measured update spans total293.917 seconds before this change and263.110
seconds after it, about10.5% shorter. The removed intermediate full probe cost
24.61 seconds; remaining differences include host/timing variation. These spans
include the diagnostic gradient observer (~11 seconds/update), exclude fresh
rollout/reference costs and checkpoint publication, and are not an end-to-end
formal-training throughput guarantee. Whole two-update jobs were586.59 versus
551.55 seconds including startup and checkpoints. Both hosts use the same GPU
type but are different physical servers. The complete per-phase data is retained
in `summary.json` and `optimizer_equivalence.json`.

This is **diagnostic-only qualification**, not a production acceptance record.
The historical BF16 chunk1/2 FAIL remains unchanged. A real two-node8-GPU run
with both runtime flags is separately underway under `runs/fast_world8`; no
4-GPU observation is relabeled as8-GPU evidence. Current64-update performance
experiments retain their original source and settings.

Code was introduced in `432cfb5`; the complete native execution identity and
resolved config are in `candidate/execution_context.json` and
`candidate/rl_config.json`. No weights are committed. Runtime changes are in
`starVLA/rl/flow_grpo/{probe_reuse,trainer,config}.py`; actual comparison uses
`scripts/analysis/checkpoint_efficiency.py`.

Commands from `/mnt/project/DriveDreamer-Policy-speed`:

```bash
export PYTHONPATH="$PWD/navsim:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/root/miniconda3/envs/ddp/bin/python
# Completed specs: preserve these outputs; give any new run fresh directories.
$PY -m scripts.cluster_flow_grpo.cluster run runs/probe_reuse/candidate_spec.json
$PY -m scripts.cluster_flow_grpo.cluster run runs/probe_reuse/resume_spec.json
$PY -m scripts.analysis.checkpoint_efficiency \
  --on /mnt/project/DriveDreamer-Policy-perf/runs/reward_overlap/overlap \
  --off runs/probe_reuse/candidate --difference reuse_inner_probe \
  --output /tmp/new_probe_optimizer_comparison.json
$PY -m scripts.cluster_flow_grpo.boundary_evidence \
  --continuous runs/probe_reuse/candidate/checkpoints/update_000002 \
  --resumed runs/probe_reuse/resumed/checkpoints/update_000002 \
  --output /tmp/new_probe_resume_comparison.json --cpu-threads 4 --streaming-load
```
