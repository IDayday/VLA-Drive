# Real action-head batching and checkpoint memory experiment

Status: **EXPERIMENTAL_NOT_READY**. The production16GPU run is unchanged.
The experiment answers the memory/throughput question; it is not evidence of
better PDMS/EPDMS or a qualified replacement optimizer configuration.

On training-rl-zt2, four previously idle A80080GB devices were used, within the
user's four-GPU limit. Original F weights, full819,503,620-parameter action head,
G16/K10, frozen prefix features, fixed saved chains, old log-probs, official
advantages and reference statistics were retained. Two predeclared scenes
(original behavior positions1,2, both with nonzero official advantages) were each
run twice, without optimizer updates, candidate replacement or resampling.

| Computation | Checkpointing | Mean forward + backward | Peak allocated |
|---|---|---:|---:|
| 160 serial calls | on | 25.714s | 10.140GiB |
| 160 serial calls | off | OOM during first forward | 76.273GiB before failure |
| 16 candidates per call,10 calls | off | 1.917s | 38.957GiB |
| 160 saved transitions in one call | off | 0.519s | 11.756GiB |
| 160 saved transitions in one call | on | 0.617s | 11.742GiB |

An additional same-flat160-layout checkpoint-on run shows that most of this
speedup comes from batching, not merely the checkpointing toggle. Timings are
short microbenchmarks on the same GPU model, not an end-to-end speed guarantee.

These are synchronized action RL+reference objective timings, including condition
projection, **excluding** replay, optimizer states/updates, communication,
rollout and CPU scoring. They must not be substituted for the production40s/step
measurement or used to promise an equivalent end-to-end speedup. The probe uses
ordinary PyTorch BF16 leaf-gradient accumulation; production uses ZeRO FP32
partition accumulation. The complete trainer's memory has not been qualified for
the new batch layout.

In rollout, the16 candidates are independent and can be batched, but the10 time
steps depend on preceding states. During update the entire saved chain is fixed
data: all160 evaluations of v(x_t,t,condition) are independent. Flattening those
axes preserves each candidate, time bucket, x_next and old probability. The
current serial implementation retains many graphs until one backward call;
without checkpointing, repeated FP32 conversions and saved tensors exhaust
memory. A larger batch can use less memory because it shares one action-network
call rather than retaining160 separate calls. No performance-critical settings
were silently changed in the running trainer.

## Numerical evidence

CPU packing tests exercise the actual probe implementation and original rollout
implementation, across one/two scenes, both SDE modes, and checkpointing on/off:
**8 PASS, exit0**. Outputs agree exactly in these fixtures; the time-dependent
test head and different scene conditions expose packing/bucket errors. These are
small mathematical/control tests, not real-model production acceptance.

The real CUDA probe computes every one of359 action parameter gradients. Complete
comparison reports are compressed losslessly in this directory. Existing
atol2e-6/rtol2e-3 were retained:

| Layout / scene | Failed tensors | Whole-model relative L2 | Cosine |
|---|---:|---:|---:|
| candidate16 /1 | 196/359 | 0.0178183 | 0.9998412 |
| flat160 /1 | 193/359 | 0.0172152 | 0.9998518 |
| candidate16 /2 | 241/359 | 0.0175868 | 0.9998454 |
| flat160 /2 | 232/359 | 0.0169924 | 0.9998557 |

Thus the BF16 leaf-gradient comparison is **FAIL**, even though gradients are
finite and the computation is fast. This is not a comparison of the production
optimizer's actual FP32 accumulated gradient. Native accumulation, high-precision
real-model gradients, Adam updates, fixed-noise outputs and exact resume still
need independent qualification before enabling the layout in formal training.
The historical chunk1/2 FAIL is unchanged. No tolerance or gate was relaxed.

The initial four attempts failed before measurement because the probe bound its
feature identity after model construction, unlike the producer/trainer. The
corrected probe binds it before construction. Those logs are preserved. All three
completed measurement processes exited0; serial_off exited1 from actual CUDA OOM.
The existing full307-test production suite was not repeated: no production source,
configuration or test under tests/flow_grpo was modified by this experiment.

For the same flat160 layout, checkpointing on/off produced exactly identical
transition tensors and all359 leaf gradients on both fixed scenes (PASS). This
checks the toggle within that layout; it does not change the cross-layout FAIL.

## Reference settings and what they imply

Pinned sources and fetched-file hashes are in `reference_lock.json`.

- [Flow-GRPO config](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/config/grpo.py):
  SD3 examples train with10 denoising steps and evaluate with40; FLUX examples use
  6/28. The image examples batch candidates. These are model/task-specific recipes.
- [Flow-GRPO base config](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/config/base.py):
  `timestep_fraction` supports training on a subset of the sampled chain, with
  a policy-gradient accuracy tradeoff. This is different from reducing the number
  of actual rollout steps. The present DDP run retains all10 transitions.
- [RLinf pi0.5 ManiSkill PPO](https://github.com/RLinf/RLinf/blob/3f71bda5fed2f40f27f8c4f238e1a29213085299/examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml):
  num_steps4, microbatch32, gradient_checkpointingFalse. It uses a different
  action model/flow-noise PPO recipe, so these settings do not certify DDP memory
  or few-step policy quality.
- [ReinFlow reproduction](https://github.com/ReinFlow/ReinFlow/blob/e722e151bed767f3ffef47527cf697f2358af55d/docs/ReproduceExps.md)
  and [project results](https://reinflow.github.io/) include one/few-step training,
  including shortcut policies and a learned exploration-noise network. Neither
  its short-step success nor its algorithm is silently imported into DDP.

For DDP, K5/6 training with the original10-step evaluation is a plausible separate
experiment, not an equivalence-preserving flag. It changes the sampled chain,
time buckets, SDE discretization and distributions. Old/current/reference must
use the new consistent step grid and variance; an old10-step buffer must not be
relabeled. Uniformly subsampling5 of10 fixed transitions is another option that
retains the rollout but changes the gradient estimator variance/normalization.
Full batching keeps all candidates/transitions and is therefore the first
engineering route to qualify before reducing K or model adaptation capacity.

## Reproduce the diagnostic

Use fresh output/control directories rather than overwrite existing evidence.

```bash
cd /mnt/project/DriveDreamer-Policy-action-rl
export PYTHONPATH="$PWD/navsim:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/root/miniconda3/envs/ddp/bin/python
CUDA_VISIBLE_DEVICES='' "$PY" -m pytest tests/analysis/test_action_batch_probe.py -q
# On an explicitly allocated idle GPU; this is a diagnostic, not train/resume.
"$PY" -m scripts.analysis.action_batch_probe --config configs/flow_grpo/frozen_action_head_epoch1.yaml --bank runs/action_head/world16/pilot --mode flat160_off --output runs/action_head/compute_probe_reproduction/flat160_off
```

The full feature precompute producer also completed. Its first CPU finalize
continuation lacked the producer's deterministic environment variables and was
correctly rejected. The continuation recipe is corrected; a retry with the same
locked identity is running. This does not affect valid read-through entries or
change the ongoing training inputs.
