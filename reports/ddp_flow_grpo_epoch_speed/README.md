# Full-epoch acceleration

The active task keeps F's own frozen visual weights, all other original trainable
parameters, full103,288-scene navtrain, G16/K10, global scene batch16, two inner
epochs, full original reference, action SFT replay, and the12912-update schedule.
No learning rate or reward is changed for throughput. Historical BF16 candidate
chunk1/2 failure and previous snapshot failure remain unchanged.

The local ReCogDrive comparison is implementation-specific: its
`recogdrive_agent.py:498` constructs the optimizer from `action_head.parameters()`;
`:297` reads cached `last_hidden_state` when configured for cached features;
`recogdrive_diffusion_planner.py:977–1008` batches fixed denoising transitions.
The current DDP actor instead trains2,233,120,260 parameters, including Qwen.
Its802,967,040-parameter DiT has24 layers and computes in FP32 under the inherited
mixed precision convention. G16/K10/chunk1 causes160 velocity calls per scene
per chain pass. Flow matching itself does not require this serial update layout;
it is currently retained because other batch layouts lack numerical acceptance.

## Code

`velocity_graph.py` captures only the original no-grad velocity kernel. It keeps
candidate/time chunk1, live parameter storage and original dtype/operations.
New observations and timesteps are copied into static inputs, and every returned
output is cloned. Changed parameter storage, shapes or grad-enabled use fail
closed. Trainable forward/backward always uses the original path, including fresh
Qwen encoding. Actor and reference have independent graph objects. No cached
condition or reference head substitution is introduced. The implementation uses
installed PyTorch2.5.1 APIs; no dependencies were upgraded.

Reference: [PyTorch CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/),
which describes reducing repeated CPU kernel-launch overhead with static buffers,
side-stream warmup and graph replay. This is an execution optimization, not a new
RL estimator or mixed precision recipe.

The full-epoch gate now permits world8/accum2 or world16/accum1 only with its own
completed actual-GPU pilot, full-rank dtype/immutability, fixed chain reuse and
zero-tolerance resume proof. An old world8 registration cannot release world16.
Graph-enabled runs also require the four fixed real scenes' exact chain,
velocity/mean/std/log-prob comparisons, live-weight mutation check and unchanged
CUDA RNG; the evidence is bound to the actual kernel and verification script.

`accelerated_epoch_run.py` can consume the existing independently running baseline
producer. It rejects incomplete five-seed or corrupted results, checks the
original evaluation identity, and never starts a duplicate baseline job. The
old evaluation process retains ownership of its six GPUs. New final inference
uses the same single-candidate ODE protocol and predetermined five seeds.

## Measured evidence so far

- CPU regression:291 PASS,5 SKIP, exit0; no full-model FP32 rerun.
- Additional controller control-flow tests:4 PASS, exit0.
- Real four-scene CUDA no-grad probe: PASS, exit0,109.11 seconds. All repeated
  tensor comparisons and generated chains/old log-probs were exactly equal.
  Live parameter update and RNG checks pass. About5.9s eager versus2.74s graph
  per complete scene-chain recomputation. This is not end-to-end training speed.
- Native eight-GPU graph trial: two actual optimizer updates completed,446.45s
  including startup, official reward worker startup, two full saves and audits.
  Completed full numerical update comparison and world16 deployment are recorded
  below; no unsupported profile is called ready based on this document.

World-size changes are not exact resume. Any expanded run starts again from the
original F SFT and qualifies/resumes its own16-rank state. Earlier8-rank weights,
logs and results are preserved; their updates are not counted toward the new epoch.

## Completed production diagnostics and deployment

- Eight-GPU eager/graph updates: PASS at zero tolerance for both update1 and2.
  All27 state files were inspected at each boundary, with1785 and1561 tensor
  leaves respectively; model, Adam, scheduler/controller and fixed-chain tensor
  payloads are identical. This is numerical comparison across code/config changes,
  explicitly not an exact resume from different code.
- Native two-node16-GPU pilot: PASS, exit[0,0],442.44s including startup, two
  complete checkpoints and full-gradient dumps. All16 ranks observe672 trainable
  tensors;6/16 official reward groups have nonzero advantages; constant groups
  remain in training. Initial ratio is exactly1. Second pre-update ratio spans
  [0.9803426266,1.0219960213]; final probe spans[0.9784536362,1.0162166357].
  All own-visual/frozen and independent reference hashes remain unchanged.
- Exact native checkpoint1→2 resume: PASS, exit[0,0],290.29s. All51 state files
  compare exactly at zero tolerance,103.10s CPU comparison. This includes the
  scheduled LR, adaptive KL controller, all rank RNG and pending behavior state.
- FP32 socket all-reduce benchmark: default123ms versus96ms with
  `NCCL_SOCKET_NTHREADS=4,NCCL_NSOCKS_PERTHREAD=4` for200MB,10 measured repeats after
  two warmups. No reduction dtype or NCCL algorithm was changed. The resumed run
  used this socket configuration and remained bitwise identical to the pilot.
  The formal launch uses those same socket settings; this is a transport change,
  not a new numerical precision profile.
- Independent CUDA Adam oracle: PASS for all672 tensors with the actual performed
  update's LR,346.35s. Its initial attempt FAIL is retained in`adam.json.gz`:
  the pre-scheduler observer read saved NEXT-step LR instead of logged`lr_used`.
  All672 first/second moments were already exact;617 master comparisons failed
  with that wrong LR. The observer now verifies`lr_used` against the registered
  schedule and checks saved next LR separately. No tolerances were changed.
  A positive unit fixture initially used a decimal literal different from the
  scheduler's floating-point product; that failure log is also preserved.
- Original-interface export: PASS, exit0,32.03s. See the actual export report.
- Final cluster regression:80 PASS,exit0. Clean checkout preparation and affected
  tests:25 PASS,1 CUDA skip,exit0. Previously recorded291 PASS/5 SKIP was the full
  flow_grpo+cluster suite before the final observer/control additions; unchanged
  full-model CPU FP32 diagnostics were not repeated.

At2026-09-19 00:43UTC the full-budget controller was started (PID1673753), using
vla-zt2 GPU0–7 plus vla-zt3 GPU0–7. rl-zt4's six existing evaluation workers remain
independent; its GPUs6,7 stay reserved. rl-zt2 used onlyGPU0 for bounded observers.
No unrelated local ReCogDrive or CPU work was stopped. The former eight-rank
training was deliberately stopped after logged update30; its last complete
checkpoint was update2. Unsaved updates are not claimed recovered or counted in
the new epoch. All old run directories/checkpoints/logs are retained.

New training starts from its own sixteen-rank original-F pilot update2 and targets
12912 updates, with the unchanged one-epoch data budget and LR/KL/BC recipe.
The native record is`AUTHORIZED_FULL_DATA_EXPERIMENT`, limited to this F profile.
It does not relabel historical chunk>=2 FAIL, old FP32 exceptions or cross-topology
comparisons as PASS. No PDMS/EPDMS improvement is claimed without completed paired
full-navtest results. The formal controller reuses the existing five-seed baseline
producer and schedules the fixed-budget final evaluation on separate GPUs.

The live deployment snapshot at 00:56:44 UTC has reached update 8 with zero reward
errors. Both hosts' eight A800 GPUs are occupied by the verified torchrun jobs,
with approximately 24.7 GB process memory per GPU (20.32 GiB peak PyTorch allocated
memory). Updates 5–6 average 44.72 s/update and updates 7–8 average 52.32 s/update,
charging fresh behavior generation once per two inner updates. The short-window
mean is 48.52 s/update versus the previous eight-GPU run's 74.64 s/update: about
1.54x throughput, not a claimed linear 2x improvement from doubling GPUs. Startup,
checkpoint saving and final evaluation are excluded from these timing estimates.
Extrapolation gives approximately 7.25 compute days for 12,912 updates versus
11.15 days before; the short observed window is insufficient to promise a stable
completion date. See `deployment_snapshot.json` for the individual measurements.

The remaining bottleneck is the grad-enabled serial DiT forward/backward and
distributed synchronization, not official CPU scoring: around updates 5–6,
forward spans are about 10 s, backward/optimizer about 31 s, and official scoring
about 2.6 s overlaps reference computation. These are host spans, not CUDA kernel
profiling, and maxima from different ranks must not be summed as an exact stage
decomposition. No-grad CUDA graph speedup should not be read as a 2.15x speedup of
the complete training update. Efficient batching of fixed-chain gradients would
require a separate numerical qualification; this deployment keeps chunk 1.

Actual actor executable digest:
`081f22e2545b70f180172ff16b1a8329eb558f131323124be593d943b8529673`.
Formal launch commit:`236f81303bbeff87fb36e64042d3d93003554016`.
Environment, processor/config hashes and immutable asset identity are in
`execution_context.json`; F weight SHA256 remains
`9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4`.

## Existing entry points

Working directory:`/mnt/project/DriveDreamer-Policy-epoch-speed`; installed Python:
`/root/miniconda3/envs/ddp/bin/python`. Set`PYTHONPATH=$PWD/navsim:$PWD` and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

```bash
# Read-only current progress; no corpus scan.
cat runs/epoch_speed/world16/experiment_control/progress.json
# Single-GPU four-real-scene graph diagnostic (fresh output path required).
python -m scripts.analysis.cuda_velocity_probe \
  --config configs/flow_grpo/frozen_full_navtrain_epoch1_world16_graph.yaml \
  --bank /mnt/project/DriveDreamer-Policy-full-navtrain/runs/full_navtrain_epoch1/pilot \
  --output /path/to/new-diagnostic-output
# Existing preflight entry; not needed again for the active qualified run.
python -m starVLA.rl.flow_grpo.cli preflight \
  --config configs/flow_grpo/frozen_full_navtrain_epoch1_world16_graph.yaml \
  --output-dir /path/to/new-preflight-output
# Multi-node training / resume share this controller. An active-instance lock
# rejects a duplicate. Use only after the current instance has exited.
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.accelerated_epoch_run \
  --spec runs/epoch_speed/world16/experiment_spec.json
# Export an actual complete boundary; never use an incomplete directory.
python -m starVLA.rl.flow_grpo.cli export \
  --checkpoint runs/epoch_speed/world16/pilot/checkpoints/update_000002 \
  --output-dir /path/to/new-export
# Explicit full navtest single-candidate original protocol, one of the fixed seeds.
python -m starVLA.rl.flow_grpo.cli evaluate \
  --config configs/flow_grpo/frozen_full_navtrain_epoch1_world16_graph.yaml \
  --checkpoint /path/to/completed-export --split navtest \
  --tokens /mnt/project/DriveDreamer-Policy/test_meta.json \
  --data-root /mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset \
  --metric-cache /mnt/project/DriveDreamer-Policy-paired/runs/metric_cache_navtest_v2 \
  --seed 42 --metric-protocol navsim_v2_official_one_stage \
  --output-dir /path/to/new-evaluation
```

The deployed controller already owns the full five-seed parallel evaluation;
these examples should not be used to duplicate its active work. Complete launch,
pilot/resume specs and real output paths accompany this report. No model weights
are committed.
