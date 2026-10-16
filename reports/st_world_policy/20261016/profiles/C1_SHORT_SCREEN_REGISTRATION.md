# Actual C1 sixteen-update registration, prepared but not launched

This supersedes the missing-source/missing-manifest part of the earlier proposal; it does not retrospectively change any previous smoke/profile result. Systems dispatched no training, official scoring or GPU model load during this preparation.

Runtime: `/mnt/project/VLA-Drive-st-master-integration-runtime-20261016`, clean source `2d3e4f5a99c248d5ff0b66791952e2d28def1d8b`; execution fingerprint `66b468ca5a47d0c86aa894c696d6370c185aba649dd849263df3b75273402370`.

Actual registration:

`/mnt/project/st-world-policy-20261016/profiles/flow_C1_profile64_8gpu_screen_v1/REGISTRATION.json`

SHA256: `05803beceb15e21ce71beb50905c18238d52e7fba4ac7b7a748027ffca89d3c3`.

Real train manifest: `/mnt/project/st-world-policy-20261016/audits/flow_policy/profile64_fp32_master_v1/TRAINING_MANIFEST.json`, identity `2aafa3368bb78c7e814f6ac37ce047093ebd9d483f48ee1fd74ed89a622bd04a`. It contains64 scenes/59 logs, selected with seed42 before scores, current observations plus original GT and canonical official contexts. Its preparation made zero official calls. Base binding is unchanged: `3b910a6d4245580d77f868e4e0af5faf2fc7fccc25c561c935f219ab1bf7f099`.

The actual config, target contract, environment, lease, command and source fingerprint are now pinned. Local and Tsinghua environment records match exactly. Strict registration checks return `DIAGNOSTIC_ONLY`; remote preflight also verified the same canonical score protocol, maps and metric roots without loading a model or running an official score. Running the metadata builder twice preserves exactly the same registration hash.

## Fixed work and resource bound

C1,8 GPUs, global scene batch32, microbatch1, accumulation4, MC chunk5. Original4 rollouts/scene,10 Euler steps, MC50 old and current, BC repeat8 and weight1; no loss/task removal. Original4096-update cosine, warmup100,lr1e-5 to5e-7; this invocation stops at16 new updates. Official scoring has4 global CPU workers,2 ordered loader threads/rank, OMP and all BLAS environment limits1; the existing runner explicitly sets Torch intra-op threads2. Primary score calls2,048, maximum4,096 including one bounded retry. Runtime cap1,800s, outer timeout2,100s with120s cleanup allowance. Initial/final native saves remain enabled; no periodic save in these16 steps (`save_every=512`).

The lease contains two **alternative single-node placements**, each with all8 exact GPU UUIDs:

1. `training-vla-zt-worker-0`, preferred only after protected W/C1 fresh-master qualification exits.
2. `training-vlawm-tsinghua-zt-worker-0`, only after Lead coordinates the other real FM evaluation controller. Its SP0 final teacher verification uses2 CPU threads and remains protected.

This is not a16-GPU/multinode run. Both placements use one shared run/output identity and native `RUN.lock`; exactly one may start. At the22:56UTC resource observation, local GPUs0/1/2 were still occupied by the protected qualification, while Tsinghua GPUs were empty. Empty GPUs did not erase the other real queued controller. No lease snapshot substitutes for a fresh launch-time check. Systems did not signal any process.

Lease validity is finite and recorded in `RESOURCE_LEASE.json`; it must not be silently extended after expiry. Lead must verify successful real-model qualification, current policy/host, all8 physical GPUs, absence of protected occupants, existing output identity and all8 GPU flocks before executing. The subsequent240-update long profile is **not** registered or authorized by this short-screen command.

## Actual command, for Lead only after the preceding checks

The machine-readable canonical command is the adjacent `COMMAND.json`. On exactly one lease host:

```bash
cd /mnt/project/VLA-Drive-st-master-integration-runtime-20261016
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
flock -n /tmp/foresight_gpu_0.lock flock -n /tmp/foresight_gpu_1.lock flock -n /tmp/foresight_gpu_2.lock flock -n /tmp/foresight_gpu_3.lock flock -n /tmp/foresight_gpu_4.lock flock -n /tmp/foresight_gpu_5.lock flock -n /tmp/foresight_gpu_6.lock flock -n /tmp/foresight_gpu_7.lock \
timeout --signal=TERM --kill-after=120 2100 /root/miniconda3/envs/ddp/bin/python -m torch.distributed.run --nproc_per_node=8 --master_port=29670 -m tools.st_world_policy.train_flow_policy train --registration /mnt/project/st-world-policy-20261016/profiles/flow_C1_profile64_8gpu_screen_v1/REGISTRATION.json
```

Output: `/mnt/project/st-world-policy-20261016/profiles/flow_C1_profile64_8gpu_screen_v1/runs/C1_profile64_8gpu_screen_v1`.

Explicit interrupted resume uses the identical command plus `--resume`; a previous operator STOP additionally needs `--acknowledge-stop`. It restores this run's complete native boundary and does not reset its cursor/RNG/4096LR. It does not turn the16-step cap into240 or initialize formal training from PROFILE weights. Do not run resume merely to produce another speed sample.

Existing per-rank region timings distinguish data, shared encoder, rollout/snapshot, official feedback, old CFM and current BC/FPO/backward/optimizer. They omit final all-gather/logging from their inner-update field; record an external completion/launch wall separately, plus native save time. Sixteen steps are a cost/finite-gradient/feedback-capacity screen, not20+100 steady-window qualification. Real chunk5 nonzero advantage, complete official feedback, parameter updates and native recovery remain to be measured; the previous tiny two-rank FusedAdam test is not a substitute.
