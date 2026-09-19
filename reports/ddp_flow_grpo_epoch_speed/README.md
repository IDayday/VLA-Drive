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
  Full numerical update comparison and world16 deployment are recorded below
  when complete; no unsupported profile is called ready based on this document.

World-size changes are not exact resume. Any expanded run starts again from the
original F SFT and qualifies/resumes its own16-rank state. Earlier8-rank weights,
logs and results are preserved; their updates are not counted toward the new epoch.
