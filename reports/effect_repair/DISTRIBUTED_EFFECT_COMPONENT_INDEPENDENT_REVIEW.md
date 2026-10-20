# Independent DDP effect-gradient component review

Target: `tools/action_effect/qualify_distributed_effect.py` at `8a0ddcb6a5081876ef55a2e4b948cb0844bcca42`. Scope is the real-pair two-GPU NCCL component, with no optimizer. The reviewer ran no GPU, native simulator or real training.

Decision: **PASS to proceed with the separate bounded two-GPU NCCL component qualification**. This is not a completed NCCL result, a qualified production DDP trainer or a CUDA exact-resume claim.

The source fixes eight actual fitting scenes before calculation, splits four per rank and two per local microbatch, and includes a case with every supervised mask on rank1 set empty. Six named task numerators/denominators are combined before distributed reduction. Denominator all-reduces retain physical ranking-gap mass and compensate for DDP gradient averaging with `world_size/global_denominator`. Each rank's gradient is compared against a separately initialized same-seed full-eight-scene reference. The declared CUDA tolerance is `rtol=2e-4, atol=2e-6`, fixed before execution.

## Suspected multi-forward reducer defect

The code performs two DDP forwards, combines both graphs and invokes one backward with `find_unused_parameters=True`. This pattern can be unsafe when parameter-use sets differ. It was not acceptable to label it broken or safe merely from that generic warning.

The independent CPU/Gloo test uses the actual `ActionEffectModel` and production contribution/reduction functions, two spawned ranks, the same four-local/two-micro/eight-global pattern, the delta-feedback mode, all six losses and both all-valid/empty-rank cases. It tests the actual structured model graph rather than a one-layer imitation. Context/pair tensors are explicitly synthetic; no real data integration is claimed.

Executed:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q -s \
  tests/action_effect/test_distributed_effect_independent_review.py
```

Actual result: **1 passed in2.85s**. Both worker processes exit cleanly. No mark-ready-twice error or deadlock occurred. Each rank agrees with the complete reference for44 gradient tensors:

| Case | Rank0 maximum absolute gradient difference | Rank1 maximum absolute gradient difference |
|---|---:|---:|
| All valid | 6.51925802230835e-08 | 6.51925802230835e-08 |
| Rank1 entirely empty labels | 1.7136335372924805e-07 | 1.7136335372924805e-07 |

The `find_unused_parameters=True` warning reports no unused parameters in this tested delta-feedback graph. That explains why this concrete pattern succeeds. It is not permission to turn the flag off in untested model modes or dynamic branches. CPU/Gloo numerical agreement does not imply NCCL/backend numerical agreement; retain the real qualification and its registered tolerance.

## Dispatch and scope boundaries

- Use exactly two explicitly allocated visible GPUs and one BLAS/Torch CPU thread per process. Do not dispatch from this review agent.
- Preserve the verified inference/training FP32 and TF32-off settings. Because deterministic algorithms are enabled, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, as in the campaign's already qualified launcher. This environment requirement is not exercised by a CPU test.
- Use actual immutable PairData and matching contexts from the approved TRAIN population. The script executes no optimizer and no simulator; it should report that narrow scope.
- A process/parse failure must make the torchrun job fail rather than allowing a missing-rank PASS. Keep dispatch-level wall limits and normal fail-fast process-group handling.
- Other loss modes, variable parameter-use branches, production sampler/resume, actual NCCL transport and GPU resource/timing are outside this CPU result.

No implementation change to the runner is required by the observed CPU result. If the real NCCL run reveals a reducer error, a valid fallback is sequential microbatch backward with previously accumulated global denominators; do not change the loss target or omit empty-rank collectives to hide it.
