# Two formal Lite runs

The user explicitly authorized two 16-GPU runs after code/real-smoke gates, on four
8-GPU servers. Run A is BaseInit; run B is Driving-VQAInit. No no-WM experiment or
old M0/epoch33 warm-start is substituted. These are fresh full-planning-stack runs.

Both use 103,288 trainval scenes, 27 fixed dataset epochs, single front camera,
long-2, 64x8x3 proposals and unchanged exact scorer. Train/val log lists remain
disjoint before final-fit concatenation. No internal validation/best checkpoint
selection; epoch27 is the registered final checkpoint. No Navtest labels enter
the train-only probe. Base/VQA backbone pretraining log exposure is unknown and
is disclosed, not called an unseen representation evaluation.

The Lite shared init is a NEW independent artifact, not the V1.1 artifact:
`/mnt/project/DriveVLA-M0-formal-runs/task_future_lite_20260905/shared_task_future_lite_seed0_v2.pt`
SHA-256: `14250f9a38edd84a1d7335c87836bb430674ae4147bf1c13b4ac8bfc93e51078`.
The two real VLM instantiations had 468 bitwise-identical effective trainable
tensors (21,249,830 parameters). The artifact additionally retains dormant legacy
head keys for compatible loading. No planning modules are loaded from either VLM.

VLM initialization paths / checkpoint fingerprints:

- Base: `/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-base-aligned`,
  `0bd7cfa0ab23300304dd627abb09abbdc38748c8c8ff6c3209baf73a81fb421f`.
- Driving-VQA: `/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-driving-vqa-dense`,
  `79fb39297e322cd2d3dc68d4f23b86ff85806b336a4d5d0ed5db9b66e4034a3c`.

The supplied VQA path is already an audited dense standalone initialization; no
additional adapter merge is performed in this task. Fresh PlanReg Q/V LoRA is
loaded from the same shared init after either VLM. EMA is created afterwards.

## Data and cache

The new, separate `input_cache_v2c` contains 103,288 input-only records with the
single-front prompt, input IDs/mask, ego/command, current/future image paths,
GT/long trajectories, tile metadata, actual logged future poses and map name.
No dynamic VLM/register/EMA outputs are cached in formal training. Consolidated
schema is v2; target name is `trajectory_target_task_future_lite_v1`.

102,865 scenes have all three future frames valid; 423 have at least one invalid
future horizon. Invalid future observations stay masked, not duplicated or marked
safe. They still receive valid current physical supervision. Successful build
means all required source records were read; it does not mean missing horizons
were fabricated. Failed earlier new-root build attempts remain untouched and are
not formal inputs. No old cache was overwritten.

## GB64 configuration to be locked from real throughput

Each run uses two nodes x eight GPUs x local batch4, accumulation1. Padding to
global batch gives 1,614 steps/epoch and 43,578 steps/27 epochs (8 repeated padding
samples per epoch). The actual dataset/sampler audit must confirm this at startup.

| Logical group | Peak LR | Matrix WD / no-decay WD |
|---|---:|---:|
| planning/readout | 2e-4 | .01 / 0 |
| fusion | 2e-4 | .01 / 0 |
| generator | 2e-4 | .01 / 0 |
| scorer | 2e-4 | .01 / 0 |
| physical decoder (future_predictor group) | 1e-4 | .01 / 0 |
| semantic Q-Former | 1e-4 | .01 / 0 |
| visual Q/V LoRA | 3e-5 | 0 / 0 |
| base VLM / LLM | Frozen | None |

AdamW betas .9/.999, eps1e-8; biases/norms/queries/embeddings/gates/all LoRA have
zero decay. Clip global norm1. Five-percent warmup starts at 1% of peak, then
cosine to 10% of peak; step scheduler is resume-safe. No implicit batch LR scaling
beyond these explicit V1.1 proposed GB64 values. WM remains on from step0,
lambda .01 -> .10 over the first 10% of optimizer steps. It is not increased using
small-probe loss ratios. EMA start/end are .996^4 and .9999^4 at GB64, with the
existing sample-normalized cosine schedule and FP32 master accumulation.

## Launch (requires validated shared lock)

On `training-vla-zt` (peer `training-vla-zt2`):

```bash
LAUNCH_FORMAL=1 PLANREG_PROTOCOL_VERSION=task_future_lite \
PLANREG_LAYOUT_LOCK=/absolute/formal_training_layout_lock.json \
PLANREG_SHARED_INIT=/absolute/shared_task_future_lite_seed0_v2.pt \
PLANREG_INPUT_CACHE=/absolute/input_cache_v2c \
PLANREG_FORMAL_RUN_ROOT=/absolute/new_formal_runs \
PLANREG_MASTER_PORT=29630 \
bash local_planreg_wm_v1/train_formal_task_future_lite_base.sh 0
```

On `training-vla-zt3` (peer `training-rl-zt4`), use the same inputs/lock and:

```bash
LAUNCH_FORMAL=1 PLANREG_LAYOUT_LOCK=/absolute/formal_training_layout_lock.json \
PLANREG_SHARED_INIT=/absolute/shared_task_future_lite_seed0_v2.pt \
PLANREG_INPUT_CACHE=/absolute/input_cache_v2c \
PLANREG_FORMAL_RUN_ROOT=/absolute/new_formal_runs \
PLANREG_MASTER_PORT=29640 \
bash local_planreg_wm_v1/train_formal_task_future_lite_vqa.sh 0
```

Launchers record runtime, source commit, resolved paired configs and parameter
groups, refuse stale artifacts, and only accept an explicit same-run resume path.
Do not claim training started merely from a shell PID; record first finite steps
from both ranks/node groups in EXECUTION_STATUS. Final export strips all training
auxiliaries; identical current-only Navtest evaluation is performed only after
the fixed final epoch, not to select the training epoch.
