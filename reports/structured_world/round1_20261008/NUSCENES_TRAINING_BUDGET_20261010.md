# nuScenes ego conditioning and training-budget audit

Our model already consumes legal ego motion. Its 24-epoch endpoint comes from
the user's round-one registration, section 17, rather than a paper-derived
schedule. External scores are useful context but do not isolate a method gap:
motion inputs, perception initialization, temporal input, optimizer and exposure
all differ. This audit changes neither a frozen run nor the evaluator and
launches no additional nuScenes training.

## Actual ego-conditioning path

`planning_metadata` in
`starVLA/dataloader/structured_world/nuscenes_adapter.py` requires a previous
pose and uses only previous/current LiDAR sensor poses. It converts the past
displacement to internal forward/left coordinates and scales it to a nominal
0.5-second interval using the actual past timestamp difference. The four state
channels are normalized dx/dy and sine/cosine of the corresponding yaw change.
These are past-motion proxies, not explicit CAN-bus acceleration or an exact
instantaneous-velocity measurement. No future displacement supplies this state.

`DDPForesight.encode_current` projects this state with the original
`action_input_model` MLP and inserts it into the Qwen history-token embedding.
The state token precedes W and action tokens under the causal language mask.
Thus both W and H_A can depend on the ego state. The original DDP head samples
q0 from H_A with ten steps from independent noise; the future generator and
FGTR then read H_A and the detached model-generated q0. Their final residual
also depends on the ego-conditioned Qwen representation. State is not merely
saved in a batch dictionary.

There are two different conditioning concepts here: current/past ego-state
input to the policy, and the generated ego plan q0 conditioning future spatial
features. Neither means feeding the exact GT future plan at deployment.
Public "with ego" rows need not share our four-channel interface. In particular,
UniDriveVLA's released planning status interface uses six CAN-bus-derived
channels and historical trajectory input. DiffusionDrive's released head
predicts planning status; its published row is not a directly supplied GT-status
control identical to ours.

## Measured training budget and executed recipe

G3's native run directory is
`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/nuscenes_full8_deployment_20261009/G3_EVENT_LOCAL_formal/G3_EVENT_LOCAL`.
Its frozen training source is
`41da7c198cf0ac4f799aee2a27a0a8126de30230`.

| Quantity | Executed value |
| --- | --- |
| Eligible train700 examples | 23,230 |
| Global batch / GPUs | 32 / 8 |
| Updates per full epoch | 726, including the correctly reduced final partial batch |
| Final updates / valid scene exposure | 17,424 / 557,520 |
| Qwen language / original action / state projection peak LR | 1e-5 |
| Geometry / future generator / FGTR / semantics peak LR | 1e-4 |
| LR schedule | 726-update linear warmup, cosine over the registered 17,424-update horizon, minimum 5% of each group's base LR |
| Executed optimizer | DeepSpeed FusedAdam in AdamW mode, betas 0.9/0.95, epsilon 1e-8, weight decay 0.001 |
| Gradient clipping | Global norm 1.0 |
| VLM adaptation | Frozen Qwen visual encoder; language parameters fine-tuned, without LoRA or EMA |
| Original FM repeat | 8 noise/time draws per scene; auxiliary/refinement losses once per scene |
| Shared perception preparation | Five full train epochs, 116,150 exposures, 4.021 training GPU-hours; no planner training |
| Formal G3 training cost | 174.22594 logged GPU-hours, excluding preparation, qualification and evaluation |

These are trainer/identity/log values, not just YAML interpretation. At the
final update the measured LRs are approximately 5e-7 for original low-LR groups
and 5e-6 for the new-module group. The generic YAML `warmup_ratio: 0.1` and a
single `min_lr` do not define the executed schedule: the trainer supplies one
epoch of warmup and a 5%-of-base minimum to every group. The final native
counter records exactly 24.0 full eligible-population epochs. FM repeat does
not multiply independent data exposure or optimizer-update count by eight.

The user registered observations at 6/12/18/24 epochs and a possible common
extension to 32 epochs under the shared extension rule. This is an initial
research budget, not evidence that 24 epochs is an optimal nuScenes recipe.
G1 has not reached the common endpoint, and its recovery remains deferred at
the user's request.

## Primary-source public recipes

VAD paper: [implementation details](https://arxiv.org/html/2303.12077v3).
Released base configs at
`1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8`:
[stage one](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/projects/configs/VAD/VAD_base_stage_1.py),
[stage two](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/projects/configs/VAD/VAD_base_stage_2.py).
The paper's 60 epochs correspond to a released 48+12 schedule, with the second
stage loading the first stage. The first 48 epochs train perception/motion/map
with all four planning-loss weights zero; planning supervision is enabled in
the last 12 epochs. Eight GPUs, one sample per GPU; AdamW 2e-4,
backbone LR multiplier 0.1, weight decay 0.01, cosine annealing, 500-step
warmup, norm clipping 35. ImageNet ResNet50 initialization; base images
1280x720 and temporal history. The last 12 epochs are not the complete budget.

ResWorld paper: [nuScenes implementation details](https://arxiv.org/html/2602.10884v1).
The paper specifies 12 epochs, eight RTX3090 GPUs, total batch eight, AdamW
1e-4, ResNet50, 256x704 images and current plus two previous frames.
The user's pinned
[released config](https://github.com/mengtan00/ResWorld/blob/a5ae973001dd6001d95b3389020a9e263d73a56e/projects/configs/resworld/resworld_config.py)
instead declares two samples per GPU, implying batch16 if launched on eight
GPUs without overrides. It also loads `geobev-r50-nuimage-cbgs.pth`, uses weight
decay 0.01, clipping 35, 500-step warmup and `step=[24]` with a 12-epoch runner;
that configured decay milestone falls outside the run. Paper and released
defaults therefore cannot be presented as one verified execution log. The
GeoBEV weight's complete pretraining exposure and exact split provenance have
not been reproduced here; its filename alone is insufficient evidence.

UniDriveVLA paper: [training stages and implementation details](https://arxiv.org/html/2604.02190v1).
It specifies three epochs of driving/general VQA preparation at 4e-5 with a
3:7 mixture, 30 joint perception/planning epochs at base 2e-4 and VLM multiplier
0.5 with LoRA and EMA, then 15 expert-specialization epochs at 1e-4 with the
VLM frozen. Driving images are 960x544. This is 45 planning-stage epochs plus
separate VQA preparation, not just the last 15 epochs.
Released configs at `a93c175af893b35dc16618e659eca4d18bb1ec86`:
[joint stage](https://github.com/xiaomi-research/unidrivevla/blob/a93c175af893b35dc16618e659eca4d18bb1ec86/nuScenes/projects/configs/UniDriveVLA/unidrivevla_stage1_2b_cotraining.py),
[specialization stage](https://github.com/xiaomi-research/unidrivevla/blob/a93c175af893b35dc16618e659eca4d18bb1ec86/nuScenes/projects/configs/UniDriveVLA/unidrivevla_stage2_2b.py).
Default NUM_GPUS=32 and samples_per_gpu=4 imply batch128; stage lengths are
computed as floor(28130/(NUM_GPUS*4))*30 or *15. At those defaults they total
9,855 configured updates. The launch script does not export NUM_GPUS from its
GPU argument: an eight-GPU example alone does not establish the actual paper
batch or full-population exposure. The configs also reference a separately
pretrained occupancy VAE. Paper stage numbering includes VQA preparation;
released stage1/stage2 names refer to the two later driving-training stages.

DiffusionDrive's paper
[nuScenes appendix](https://arxiv.org/html/2411.15139v3)
specifies an official SparseDrive perception checkpoint followed by ten
planning-stage epochs. Therefore ten epochs is not its total from-scratch
training budget. The official nuScenes branch at
`ae54fd87b32b3762f20e63ffd0af91d343cade85` provides a
[SparseDrive stage-one config](https://github.com/hustvl/DiffusionDrive/blob/ae54fd87b32b3762f20e63ffd0af91d343cade85/projects/configs/sparsedrive_configs/sparsedrive_small_stage1.py)
with 100 perception epochs, batch64 on eight GPUs, AdamW 4e-4 and backbone
multiplier 0.5. Its
[DiffusionDrive stage-two config](https://github.com/hustvl/DiffusionDrive/blob/ae54fd87b32b3762f20e63ffd0af91d343cade85/projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py)
loads `sparsedrive_stage1.pth` and uses ten epochs, batch48 on eight GPUs, AdamW
3e-4 and backbone multiplier 0.1. Both declare weight decay 0.001, cosine,
500-step warmup and 704x256 input. The source-defined schedule is 43,900
perception updates plus 5,860 planning updates at its declared 28,130-frame
population. These are released configured counts, not recovered author logs or
a verification of the downloaded checkpoint's precise exposure. They do not
mean the planner was trained for 110 epochs.

## What the budget difference can explain

Epoch, update, scene exposure and GPU-hour comparisons answer different
questions. A larger batch can give more data exposure but fewer optimizer
updates; a pretrained perception system can start much farther along even
with fewer planner epochs. If populations were identical, VAD's 60 epochs at
batch8 would give approximately 2.5 times our total driving-training exposure
and ten times our total updates. Its planning-supervised last 12 epochs would
give half our planning-stage exposure and roughly twice our updates, after
48 epochs of perception preparation. These are conditional calculations,
not a measured VAD run.
By contrast, UniDriveVLA's default released 45-epoch configuration has fewer
configured updates than our 17,424; it still has more nominal driving exposure,
additional preparation, and a different staged optimizer. It would be wrong to
say all stronger public models simply take more optimizer steps.
DiffusionDrive likewise has fewer configured planning updates than ours,
but a much longer separate perception stage.

Our randomly initialized large original action head shares the conservative
1e-5 rate used with the generic Qwen language model. Public driving/planning
modules commonly use 1e-4 to 2e-4. This is a plausible optimization mismatch
to investigate, not proof that raising LR will fix the gap: architecture,
loss reduction and optimizer states also matter. We have not calibrated a
separate nuScenes action-head LR experiment.

Native logs add a useful diagnostic: all 726 warmup updates clip the gradient,
with median raw norm 7.9075; 94.74% of the first six-epoch updates clip, compared
with 3.31% of the last six-epoch updates. Adam's adaptive normalization means a
clip factor cannot be interpreted directly as the same factor on the parameter
update. These measurements alone do not establish harmful clipping.

G3's prefix average L2 changes from 0.5836 to 0.5848 m between epochs 18 and 24,
while box collision drops from 0.5484% to 0.2516%. L2 is locally flat under the
current late-stage schedule, while collision remains responsive. Neither
"all tasks are fully learned" nor "longer training must close the gap" follows.
At the endpoint LR is already near its registered floor; merely adding epochs
at that floor is not equivalent to designing and training a longer schedule.

Our population also requires both a legal previous pose and a complete
six-point future. The 23,230 samples are not interchangeable with a public
configuration declaring 28,130 keyframes and using partial-future masks. This
is another reason not to convert a paper's epoch count into an exact matched
update or exposure count without inspecting its actual manifest and sampler.

The minimum causal evidence remains G1/G3 at the same endpoint and inputs,
followed by a registered optimization control shared by both groups if needed.
Current public gaps cannot fairly be assigned to the structured-supervision
idea, and more compute cannot be assumed to remove input or evaluator gaps.

The companion [native-log audit](NUSCENES_TRAINING_BUDGET_20261010.json) verifies
all 17,424 consecutive updates, final counters, per-group endpoint LRs,
FP32-master update evidence and source-file hashes. It contains no scene
tokens, private images or checkpoints.
