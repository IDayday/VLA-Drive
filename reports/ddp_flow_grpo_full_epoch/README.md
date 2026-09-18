# Full navtrain, one data epoch, F-only

This is the user's explicitly authorized full-data experiment. It starts from
the original action-only **frozen_visual SFT step100000**, SHA256
`9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4`.
The previous 64-update checkpoints are not initialization or reference models.
Both baseline and trained policy use single-candidate original ten-step ODE.

## Data and budget

All **103,288** official navtrain scenes, **1,192** logs. No navtrain holdout.
The former 1,696-scene RL dev set now participates in training and cannot be
used as a held-out metric or for checkpoint selection. The original published
asset manifest is unchanged; `prepare_epoch.py` writes a separate split file.
Previously completed full-corpus verification is reused through an explicit,
source-bound receipt, as requested. This is not a new byte scan of the corpus.

Global scene batch16, eight GPUs, microbatch1, accumulation2, G16, K10,
inner_epochs2: `ceil(103288/16)=6456` fresh behavior batches and **12,912 actual
optimizer updates**. The cyclic sampler adds eight deterministic repeated
scenes in the final batch: 103,296 fresh scene exposures, 1,652,736 candidate
trajectories and 206,592 action SFT replay exposures. No scene is dropped.
The two inner epochs reuse the original chain, old logprob and advantages;
they are not two fresh data epochs.

Full navtest means exactly **12,146** official scenes, not a subset. Both
F-SFT and the fixed final checkpoint use seeds42,43,44,45,46 with noise keyed
by token and seed. The final checkpoint is prescribed by the budget, never
selected from navtest. Report actual v1.1 PDMS and v2 one-stage EPDMS separately;
the single-scene training reward remains the original v2 reward without
adjacent two-frame comfort. The v2 evaluator still computes available adjacent
metrics using whole-log shards. No reranking, best-of-N or trajectory replacement.

## Stability changes and their basis

Original nonvisual SFT group LRs are all1e-5; multiplier0.1 gives peak1e-6.
Warmup is **388 optimizer updates** (ceil3% of12912), followed by cosine decay
to **1e-7** on the final update. The first update uses1e-6/388; no zero-LR Adam
step. The schedule horizon is part of immutable config, independent of a
diagnostic `max_updates` limit. The native trainer steps the scheduler exactly
once per optimizer update. Accelerate1.5.2 otherwise multiplies a scheduler
step by world size with its default settings. Installed-wrapper tests cover
this ownership change; actual CUDA resume evidence is reported separately.

The cosine shape follows the standard [Transformers optimization
implementation](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/optimization.py).
Three percent warmup and a ten-percent floor are this experiment's predeclared
engineering settings, not claimed to be optimal or copied from a driving paper.

BC remains the original action flow-matching SFT replay, coefficient0.1,
counted once per scene per update, not G*K times. No video/depth/world-model
tasks are enabled. The full fixed SFT policy provides its own reference
conditions, visual parameters and action head. It is never updated or mixed
with the actor. PPO clipping0.02 and global gradient norm clipping1 remain.

Reference KL starts at **beta0.04**, with target **0.02**, coefficient bounds
**[0.01,1]**, and feedback horizon **1024 scene exposures**. Once per update:

`beta_next = clamp(beta * (1 + clip(global_KL/target-1,-.2,.2)*16/1024), .01, 1)`.

This is the bounded proportional feedback in [TRL v0.11.4
AdaptiveKLController](https://github.com/huggingface/trl/blob/v0.11.4/trl/trainer/utils.py#L51),
which cites Ziegler et al. 2019. Bounds are our explicit retention safeguard.
The observation here is the globally scene-weighted **dimension-mean
conditional Gaussian transition KL**, not text sequence KL or an exact joint
trajectory probability ratio. The target is a registered engineering choice
in these units; it is not borrowed as an equivalent PPO/LLM threshold.
[Official Flow-GRPO configs](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
also use reference regularization; that does not establish a driving optimum.

The controller changes only the next update's loss coefficient. Its state is
saved by Accelerator and must match the resumed optimizer boundary. No
reference refresh, no current-VLM features supplied to reference, no changes
to saved old probabilities/advantages, and no artificial reward signal.
Logs distinguish LR used from next LR, beta used from next beta, and measured
KL from its target. This run changes data coverage AND stabilization, so it
cannot isolate a causal data-only effect relative to the earlier runs.

## Supported experiment scope and evidence

Single-host `training-vla-zt2` GPUs0..7, BF16/ZeRO2 with observed FP32
partition accumulation/reduction/Adam/master weights, activation checkpointing,
candidate chunk1, transition chunk1. Existing reward/reference overlap and
full-chain inner-probe reuse are enabled; the reference/BC/model parameter
contract is otherwise preserved (672 trainable tensors;2,233,120,260 params).
Full navtest inference uses `training-rl-zt4` GPUs0..5, leaving6,7 free. Each
evaluation worker uses one CPU thread; v1 scoring uses eight separate CPU
workers. `training-rl-zt2` is only used for a bounded one-GPU oracle diagnostic.
No unrelated jobs or pressure processes were stopped in this turn.

The explicit `full_navtrain_epoch` gate requires this exact recipe, full
official token set, a completed matching native eight-GPU two-update pilot,
nonzero official group advantages, all-rank dtype/gradient/frozen proofs, and
exact checkpoint1-to2 resume. It preserves the previous diagnostic8,
bounded-research64 and formal2000 limits. It does not issue a blanket READY
record or change historical failed tests.

Historical BF16 chunk1/2 remains **FAIL**. The earlier FP32 scene0/G16
elementwise-logprob exception and cross-topology zero-tolerance differences
also remain recorded in `reports/ddp_flow_grpo_runtime_world8`. No evidence
here supports chunk>=2, a different world size, or guaranteed reward gains.

CPU regression: **283 PASS,4 SKIP**, exit0 (`cpu_regression.log/.xml`), including
the existing real official CPU reward/adjacency tests. The four skips are CUDA
tests run with `CUDA_VISIBLE_DEVICES=''`, not passes. Controller tests: **3 PASS**,
exit0. These are not GPU model acceptance or RL performance improvements.
An initial new test-fixture generator/list bug was corrected before this final
regression; no model tolerance was changed. GPU results and actual deployment
are recorded in the companion deployment report after execution.

## Reproduce on this installation

Python is `/root/miniconda3/envs/ddp/bin/python`; workspace is
`/mnt/project/DriveDreamer-Policy-full-navtrain`. Keep the installed environment.
Set `PYTHONPATH=$PWD/navsim:$PWD` and
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.

```bash
python -m scripts.flow_grpo.prepare_workspace
python -m scripts.flow_grpo.prepare_epoch
python -m scripts.cluster_flow_grpo.cluster run runs/full_navtrain_epoch1/pilot_spec.json
python -m scripts.cluster_flow_grpo.cluster run runs/full_navtrain_epoch1/resumed_spec.json
python -m scripts.cluster_flow_grpo.boundary_evidence \
  --continuous runs/full_navtrain_epoch1/pilot/checkpoints/update_000002 \
  --resumed runs/full_navtrain_epoch1/resumed/checkpoints/update_000002 \
  --output runs/full_navtrain_epoch1/exact_resume.json --cpu-threads 8 --streaming-load
python -m scripts.flow_grpo.register_epoch \
  --config configs/flow_grpo/frozen_full_navtrain_epoch1.yaml --root runs/full_navtrain_epoch1
python -m scripts.analysis.full_epoch_run --spec runs/full_navtrain_epoch1/experiment_spec.json
```

Use fresh diagnostic output/control directories when repeating completed
pilots; existing checkpoints are protected. The epoch controller resumes its
latest validated complete checkpoint and only bootstraps from pilot update2
when no later epoch checkpoint exists. It never repeats already saved updates
just because export/evaluation failed. Reinvoking the controller after an
interruption also resumes evaluation transactions and preserves failed attempts.
The published launch specs and asset receipt must match actual source; they
are evidence, not a way to bypass validation after editing code.

```bash
python -m starVLA.rl.flow_grpo.cli export \
  --checkpoint runs/full_navtrain_epoch1/train/checkpoints/update_012912 \
  --output-dir runs/full_navtrain_epoch1/train/export_update12912
python -m scripts.cluster_flow_grpo.parallel_evaluation \
  --config configs/flow_grpo/frozen_full_navtrain_epoch1.yaml \
  --checkpoint runs/full_navtrain_epoch1/train/export_update12912 \
  --output runs/full_navtrain_epoch1/evaluation/last_navtest_seed42 --split navtest \
  --tokens /mnt/project/DriveDreamer-Policy/test_meta.json \
  --data-root /mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset \
  --metric-cache /mnt/project/DriveDreamer-Policy-paired/runs/metric_cache_navtest_v2 \
  --seed 42 --slots runs/full_navtrain_epoch1/evaluation_slots.json
```

The controller runs all five predeclared seeds and actual v1.1 rescoring via
`scripts.analysis.paired_dev_pdms_v1`, now supporting explicit navtest/seed.
It writes scene deltas, zero-to-nonzero/nonzero-to-zero and high-score
degradation counts, per-seed and five-seed summaries, and log-bootstrap CIs.
Engineering execution and PDMS/EPDMS improvement remain separate conclusions.
