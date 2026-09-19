# Action-head RL with frozen encoder features

The user's latest priority is a real paired PDMS/EPDMS improvement over the original
F SFT, not throughput alone. The explicitly revised parameter contract freezes all
modules outside the checkpoint's `action_model` object. The entire original action
head remains trainable: 359 tensors, 819,503,620 parameters. No LoRA or partial DiT
layer freeze is used in the main configuration. This preserves all of the action
network's adaptation capacity while reducing trainables from 2,233,120,260.

The former full-parameter job was stopped only after update 100 had a complete
checkpoint. Its weights/logs remain at the path in `migration.json`. The new job
starts from the ORIGINAL frozen_visual step100000 SFT; it is not an exact resume
across changed optimizer contracts. Original SFT baseline evaluation was left
running on rl-zt4 GPUs0–5; GPUs6,7 remain reserved.

## Implementation and boundaries

`action_head_policy.py` selects parameters by actual module objects, intersects
with source SFT trainables, and rejects aliases crossing the frozen/trainable
boundary. Existing optimizer coverage and source-versus-actor manifests remain
active. Source configuration and checkpoint files are not modified.

The cache stores the eight normalized Qwen action-token states BEFORE the trainable
`action_model.qwen_proj`. Actor and fixed original-SFT reference each use their own
projection and full action head. Frozen upstream weights are equal to their own
SFT initialization. The original model SFT forward, repeat count, action labels,
normalization, velocity kernel, full G16/K10 chain, BC coefficient and KL controller
remain unchanged. Dataset retrieval uses cached original labels ONLY for replay;
PolicyObservation excludes those labels. Cache misses use original current images,
and never replace scenes or candidates. No auxiliary task is introduced.

Cache identity includes SFT weight identity, processor/tokenizer/config, immutable
input asset identity, dependency versions, encoder source and actual numerical
settings. Records have atomic content seals; conflicting identities and corrupt
entries fail. Concurrent precomputation and training use per-token locks. A cache
can be populated on demand while three separate GPUs precompute the remaining
navtrain features. Readonly use additionally requires the complete publication.
Original inference remains unchanged and does not require this training cache.

## Reference basis and choice

- [Official Flow-GRPO SD3 training](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3.py)
  freezes the VAE and all three text encoders, and trains the generator transformer
  in full or with LoRA. Thus “full parameter” need not mean training the conditioning
  encoder. Its LoRA example uses attention projections, rank32 and alpha64.
- [Official Flow-GRPO FLUX training](https://github.com/yifan123/flow_grpo/blob/main/scripts/train_flux.py)
  additionally targets feed-forward projections in its LoRA configuration.
- [RLinf pi0](https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi_rlinf/rl_action_model.py)
  exposes `train_expert_only`; it freezes SigLIP and the language expert when set.
  It also supports training the VLM. This is an option, not a universal VLA rule.
- [ReinFlow](https://reinflow.github.io/) jointly trains the flow velocity and an
  exploration-noise network; that is a different algorithm, not added here.

The choice to retain the full DDP head, rather than immediately use LoRA, follows
the user's performance-first instruction. No claim that this choice guarantees
improvement is made. The fixed paired reporting contract is in
`performance_contract.json`; engineering checks and speed are separate outcomes.

## Validation status

CPU: 307 PASS, 5 CUDA/resource skips, exit0, 210.13s for the complete flow_grpo and
cluster suites. A previous run had 305 PASS/1 FAIL/5 SKIP because source edits
occurred between the restart-control test's two identity checks. That rejected a
changed executable as designed. The original log is retained. The stable rerun and
27-test affected regression both passed. No full-model CPU Qwen FP32 rerun.

Actual A800 BF16/ZeRO-2 validation completed, with exit0 from every successful
controller below:

- Four fixed real scenes: cached/uncached conditions, velocities, transition
  statistics, full behavior chains, old log-probs, original SFT losses and all359
  RL-only gradients are exactly equal. Two scenes have official nonzero advantages
  and nonzero gradients in all359 tensors; the two constant-reward scenes correctly
  have zero RL gradients. No artificial advantages or replacement candidates.
- Vendored source SFT and original10-step ODE oracle: max absolute error0.
- Native16GPU pilot: two updates on one fixed behavior batch; all359 gradient
  tensors observed on every rank. Actual accumulation, communication, partition
  buffers and Adam states are FP32. Parameters are stored as BF16, but the
  inherited action velocity kernel explicitly uses CUDA FP32 autocast; its
  observed action-module inputs/outputs include FP32. This is not an entirely
  BF16 forward path. See `throughput_diagnosis.json` for the clarification.
- First pre-update ratio is exactly1. Second pre-update range is
  [0.9999966621, 1.0000050068]. Official nonzero-advantage fraction is6/16.
  This warmup diagnostic is a correctness result, not evidence of reward gain.
- Independent actual AdamW oracle: all359 tensors pass the existing moment/master
  tolerances, and saved forward weights equal the actual master-weight cast.
- Inner-epoch resume: update1→2 and continuous1→2 have exactly matching51 files,
  including model, optimizer, RNG and pending behavior/stream states. No tolerance
  relaxation or pending-chain resampling.
- Every rank's complete frozen/reference tensor hashes are unchanged.
- Original `infer.VLAAgent` export:989 tensors identical, prediction max error0.

The first cache probe's source ODE oracle FAIL remains archived: the observer
omitted the audited `optimized` inference mode. All cache/gradient comparisons in
that failed attempt were already exact. The corrected observer passed without
changing model kernels, seeds, scenes or tolerances. The earlier CPU interrupted
identity test and original BF16 chunk1/2 FAIL are also preserved, not rewritten.
Raw failed evidence is compressed losslessly in the adjacent `.gz` files.

Engineering: READY_FOR_THIS_PROFILE for the explicitly registered single F
full-data experiment only:16GPUs, BF16/ZeRO-2 with FP32 accumulation/communication,
full action head, candidate chunk1, transition chunk1, G16/K10, global scene
batch16, two inner epochs, frozen-prefix cache and no-grad velocity CUDA graph.
This does not qualify other candidate chunks, topology, precision or U weights.
The machine-readable gate says `AUTHORIZED_FULL_DATA_EXPERIMENT`; it is not a
blanket production release. Code/config/assets changes invalidate that gate.

Performance: INSUFFICIENT_EVIDENCE pending new action-head RL paired evaluation.
No previous full-parameter or toy results are presented as this experiment's gain.
The original SFT full-navtest seed42 result is PDMS88.8644 and EPDMS88.2378 on a
0–100 scale. These are baseline values, not the new RL results or a five-seed mean.

## Live experiment and reproduction

Execution source commit:`2e7d0220287d394dcc7d36d1076548ae2949ab31`.
Native executable digest:`02d7a7287feaf5730086457160eb05bf6e7a4fdf3d437267e535af41219e8e4d`.
Checkpoint/config/environment/dtype evidence is in `validation.json` and the
compressed environment and parameter manifests. The real runtime root is
`/mnt/project/DriveDreamer-Policy-action-rl/runs/action_head/world16`.

The launched experiment uses training-vla-zt2 and training-vla-zt3, eight GPUs
each. Feature precomputation uses rl-zt2 GPUs1–3; completed diagnostics used GPU0.
The existing baseline producer owns rl-zt4 GPUs0–5, with6,7 reserved. Final RL
evaluation follows that baseline on the evaluation GPUs, independently of training.
No duplicate original-SFT inference job is launched.

Budget:103288 full-navtrain scenes,6456 fresh16-scene behavior batches,12912
optimizer updates with two inner epochs. Padding is8 scenes, not extra unique
data. This gives1652736 generated candidates and206592 replay-scene exposures.
The first two registered updates count toward the budget. The model starts from
the original F SFT, not the former full-parameter RL checkpoint. The fixed final
checkpoint is compared on all12146 navtest scenes with seeds42–46; navtest is not
used for LR/noise/seed/checkpoint selection. Both official protocols, log-bootstrap
paired intervals and zero/high-score regressions are recorded.

From this worktree, use the existing environment and entries below. Existing
immutable diagnostic output directories must not be overwritten; use a new
output/control directory when deliberately reproducing a diagnostic.

```bash
cd /mnt/project/DriveDreamer-Policy-action-rl
export PYTHONPATH="$PWD/navsim:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/root/miniconda3/envs/ddp/bin/python
CFG=configs/flow_grpo/frozen_action_head_epoch1.yaml
RUN=$PWD/runs/action_head/world16

# Optional read-only preflight; full corpus verification is already locked.
CUDA_VISIBLE_DEVICES='' "$PY" -m starVLA.rl.flow_grpo.cli preflight --config "$CFG" --output-dir "$RUN/preflight"

# Single-GPU original-source/cache/RL-gradient probe on an allocated idle GPU.
"$PY" -m scripts.analysis.action_head_cache_probe --config "$CFG" --bank /mnt/project/DriveDreamer-Policy-epoch-speed/runs/epoch_speed/world16/pilot --output "$RUN/cache_probe_reproduction"

# Actual16GPU diagnostic and explicit inner-epoch resume specs are archived here.
# Give copied specs fresh output/control paths before reproducing them.
"$PY" -m scripts.cluster_flow_grpo.cluster run "$RUN/pilot_spec.json"
"$PY" -m scripts.cluster_flow_grpo.cluster run "$RUN/resumed_spec.json"

# Full experiment/start-or-resume: one active controller only. It resumes the
# latest complete checkpoint and independently evaluates baseline/final models.
"$PY" -m scripts.analysis.accelerated_epoch_run --spec "$RUN/experiment_spec.json"

# Export after the final complete boundary; completed identical exports reuse.
"$PY" -m starVLA.rl.flow_grpo.cli export --checkpoint "$RUN/train/checkpoints/update_012912" --output-dir "$RUN/train/export_update12912"

# Individual original-protocol evaluation, on an allocated evaluation GPU.
# The controller normally runs all5 seeds plus v1 PDMS from the same trajectories.
"$PY" -m starVLA.rl.flow_grpo.cli evaluate --config "$CFG" --checkpoint "$RUN/train/export_update12912" --split navtest --tokens /mnt/project/DriveDreamer-Policy/test_meta.json --data-root /mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset --metric-cache /mnt/project/DriveDreamer-Policy-paired/runs/metric_cache_navtest_v2 --seed 42 --metric-protocol navsim_v2_official_one_stage --output-dir "$RUN/manual_final_navtest_seed42"

# Finalize precomputed frozen features only after the producer finishes; the
# current launch already has a bounded CPU continuation for this command.
CUDA_VISIBLE_DEVICES='' "$PY" -m scripts.analysis.precompute_action_features --config "$CFG" --finalize
```

No held-out `rl_dev` split is claimed in this full-navtrain experiment. Validation
of the algorithm uses fixed calibration observations; performance comes from the
predeclared complete navtest comparison. The controller's final
`paired_performance.json` is produced only after both models' complete five-seed
evaluations. A running job, training reward, loss or throughput does not meet the
performance claim rule in `performance_contract.json`.
