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

Actual GPU cache probe and target world16 pilot/resume are in progress. The first
probe had exact cached/uncached conditions, transitions, full chains, original SFT
losses and all359 RL-only gradients on all four fixed scenes. Its source ODE oracle
failed because the diagnostic omitted the audited `optimized` inference mode.
That FAIL is preserved; the corrected diagnostic is rerunning without changing
model kernels, seeds, scene selection or tolerances. No acceptance is issued from
this partial evidence. Historical BF16 chunk1/2 failures remain unchanged.

Engineering: NOT_READY pending actual target GPU/resume/export acceptance.
Performance: INSUFFICIENT_EVIDENCE pending new action-head RL paired evaluation.
No previous full-parameter or toy results are presented as this experiment's gain.
