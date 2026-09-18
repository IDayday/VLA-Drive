# Hyperparameter sources (2026-09-18)

This is a source/config comparison, not evidence of RL improvement or optimal hyperparameters. Both F and U use the same RL recipe. Their own visual tensors remain frozen; their other SFT-trainable tensors remain trainable. No LoRA, new auxiliary loss, or reward modification was introduced.

Reference: `yifan123/flow_grpo` commit `879042cf5707f8b90daa98d147d7deac2317c5da`, `config/base.py` plus ordinary `pickscore_sd3()` in `config/grpo.py` (not Fast/CPS/GRPOGuard). These configure image generation, not driving. The mathematical sampler is separately pinned in `reference_lock.json`.

| Setting | Official locked SD3.5 / PickScore code | Current DDP F and U | Source of our value |
|---|---|---|---|
| Trainable parameters | LoRA | 672 active tensors, 2,233,120,260 parameters; own visual frozen | User's SFT-minus-visual contract |
| LR | 3e-4 | Fixed 1e-6 for each remaining source group | Both source nonvisual groups 1e-5, user formula ×0.1 |
| Adam betas | (0.9, 0.999) | (0.9, 0.95) | Source SFT optimizer |
| Adam epsilon | 1e-8 | 1e-8 | Source SFT optimizer |
| Weight decay | 1e-4 | 0.001 | Source `trainer.optimizer.weight_decay`; actual RL groups explicitly set this |
| Gradient norm clip | 1 | 1 | Source recipe |
| Group size | 24 | 8 | User's paired protocol |
| Train/evaluation generation steps | 10 / 40 | 10 / 10 | Original checkpoint steps retained |
| Noise level | 0.7 | 0.1 | User's paired protocol |
| PPO clip | 1e-4 | 0.02 | User's paired protocol; not paper-optimized for DDP |
| Inner epochs per behavior batch | 1 | 2 | User's paired protocol; old probabilities and chain retained |
| Advantage clipping | ±5 | ±5 | Fixed protocol, also matches official base |
| Reward standard deviation | Code sets global_std=True | Within each scene's candidate group, population std; epsilon 1e-6 | Fixed paired protocol |
| Reference coefficient | 0.01 for PickScore; 0.04 for GenEval/OCR | 0.01 conditional transition KL | Fixed paired protocol; reductions and models differ |
| Original action SFT replay coefficient | No DDP action replay | 0.1, independently weighted per scene | User's retention requirement |
| Rollout batch | 48 prompts ×24 =1152 image candidates | 16 scenes ×8 =128 trajectory candidates | User global scene batch16 |
| Optimizer updates per collected batch | 2 minibatch updates with one pass; code batch576 candidates | 2 passes on the same16-scene batch | Different minibatching semantics; not interchangeable 'epochs' |
| Precision | FP16 base config | BF16 model, measured FP32 accumulation/reduction/ZeRO partition/master/Adam | Qualified numerical profile |
| Chunking | Different image implementation | Candidate1 / transition1 only | Real numerical acceptance; historical BF16 chunk1/2 FAIL preserved |
| Budget | Task-dependent outer epochs | 2000 actual optimizer updates per variant | User's pre-registered budget, not a convergence claim |

Official code links:
- https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/config/base.py
- https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/config/grpo.py

The original paper v1 appendix B.2 reports G=24, noise=0.7, train/eval steps10/40, LoRA rank32/alpha64, and **KL0.001 for PickScore /0.004 for GenEval and text rendering**. The locked code has different KL values. These sources must not be conflated: https://arxiv.org/html/2505.05470v1#A2.SS2.

Current executable configs: `configs/flow_grpo/paired_world16_{frozen_visual,unfrozen_visual}.yaml`. SFT source configs are pinned to release commit `f9449d55bea6895a7a0bd86d09d7ab85fd353f26` and independently hashed. Actual RL optimizer construction is in `trainer.py` and `contracts.optimizer_groups`; a different legacy top-level SFT `trainer.weight_decay` field does not override `trainer.optimizer.weight_decay` here.

The current PPO clip is numerically200 times the official base value. Smaller noise and LR do not, by themselves, prove the combined recipe is conservative or superior. Dimensional-mean log probability is the configured surrogate, not the joint trajectory probability. Model, reward, action dimensionality, KL reduction and trainable scope differ, so raw hyperparameters alone are not an apples-to-apples algorithm comparison.

At2000 updates with inner_epochs2:1000 fresh behavior batches,16000 fresh scene rollouts,128000 candidate trajectories,32000 scenario-weighted SFT replay exposures. This is not2000 dataset epochs. With101592 train scenes, unique coverage must be measured from the actual cursors; fresh exposures alone are15.75% of one full pass. Five fixed evaluation seeds are42–46; the training seed is42.
