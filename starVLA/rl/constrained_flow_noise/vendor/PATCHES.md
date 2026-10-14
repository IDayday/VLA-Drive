# Locked upstream reuse

The machine-readable source and SHA-256 inventory is
`configs/flow_rl/upstream_lock.json`. Original licenses accompany these files.
This is a DDP adaptation, not a complete reproduction of any upstream trainer.

| Source | Reuse and necessary changes |
| --- | --- |
| RLinf `f6c0dafbf321cbd876ffe1c90b378bfd33ddd594`, `rlinf/models/embodiment/modules/explore_noise_net.py` | `ExploreNoiseNet` and `MLP` copied; only RLinf logging import replaced with standard-library logging. Input is DDP's existing final DiT hidden, before decoding; output is four standard deviations at each of eight positions. |
| Same RLinf commit, `rlinf/models/embodiment/openpi/sampling/rl_sampler.py` | File copied byte for byte. Reuse Gaussian density/entropy helpers behind a strict positive-sigma check. The forward time coordinate is adapted with `s=1-t`, `v_DDP=-v_OpenPI`. Sigma is the discrete transition std, with no additional timestep scaling. |
| ReinFlow `e722e151bed767f3ffef47527cf697f2358af55d`, `model/flow/mlp_flow.py`, `model/flow/ft_ppo/ppoflow.py` | Read-only reference for the learned-noise chain. No MuJoCo, critic or MLP velocity network is migrated. Action, innovation and logprob clamps are not copied: they would invalidate this task's ordinary Gaussian density contract. |
| OmniSafe `15603dd7a654a991d0a4648216b69d60b81a6366`, `omnisafe/common/lagrange.py` | File copied byte for byte. Five independent components use the original negative cost-excess loss, separate Adam optimizers and nonnegative projection. Distributed counting and one update per fresh first batch are task-specific. |
| GDPO `4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115`, `trl-GDPO/trl-0.18.0-gdpo/trl/trainer/grpo_trainer.py` | Componentwise standardization is the reference. The small-tensor reference mode checks sample-std normalization before batch whitening. Formal mode uses stable FP64 centering and registered population-std floors; it omits final batch whitening. No token trainer is copied. |
| verl `4f80e465c2ec79ab9c3c30ec74b9745de61d0490`, `recipe/dapo/src/dapo_ray_trainer.py` | Groupwise filtering and bounded same-behavior replenishment are adapted without Ray/vLLM. This task retains partial/zero-PG collections at its cap and selects by original input order. All candidates within retained groups remain. |
| DAPO entry `33fe3176f0bb212588e84fc8ccf50dd554975144` | README-only source reference. No root license was found at this commit; no code is copied from this repository. |

New task code supplies the DDP hidden interface, strict chain contract,
dimension-mean proximal surrogate, geometry costs, fixed safety budgets,
component floors, first-batch dual statistics and finite collector. Its conditional
Gaussian KL is not an ODE output-distribution KL. Standardization and filtering
do not provide an unbiased uniform CMDP gradient or a hard safety guarantee.

Numerical verification is in `tests/constrained_flow_noise/test_core.py`:
same-weight upstream output, hidden and all MLP parameter gradients;
OpenPI/DDP mean sign conversion; Normal/Independent density and analytical KL;
mean and variance score-function gradients; component, dual and collector cases.
Real DDP/distributed qualification and actual deployment results are reported
separately and are not implied by passing these small-tensor tests.
