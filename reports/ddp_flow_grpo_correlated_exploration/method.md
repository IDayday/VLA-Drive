# F-only exploration investigation — 2026-09-18

This is an implemented, bounded investigation, not a production acceptance. The
original BF16 candidate chunk1/2 failure is unchanged. F-SFT's own visual weights
stay frozen; Qwen language, history/projector and action DiT remain trainable.
Neither new sampler uses LoRA, a noise network, extra auxiliary losses, selected
candidates, teacher trajectories, or a changed reward. U is not run.

## Why increasing the noise knob was insufficient

For DDP's original linear interpolant, time increases from noise to action:
`x_t = (1-t) epsilon + t action`. Its velocity is `action-epsilon`, and the score
associated with this isotropic interpolant is `(t v_t - x_t)/(1-t)`.
An exact score-corrected SDE changes paths while preserving the corresponding
time marginals. Thus larger diffusion alone does not guarantee a broader terminal
policy. Ten-step discretization also makes excessive noise an accuracy risk.
This is a mathematical implication of the probability-flow/SDE correspondence,
not proof that a learned DDP is exact. See [Song et al.](https://arxiv.org/abs/2011.13456).

The saved real-model banks exhibit strong contraction of the initial candidate
spread. With G16, the median normalized chain RMS standard deviation falls from
0.966 to 0.0114 for correlated Flow SDE. Most baseline nonconstant groups differ
only in progress:39 of47 fixed scenes. G16 increases candidate count but does not
by itself create different driving decisions. The user-identified exploration
limitation is supported by these observations; flow matching is not inherently
incapable of multimodality.

## Two explicit implementations

`starVLA/rl/flow_grpo/temporal_noise.py` defines a full-rank waypoint covariance
`Q_ij = rho^|i-j|`, with rho0.8 fixed before scoring. Its diagonal is1, so
correlation does not increase per-coordinate innovation variance. Correlation is
across the eight physical future waypoints, independently for each action
channel. Innovations at different denoising steps remain independent. The
initial latent remains N(0,I), and original normalization is unchanged.

**Correlated score-corrected Flow SDE:**

```
g(t) = eta sqrt((1-t)/t)   # at t=0 use the original first_dt endpoint rule
score = (t*v - x)/(1-t)
mean = x + v*dt + 0.5*g(t)^2*Q*score*dt
std = g(t)*sqrt(dt)
x_next = mean + std*L*z; L L^T = Q; z ~ N(0,I)
```

The covariance must enter BOTH drift correction and density. Merely coloring
sampled noise with an isotropic score correction/log-prob would be inconsistent.
At rho0, the original arithmetic/sampling branch is retained. Overlapping real
banks match their previously saved trajectories and rewards exactly.

**Correlated noisy Euler:**

```
mean = x + v*dt
std = eta*sqrt(dt)
x_next = mean + std*L*z
```

This intentionally defines a broader discrete stochastic policy; it is not
claimed to preserve the original SFT marginals. At eta0.1, dt0.1 its per-step
std is0.03162; the original Flow SDE final std is0.01054. The fixed eta is a
diffusion amplitude, not ReinFlow's learned discrete transition standard
deviation. Original deterministic single-candidate ODE inference is unchanged.

Both modes whiten the residual with L, include its log determinant, and use the
same covariance in current/old log-prob and full-SFT reference conditional KL.
Entries are Cholesky conditional-density contributions, not independent waypoint
marginals. Their sum is the exact joint transition Gaussian density. The retained
`flow_grpo_dimension_mean` objective divides by32 action dimensions, and its
exponentiated difference is a tempered surrogate, not an exact joint probability
ratio. Tests compare joint density and gradient to independent FP64
MultivariateNormal/AR conditionals. Partial action-dimension masks and
waypoint-varying std are explicitly rejected for correlated transitions.

## Primary references and differences from them

The broader prior investigation is preserved in
[the previous research review](../ddp_flow_grpo_frozen_improvement/research.md).
This extension inspected these additional primary papers/source files:

| Work | Locked source | What informed this change; what was not copied |
|---|---|---|
| [Colored Noise PPO, AAAI2024](https://arxiv.org/abs/2312.11091) | [cn-ppo-paper-code](https://github.com/jkbjh/cn-ppo-paper-code/tree/f2422442468a646ae37dc5e40cca72c3a7b62260) | Temporally correlated exploration; our waypoint AR(1) covariance is not their Fourier-colored online-action process. |
| [Lattice](https://arxiv.org/abs/2305.20065) | [distributions.py](https://github.com/amathislab/lattice/blob/846d02fa993b9b80ce5ecb806463e0a05711bad3/src/models/distributions.py) | Structured Gaussian exploration and covariance-aware likelihood; no latent noise network or humanoid-specific policy is imported. |
| [ReinFlow](https://arxiv.org/html/2505.22094v3) | [ppoflow.py](https://github.com/ReinFlow/ReinFlow/blob/e722e151bed767f3ffef47527cf697f2358af55d/model/flow/ft_ppo/ppoflow.py), lines249–263 and354–361 | Velocity Euler mean followed by a Gaussian transition. Our fixed diffusion, no noise learning/clipping, is an adaptation, not a full ReinFlow reproduction. |
| [Score-SDE / probability flow](https://arxiv.org/abs/2011.13456) | Paper | Explains why preserving a narrow learned marginal is different from broadening exploration. |

The research repositories were read, not installed or executed. Cached source,
SHA and tree responses are under `runs/correlated_exploration/references` and the
previous `runs/frozen_rl_research/references`. Numerical tests use independent
formulas, not claims inferred from a paper's reported performance.

## Experimental boundary

Calibration:64 predeclared training scenes,61logs, zero development overlap,
same scene tokens, seed keys, initial latents and standard-normal innovations;
all16 candidates retained. Both rho0/rho0.8 and eta0.1/0.2 Flow controls are
preserved, including the harmful eta0.2 result. No best-of-N trajectory enters
training or evaluation. Candidate-max statistics are explicitly diagnostic only.

Training: each new arm starts again from F-SFT; G16,K10, global scene batch16,
eight GPUs/accumulation2, candidate/transition chunk1, inner_epochs2, LR1e-6,
clip0.02, KL0.01, action SFT0.1, seed42. Each arm runs8 actual optimizer updates:
64 fresh scenes/1024 candidates/128 SFT replay exposures. BF16 stored model/Qwen,
inherited FP32 action computation, FP32 ZeRO2 accumulation/reduction/master/Adam,
TF32off. No training hyperparameter search or long run is released.

Evaluation:1696 fixed development scenes/16logs, original single-candidate
ten-step ODE and stable seed42+token noise. Report both genuine NAVSIMv1 PDMS and
v2 official one-stage EPDMS, including available adjacent-scene comfort. Training
reward remains the v2 single-scene reward without two-frame comfort. This dev
set may have appeared in source SFT; it is not an SFT-unseen or Navtest result.

Increasing spread or obtaining nonzero gradients is not sufficient evidence of
improved ODE behavior. Correlation here is not proof of new semantic driving
modes. The bounded results and their whole-log uncertainty must be read together.
