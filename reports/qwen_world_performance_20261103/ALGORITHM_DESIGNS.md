# Independent performance-oriented RL: algorithm research

The local adaptation uses the same complete IL70 foundation and unchanged
canonical driving scorer. A loss implementation and CPU tests do not establish
NAVTEST improvement or production GPU stability. Root owns the shared runtime,
registration, batch shapes, resource admission and deployment.
World IL uses a clean-x L1 trajectory term; these velocity-MSE adaptations are
not identical to that original pretraining loss, despite the matching-based
motivation of the cited work.

## Primary sources and reuse

- [Advantage Weighted Matching (ICML 2026)](https://proceedings.mlr.press/v306/xue26c.html)
  motivates learning through advantage-weighted matching rather than only the
  sampled reverse-process transition objective. Its image-generation results
  do not establish a driving gain.
- [AdvantageFlow, v2 (2026-09-29)](https://arxiv.org/html/2605.26013v2)
  uses rollout-policy prediction regularization to make the signed prediction
  objective strictly convex in prediction space. Our bounded coefficient below
  is a conservative local construction, not its exact adaptive regularizer.
- [Flow Matching Policy Gradients, ICLR 2026](https://openreview.net/pdf/376ea41422d8f63f58a4916b9a9783de90e8e461.pdf)
  motivates a PPO-compatible ratio made from CFM losses. Here that ratio remains
  a surrogate; it is not an evaluated final-action density.
- [FPO++ / Flow Policy Gradients for Legged Robots](https://fpocontrol.github.io/)
  discusses further variance and entropy regularization. The present S3 does
  not reproduce its full recipe, critic, entropy treatment or robot setup.

The implementation reads and reuses `ResearchLoss`, `FPOTrainingLoss` and
`RescueCFMLoss`. In particular it reuses the original physical mean-anchor
operation from `fpo_training._physical_anchor`, and ordinary PPO from
`policy.policy_gradient_objective`; it does not stack a new PG loss on top of
the original PG. The old FPO material validators are G8-specific, so the new
adapter seals actual G16/G32 material independently.

## S2: conservative AWFM

For each scored normalized endpoint, use four fixed Monte Carlo pairs with
sigma uniform in [.2,.8], state `(1-sigma)*endpoint + sigma*noise`, and target
velocity `noise-endpoint`. These private draws, the actual collection-start
behavior prediction, and finalized advantages are detached and retained across
the trainer's inner updates. **MC4 is not the inner update count M2.**

With `A=.25*clip(A_final,-5,5)*mask`, minimize

```
mean_MC,candidate [ A*mean24(v-current_target)^2
                 + (relu(-A)+1)*mean24(v-v_behavior)^2 ]
  + original_IL70_beta50_physical_mean_anchor
```

The behavior consistency term remains present for A0 candidates. It initially
has zero gradient when current and behavior predictions agree, and can retain
the behavior after other updates move the shared policy. Its quadratic
coefficient is `A+relu(-A)+1 >= 1`; the velocity-space Hessian of one unaveraged
24-coordinate candidate has eigenvalues at least `2/24`. This does not imply
convexity in neural-network weights or bounded parameter-gradient competition.

`awfm.py` exposes `AWFMLoss(cfg)`, `attach_awfm_material(actor,rollout,rank=...)`,
and `validate_awfm_material(rollout,rank=...)`. Scalar metrics separately report
the signed energy, consistency term and original weighted IL anchor.

## S3: per-MC FPO adapter

Use two fixed MC pairs from the same [.2,.8] interval. The energy is
`sum_8poses(mean_xyz(squared_velocity_error))`, eight times the mean24 units of
S2 and RescueCFM. Old energy is the actual collection-start behavior's detached
energy at the same states, not the permanently frozen IL reference.

Each MC draw independently uses `rho=exp(E_old-E_current)` and ordinary PPO
clip .01 with finalized SafetyEP advantages capped at5. No ratio-temperature,
straight-through clamp, ASPO, Huber replacement or invented endpoint likelihood
is added. Nonfinite energy/ratio refuses execution rather than silently changing
the objective. The original IL70 beta50 mean anchor remains separate.

`fpo_adapter.py` exposes `FPOAdapterLoss(cfg)`, `attach_fpo_material`,
`validate_fpo_material`, and the generic-G `per_mc_fpo_objective`.
Both loss classes call **`self.record_policy_components(pg,weighted_anchor)`**
when the root capture mixin supplies it; the actor API is unchanged.

## S4: actual physical-time credit

Only existing scorer output may be used:

- `non_drivable_body_by_time`:41 actual binary samples at0,.1,...,4seconds;
- `official_first_at_fault_collision_seconds`;
- `official_first_ttc_infraction_seconds`.

Null/missing failed-metric times are unknown, never a fabricated4-second event.
The indicator remains and only that unavailable term is omitted and audited.
An actual event at0seconds is valid evidence; an initially non-drivable flag is
reported without skipping the group or its subsequent40 observed samples.
Feedback and canonical scores remain unchanged; no future label is added to actor input.
Physical driving time is unrelated to the flow model's denoising steps.

The complete **user-defined** cost is

```
c_ind = .40*(1-NC) + .35*(1-DAC) + .25*(1-TTC)
c = c_ind + .15*(1-t_collision/4) [actual event only]
          + .10*(1-t_TTC/4)       [actual event only]
          + .10*mean(actual_non_drivable_flags[1:41])
```

These event terms are not multiplied a second time by indicator deficits.
Absent/unknown time or duration contributes no extra term, with its availability
recorded; this is not an imputed late event and can affect relative rankings
when feedback coverage differs. All-feasible groups retain original SafetyEP.
In mixed groups feasible credit remains exact; only the unsafe subset receives
centered full-cost refinement, uniformly scaled by at most1 and bounded to half
the actual feasible-credit gap and half the unsafe-negative slack. Thus unsafe
credit stays negative and below every feasible candidate. All-infeasible groups
use `mean(c)-c` over the **entire** complete cost, not only equal coarse-cost ties.
A positive least-failed advantage is not a successful teacher.

`temporal_credit_parts(components[B,G,4],feedback_records,base_parts=None)` returns
raw SafetyEP components, `temporal_delta` and an availability/application audit.
The root finalizes the fresh global normalization afterwards; changing raw
credit can change the shared denominator, so unchanged raw groups are not
claimed to have unchanged normalized credit.

## CPU evidence and limits

The directed tests cover numeric negative-A velocity Hessians, exact FPO energy
units and per-MC clipping, detached old targets, G16/G32 material integrity,
private RNG preservation, differentiable current memory, original-anchor value
and gradient parity, and an Adam/pending-payload serialization roundtrip to the
next inner update. Temporal tests cover all-feasible parity, mixed precedence,
unknown and initially invalid real feedback, the exact user cost terms, and
G32 full-cost centering.
The reduced actor and synthetic scoring fixtures are not production weights,
real NAVTEST data or a GPU qualification. Root still performs integration and
production native recovery checks before declaring a live experiment.
