# Direct Flow terminal-gradient reference, before any navigation proxy training

Status: **CPU_REFERENCE_EXECUTED; NAVIGATION_ADAPTATION_NOT_RUN; NEEDS_INDEPENDENT_REVIEW**.
This is an evidence closure for original plan §10, not approval, a new formal arm,
or evidence that the current C2/C3 policies improve driving. No trainer, reward,
checkpoint, simulator, input, deployment path, or shared source was modified.

The inspected paper is [Direct Flow Q-Learning, PMLR 306](https://proceedings.mlr.press/v306/cao26f.html),
method §3 and Appendices A–C. Official source is
[`agents/dfql.py` at 8b6b15803cc1e5605ad1524553a2c95e103b0bad](https://github.com/2012060/Direct-Flow-Q-Learning/blob/8b6b15803cc1e5605ad1524553a2c95e103b0bad/agents/dfql.py).
The conference Software link redirects to this repository. Existing local PDF
was reused and its hash compared with the current official PDF bytes. Exact
file hashes, read scope and CPU dependency provenance are in SOURCE_MANIFEST.json.

## 1. Three different derivatives, not three names for the same algorithm

Let `z_k` be a complete raw trajectory particle, `h=1/K`, `tau_k=k/K`,
`z_{k+1}=z_k+h*v_theta(H,z_k,tau_k)`, and `z_0=epsilon`. For our future adapter,
`K=10`, `z_k` has shape `[B,8,4]` and represents normalized x/y/sin(yaw)/cos(yaw).
The `k` index is **flow integration**, not the eight physical trajectory times.
Let `R(z)=-J_S(decode(z))` denote an explicitly defined smooth training surrogate.
There is no learned Bellman critic in this proposed contextual-trajectory adapter.

**Full BPTT reference.** Freeze H and environment labels, but retain gradients
through every Euler state and the exact raw-to-physical conversion:

```
L_BPTT(theta) = -mean_b R(z_K(theta))
S_0 = 0
S_{k+1} = (I + h*J_k)*S_k + h*B_k
J_k = partial v_theta / partial z_k
B_k = partial v_theta / partial theta at fixed z_k
nabla_theta L_BPTT = -mean_b S_K^T * g_b
where g_b = nabla_z R(z_K).
```

For a shared trainable condition projection, `B_k` includes its direct path.
Detaching projected H would incorrectly remove that path. The upstream frozen
encoder can be detached; the selected actor projections cannot be detached merely
for convenience. A small-batch reference need not run the full encoder again.
The original model's precision, dropout policy and integer time buckets remain
part of the eventual execution contract; the present float64 affine toy verifies
only the mathematics. This BPTT reference is **not the complete RFO algorithm**.

**DFQL paper surrogate.** At the current snapshot, form a frozen actual rollout
and terminal gradient. Treat `z_k` and `g` as constants for actor differentiation:

```
w_k = 1/(K-k)
L_inject_paper = -mean_b sum_k h*w_k * <sg(g_b), v_theta(H,sg(z_k),tau_k)>
```

This omits the subsequent-state Jacobian products. With local shifted targets
`sg(v_theta + h*g)`, the half-squared-error derivative equals the derivative of
`-h*<g,v_theta>` **at that same snapshot** (paper Eq. 13–15, Appendix A.1).
The two scalar losses are not equal. Reusing a stale shifted target for additional
updates is a new objective; it is not the same local identity.

**Canonical PDMS.** The official tracker/simulator/scorer is not this smooth
function. Converting its scalar to a tensor supplies no action derivative. A
finite-difference check of `J_S` proves its implementation derivative; it does
not prove that following that derivative improves actual executed PDMS.

## 2. What the pinned official code actually computes

The CPU program executes the unchanged AST of `DFQLAgent.actor_loss` from the
pinned file. It supplies a small differentiable mathematical actor and two
quadratic toy critic functions, not pretrained networks or a robot environment.
This exercises actual upstream detaches, clipping, weighting and reductions.
It does not instantiate or reproduce the full Flax agent/training loop.

| Symbol / lines in pinned dfql.py | Inspected behavior | Navigation consequence |
|---|---|---|
| actor_loss, 45–80 | Gaussian initial noise; actual Euler path; detach after **every** step; retain each local velocity graph | Retain actual ten-step rollout, not a GT bridge endpoint; partial state sensitivity is intentional |
| actor_loss, 65–66 | A second velocity evaluation per step on the GT linear bridge, sharing initial noise | This is upstream dense BC; it must not silently replace the original navigation Beta/repeat8 FM anchor |
| actor_loss, 76–90 | Clip detached terminal action to [-1,1], compute mean/min ensemble Q, **sum over batch**, then differentiate w.r.t. the clipped-action argument | Do not copy robot clipping; batch sum is important, as the later loss already averages batch |
| actor_loss, 90–95 | Broadcast one terminal gradient to all velocities; weight 1/(K-k); multiply by 2*alpha/K; mean over K,B,d | Preserve/report exact reduction, not merely the time weights |
| actor_loss, 97–106 | Dense BC mean; optional reciprocal absolute mean-Q normalization, detached, with no epsilon | Do not enable normalization by default for a road proxy; its value can be zero |
| critic_loss, 21–43 | Bootstrapped target using next observations, discount and target critic | Not applicable to one fixed-context complete-trajectory outcome; no Bellman migration |
| compute_flow_actions, 187–206 | Euler integration followed by robot action clipping | Original navigation still emits its un-clipped raw complete trajectory |
| create, 244–249; networks.py 197–238 | Upstream actor enables a 5*tanh velocity guard | Do not add this bound to the original DiT while claiming only an update rule changed |

Writing the actual source policy term with `D=32`:

```
L_Q_source = -(2*alpha/K) * mean_(k,b,d)[w_k * v_kbd * g_bd]
L_BC_source = mean_(k,b,d)[(v_BC_kbd - (target-noise)_bd)^2]
```

For the same `v,g,alpha`, the source Q term equals
`2/(K*D)` times `alpha*L_inject_paper`. At K=10,D=32 the factor is **1/160**.
The paper's sum-over-step/sum-over-coordinate BC becomes `L_BC_source` by a
factor `1/(K*D)`. Therefore the combined source/paper actor objectives also differ
by an effective factor **2** in RL versus BC, before optional normalization.
This is not proof of a code defect; it is a concrete convention that must be
bound and tested. Copying a published alpha without its reductions is unjustified.
The commented-out sum-over-action Q line is not the executed source line.

For `q_agg=min`, the guidance uses the minimum critic, but normalization/logged
`q` uses the mean ensemble (`q = mean(qs,axis=0)`). The CPU audit covers both
aggregations and normalization on/off. Critic weights are not functions of the
actor's `grad_params`; the detached action makes `g` constant w.r.t. that actor
derivative without a visible second `stop_gradient(g)` call. A PyTorch port
should make this boundary explicit and avoid a Hessian graph. The source computes
the derivative with respect to the already-clipped argument; it does not multiply
by the clipping Jacobian of the un-clipped endpoint. No gradient-norm clipping of
`g` appears in this actor objective. These details cannot be inferred from a short
verbal description of “terminal-gradient injection.”

## 3. What is and is not established by the paper's derivation

Appendix A.2 analyzes **functional, step-local velocity perturbations** and a
large-K direction argument. It does not establish elementwise equality with
BPTT at K=10, nor a global guarantee for a shared neural parameter update,
nonlinear optimizer, FM anchor, or imperfect road proxy. Small per-step Jacobian
terms need not imply their full product is close to identity over a fixed unit
interval. The actual neural update also couples every sampled time through its
parameter Jacobian/Gram matrix.

The reference includes a particularly simple counterexample with **zero state
Jacobian**, K=10 and a 32-coordinate particle. Only one coordinate is active,
`v_k=theta*b_k`, with `b_0=1`, `b_9=-0.2`, other `b_k=0`; reward is the terminal
first coordinate. Exact reward derivative is `0.1*(1-0.2)=0.08`; the DFQL-paper
weighted sensitivity is `0.1*(0.1-0.2)=-0.01`. Thus a small DFQL parameter update
moves this reward in the opposite direction to full BPTT. This refutes an
unconditional finite-step/shared-parameter equivalence claim. It does not refute
the stated velocity-space asymptotic analysis or the paper's benchmark results.

Likewise, “no BPTT” does not mean constant activation memory or faster training:
the upstream implementation retains K local velocity graphs and evaluates dense
BC K times. Any recomputation/cost benefit on the actual DiT must be measured.

## 4. Bounded navigation mapping

Existing `DiTFlowPolicy.velocity` already supplies raw `[B,8,4]` velocities and
original integer time buckets. Its `project_condition` detaches encoder input
but trains the original actor projection. The future implementation can reuse
these interfaces without changing the action head or checkpoint tensors.

The current decoder is exactly:

```
x = raw_x * 8.805105 + 10.172484
y = raw_y * 2.277741 + 0.360762
yaw = atan2(raw_sin, raw_cos)
```

The CPU test imports that actual decoder and checks the chain rule. A physical
proxy derivative must include these scales and yaw derivatives. Treating the
last two channels as angles is wrong. Near sin=cos=0 the heading derivative is
ill-conditioned; a future proxy must report this validity condition rather than
quietly clipping, renormalizing or repairing the generated trajectory.

Copying [-1,1] clipping would constrain x to roughly [1.367,18.978]m and y to
[-1.917,2.639]m for this normalizer. That changes valid navigation actions and is
not an acceptable optimization-only port. No such clipping is used by the local
navigation decoder test or proposed next diagnostic.

For an eventual matched policy comparison, keep original FM/Beta/repeat8 and its
original reduction; add an explicitly identified terminal-gradient term. That
is **DFQL-style injection adapted to navigation**, not complete official DFQL.
The source dense-BC branch is reproduced only in the arithmetic audit, not
proposed as a replacement for the campaign's existing anchor. No alpha is
selected or trained here.

## 5. Exactly one next diagnostic, not another training matrix

**NOT_RUN / requires Lead registration and independent reviews:** a bounded
road-direction test on actual generated TRAIN trajectories, before training
BPTT or injection policies.

1. Freeze one current legal policy/checkpoint, inference seed, original ten-step
   sampler and existing training manifest. Select up to 32 scenes by a fixed
   hash before reading outcomes; use natural samples, not dev-selected failures.
   Report how many have a nonzero road-proxy gradient; if too few, report
   insufficient directional information rather than manufacture failures.
2. Implement only a **command-space full-vehicle road-margin surrogate** against
   the same map union/coordinate contract. Use rear-axle-to-footprint geometry
   and original trajectory interpolation; account for holes, disconnected lane
   unions, boundary nonsmoothness and map coverage. It is not a tensorized
   LQR/bicycle implementation. First check finite differences away from switches.
3. On each unchanged generated trajectory, compare original, minus-gradient,
   plus-gradient and fixed-seed same-norm random perturbations. Use a single
   predeclared small physical perturbation radius and the same raw/physical
   metric for all directions; no radius search or GT bridge. Up to **128**
   canonical single-trajectory score calls including original trajectories,
   with legitimate identical-hash reuse. Bind map, metric context and trajectory
   hashes. Non-finite/singular/zero-gradient cases remain in denominators with
   explicit status, not fabricated successes.
4. Rescore through the original tracker, bicycle simulator and fixed reference
   backend. Compare executed DAC, minimum/full-body road margin where available,
   progress, NC/TTC, comfort, zero-score/high-score transitions, and displacement.
   Report direction agreement and the cases where proxy improves but execution
   worsens. No dev/Navtest or policy update in this diagnostic.

The proposed fixed radius and footprint discretization must be specified from
actual geometry/precision before this diagnostic can run; they are not silently
chosen in this report. If the proxy has no reliable executed-direction signal,
do not train a predictor or tune DFQL weights to conceal that mismatch. If it has
usable signal, the next separate decision is the already-planned small matched
BPTT versus injection adaptation with original FM—not automatic formal launch.

The current C2/C3 4096-step results supply no reliable practical gain. Their sparse
feedback supports examining whether a smooth **validated** signal is useful; it
provides no evidence that this proposed road proxy or DFQL-style port works.
