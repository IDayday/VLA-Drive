# Independent DFQL formula and implementation reference review

**PASS_REFERENCE** — mathematics and bounded CPU reference only. Reviewer: `/root/spatial20k_correctness`. Reviewed Lead integration `634794fe` (author `77c69ac`), with official repository commit `8b6b15803cc1e5605ad1524553a2c95e103b0bad`. No source, model, trainer, optimizer, checkpoint or scoring implementation was changed. No real policy, map, private scene or simulator was loaded. No training, navigation evaluation, direction diagnostic or dispatch is approved by this report.

## Sources actually read and verified

The local paper and source files in `consequence_proxy_evidence_v1/SOURCE_MANIFEST.json` were individually rehashed and matched their recorded sizes and checksums. The inspected primary source was [Direct Flow Q-Learning](https://proceedings.mlr.press/v306/cao26f.html), equations 13–21 and Appendix A equations 22–39, together with the fixed [official actor implementation](https://github.com/2012060/Direct-Flow-Q-Learning/blob/8b6b15803cc1e5605ad1524553a2c95e103b0bad/agents/dfql.py). No claim about a newer upstream implementation is made.

- PDF SHA256: `3e6cfbca11f6c79e685bb558569d8e8cc895bbc45925b427c3a519500ef3cd54`.
- Official `agents/dfql.py`: `c893151303f507bd77fdb918b8a956d92a582c9f2f6aa7cb9acbcc132c4ecc21`.
- Reviewed `reference.py`: `479238d0a524e3c1cde786ea327003dbb4d5b275389be6b6e75b70c7cba5cb47`.
- Reviewed source manifest: `bfafa40325889ea8a643449c9d6d67864073afef735339e0be62a4079a674632`.

## Mathematical findings

1. **Detach and local derivative.** The actual upstream function saves each local velocity graph, advances the Euler state and detaches that state after every step. The terminal critic gradient is evaluated at the clipped terminal argument. The critic and terminal argument do not depend on actor `grad_params` in that derivative, so the injected terminal gradient is constant with respect to the actor update. This is not full BPTT. A PyTorch adapter should preserve that explicit boundary and should not accidentally include a critic Hessian graph. Clipping the terminal argument before differentiating the critic also does not multiply by the clipping Jacobian of the original unclipped state.
2. **Local shifted-target identity.** At the same parameter snapshot, half squared error to a detached velocity-plus-gradient target and the negative terminal-gradient dot product have equal velocity gradients. Their scalar losses need not match. The result does not justify retaining a stale shifted target across later parameter updates. The paper Appendix A.1 mentions a clipped gradient constant, whereas the fixed official actor function has no gradient-norm clip on that terminal gradient; the evidence report correctly distinguishes this source behavior.
3. **Exact reductions.** With `h=1/K`, `w_k=1/(K-k)`, the paper local term is `-mean_b sum_k h*w_k*dot(g_b,v_kb)`. The executed source term is `-(2*alpha/K)*mean_(k,b,d)(w_k*v_kbd*g_bd)`. The critic first sums its aggregated Q over batch, so no additional `1/B` is hidden inside `g`. Source/paper policy-term ratio is `2/(K*D)`: at K=10 and D=32, **1/160**. Dense-BC source/paper ratio is `1/(K*D)`, giving an effective **2× relative policy-to-BC coefficient**, before optional Q normalization. This is a convention difference, not evidence of an upstream bug. The commented-out sum-over-action expression is not executed.
4. **Ensemble and normalization.** Guidance uses mean or minimum critic according to `q_agg`; optional normalizer always uses mean ensemble Q, is detached, and has no epsilon. The tests exercise finite, nonzero normalizers only. They establish neither robustness at zero Q nor navigation-suitable normalization.
5. **Shared-parameter counterexample.** For zero state Jacobian, K=10, `v_k=theta*b_k`, `b_0=1`, `b_9=-0.2`, other entries zero, and reward the first terminal coordinate, exact reward derivative is `0.08` while the locally weighted derivative is `-0.01`. A small descent step on the corresponding negative-reward losses changes reward by `+6.4e-5` versus `-8e-6`. This is a valid counterexample to an unconditional finite-K neural-parameter direction-equivalence claim. It does not refute the paper's analysis of independent velocity-space perturbations or establish that DFQL usually fails. A neural parameter update couples velocity outputs through its parameter Jacobians, unlike independent functional perturbations.
6. **BPTT reference and navigation conversion.** The affine Euler toy agrees with its analytic sensitivity recurrence and finite differences. The existing navigation decoder test correctly includes x/y scales and the `atan2(sin,cos)` derivative at a nonsingular input. It does not validate singular-heading behavior or an actual road cost. Robot terminal clipping and bounded velocity output must not be copied into the existing navigation policy under a claim of an unchanged action definition. No such navigation modification occurs in this reference.

## What the AST test actually covers

`reference.py` verifies the full official file checksum, extracts the existing `DFQLAgent.actor_loss` function node, and compiles that node without modifying its body. JAX differentiation of this function is compared with a PyTorch transcription using deterministic toy actor/critic functions. All upstream loop detaches, weights, Q aggregation, clipping and reductions in that function are therefore executed. The actor has three synthetic parameter arrays; the two critics are fixed quadratic functions. Diagnostic `sample_actions` is a stub that cannot affect the returned actor objective.

This is **not** an instantiation of the official Flax networks, its full agent, Bellman-critic training, optimizer, shared encoder, checkpoint path, precision behavior, or stochastic training loop. The toy velocity bound is not a reproduction test of `ActorVectorField` itself. The actual source's bound was inspected separately. A trainable navigation condition projection's gradient is correctly discussed in the proposal but not tested by the present 10 arithmetic cases. DDP, microbatch accumulation, real DiT dropout, memory savings, speed and navigation direction are not covered.

The evidence report already states these limitations. No mathematical or scope claim in the reviewed reference requires a source fix. Unsupported extrapolations to reject in later summaries are: “13 real-model tests”, “DFQL equals BPTT”, “constant official PDMS provides an action gradient”, “no BPTT means constant memory/faster training”, and “reference PASS proves a driving improvement”. The report's separate C2/C3 empirical assessment is outside this mathematical review and has not been independently re-evaluated here.

## Independent execution

The existing isolated `/tmp/st-dfql-reference-deps-20261018` packages were reused. No environment installation or upgrade occurred. The owner script was executed from its isolated worktree so its real decoder import resolved to the exact source manifest. `run_reference_checks.py` controls one owned CPU process per invocation with CUDA hidden, one CPU affinity set by the reference, a sampled 2 GiB RSS stop, a shared 110-second supervisor deadline within an external 120-second cap, and bounded stdout/stderr. It creates and removes only its own tiny wrong-hash fixture. This is a local reproduction helper, not a training controller.

Results: **10 arithmetic/behavior checks + 3 CLI checks PASS**, 10.912 seconds total. CPU computation reported 5.161 seconds and 877,649,920 bytes peak RSS; supervisor sampled peak was 888,492,032 bytes. Both are below the 2 GiB budget. CUDA remained uninitialized. Logs, exact commands, CPU evidence and negative exit codes are retained next to this report.

| Independent result | Maximum error / observed value |
|---|---:|
| Official JAX / PyTorch objective, four aggregation-normalization combinations | 4.44e-16 |
| Actor parameter gradients in those comparisons | 2.78e-17 |
| Shifted-MSE / linear local-gradient identity | 1.04e-17 |
| Policy-term reduction ratio | 0.00625 = 1/160 |
| Euler BPTT / analytic sensitivity | 4.44e-16 |
| Euler BPTT / finite difference | 5.40e-11 |
| Navigation decoder derivative | 4.44e-16 |
| CLI help / wrong official hash / immutable result | expected exit codes 0 / 1 / 1 |

The 13 count consists of ten mathematical cases and three CLI cases. It is not thirteen independent navigation experiments. The current actual-road direction diagnostic remains **NOT_RUN**, and formal launch approval remains **false**.
