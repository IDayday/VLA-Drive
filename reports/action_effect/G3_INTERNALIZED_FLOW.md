# G3: physical-effect knowledge inside one Flow trajectory generation

Status: **nine formal B models trained1000 updates; paired policy comparison
pending**. The original S0 DiT, normalization, noise law and Euler10
sampler are preserved. RouteA checkpoints do not initialize routeB.

The original raw action is8×4 normalized XY/sin/cos. Actual FM uses
`x_u=(1-u)*epsilon+u*a`, target `a-epsilon`, original Beta-derived u including its
small negative tail, repeat8 independent noise/time replicas. At inference, one
complete generation means10 Euler evaluations, not one network call. No original
complete trajectory is generated first and then refined in routeB.

B0: original plus-target bridge `x_plus` learns `a_plus-epsilon`.
B1-trajectory: source bridge `x_minus` learns residual
`a_plus-epsilon-stopgrad(v_base(x_minus,u,z))`.
This is an explicitly named **off-path surrogate**, not the exact CFM derivative
of the source bridge or a proved simulator/optimal-control gradient.
B1-effect: exactly the same source-bridge velocity target, plus independently
verified-pair native-effect and effect-representation supervision.

The frozen G1 teacher is queried only on the real, actually simulated physical
`(a_minus,a_plus)` endpoints and current W/A/ego. It never sees an arbitrary
`epsilon+v` proxy endpoint. Its detached antisymmetric256-vector is the student
representation target. Native11-channel delta targets and gap-sensitive direction
loss use fixed physical scales and full global valid/repeat8 denominators. The
student predicts using only `(F_plan,x_minus,u)`; neither teacher target, real
feedback nor future trajectory is a forward input.

The fixed effect objective is velocity1 +keep1 +delta.2 +direction.2
+representation.2. All three arms allocate the same2,775,567 parameters and seed
schedule; common velocity initialization is unchanged. Their68,619 training-only
auxiliary parameters are removed for export, leaving2,706,948 velocity parameters.
A/B have separate optimizers and model state. Base/VLM remain frozen and no new
visual precision, model architecture, camera or input contract is substituted.

The formal shared1024 recipe uses the same1:1 six-unique-repair/785-identity
reference sampler as routeA, actual repeats/exposure reported rather than called
new data.1000 updates, batch32/microbatch8,50 warmup and unchanged1000 horizon
remain fixed. B0 and B1-trajectory real16 diagnostic runs each consumed512 scenes
and took7.722/7.733s segment, with~3.52GB GPU allocated including the frozen DiT.
These are small engineering smokes, not convergence or image-to-final latency.

After formal endpoints, all arms must use the same1696/16-log DEV, current official
fixed-reference S0 scores, fixed scene noise and Euler10. Report paired-log
bootstrap, original zero repair, high-quality preservation/new zeros, safety,
progress/comfort, same-noise drift/diversity, cost and export. An auxiliary MSE
advantage or an oracle bank cannot substitute for actual generated policy gain.
An available historical RL endpoint with different population/checkpoint is not
a matched same-compute RL control; that comparison remains unavailable unless
an actually matched artifact is registered.

Prior implementation cards and independent mathematical/gradient tests are in
[VERIFIED_PAIR_INTERNALIZATION_PREREG.md](VERIFIED_PAIR_INTERNALIZATION_PREREG.md),
[VERIFIED_PAIR_INTERNALIZATION_IMPLEMENTATION.md](VERIFIED_PAIR_INTERNALIZATION_IMPLEMENTATION.md),
[B_VERIFIED_PAIR_INDEPENDENT_REVIEW.md](B_VERIFIED_PAIR_INDEPENDENT_REVIEW.md) and
[RUNBOOK.md](RUNBOOK.md). The old unsupported-proxy diagnostic is retained for
traceability and is not the main method.

Scientific test: does B1-effect outperform the same-data B0 and B1-trajectory
beyond regret/risk uncertainty? **UNTESTED/PENDING**. No independent method
contribution is claimed until actual paired policy results support it.

All B0/B1-trajectory/B1-effect seed42/43/44 runs now have actual1000-step COMPLETE
receipts. This is9 separate trained models, not a combined candidate pool. G1's
own step1000 teacher is fixed; no external scorer/routeA initialization is loaded.
The current index captures actual training segment time and memory. Complete
Euler10 policy results, safety decomposition and endpoint-image qualification are
still required before judging effect supervision versus ordinary correction FM.
