# Executed reference evidence

The final source hash is recorded inside `CPU_REFERENCE.json`. It executed with
one CPU affinity, JAX CPU 0.4.35 and PyTorch 2.5.1+cu124 in float64. CUDA was
invisible and `torch.cuda.is_initialized()` remained false. Mathematical checks
completed in 4.26 seconds, peak process RSS 877,182,976 bytes. These are CPU
reference costs, not DiT training performance.

All **10 arithmetic/behavior checks** passed:

- Four unchanged official-JAX actor-loss versus PyTorch comparisons: mean/min
  critic aggregation × optional normalization off/on. Full loss error at most
  4.44e-16, actor parameter gradient error at most 2.78e-17. Toy endpoints included
  values outside [-1,1], exercising the upstream robot clipping branch.
- Eq. 14/15 shifted-MSE versus linear-gradient local identity: max error1.04e-17;
  scalar objectives differ as expected.
- Paper-versus-source reduction ratio at K=10,D=32: exactly 0.00625 within1e-13.
- Full Euler BPTT versus analytic sensitivity recurrence: error4.44e-16;
  finite-difference error5.40e-11.
- Finite-K, shared-parameter opposite-direction example: exact BPTT small-step
  reward change+6.4e-5; DFQL-paper injection change−8e-6, with zero state Jacobian.
- Actual navigation decoder chain rule: error4.44e-16, no robot clipping used.
- Adding a detached true-score constant does not change the actor gradient.

`CLI_CHECKS.json` records `--help`, wrong official-source hash rejection and
immutable-result rejection. Rejected inputs did not publish a success result.
`CPU_REFERENCE_ATTEMPT1.json` preserves the first successful run. The final
version added explicit non-finite assertions, source identity and CPU/CUDA
accounting; it reran the same ten mathematical checks. No tolerance was relaxed.

The comparison uses mathematical toy networks because this task is a formula
reference. No real DiT checkpoint, road map, private data, original simulator,
optimizer state, DDP or real trajectory score was loaded. It is not a substitute
for the subsequent real generated-trajectory direction diagnostic. Implementers
have not signed independent correctness or resource approval.
