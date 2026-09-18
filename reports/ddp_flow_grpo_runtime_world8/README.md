# G16 runtime and numerical evidence, 2026-09-18

**NOT_READY. No formal release is issued by this report.** Historical BF16
chunk1/2 remains FAIL. New zero-tolerance failures are also retained verbatim.
The native source digest is13de4419718444787bbffa877b5e5ef25749a4fcff00eb9d109e4ce1f025e0eb.

## Fixed FP32 mathematical comparison

Both predeclared saved rank0 scenes from the native pilot were run, with their
own official rewards, full input and actual BF16 source weight values lifted
toFP32. SDPA MATH, TF32off, G2 andG16, K10, chunk1 versus2; unchanged
atol2e-6/rtol2e-3. No scene or artificial advantage substitution.
All full parameter gradients, optimizer-received gradients, Adam states,
forward weights and fixed-noise updated ODE outputs passed. Scene1 has real
nonzero official advantages. Scene0 has a constant reward group and cannot
prove a nonzero RL update. These are FP32 Adam diagnostics, not ZeRO acceptance.

Scene0/G16 retained FAIL: exactly1/5120 elementwise log-density values exceeds
the historical allclose test; no other comparison fails. Both process exits0
mean the observer completed, not that every internal comparison passed.
The complete per-parameter reports are compressed losslessly in
fp32_scene_0.json.gz and fp32_scene_1.json.gz (gzip -dc to inspect).

A separate no-grad run reproduces the same failure. FP32 transition means
differ by at most4.7684e-7. At flat index4945 the log densities are0.0037179589
and0.0037310719. Evaluating those same means/covariance inFP64 gives
0.0037178607 and0.0037300853. The exact Gaussian mean-sensitivity identity
explains the FP64 difference within4.47e-15. Therefore FP64 probability
arithmetic alone does not remove this layout-dependent mean perturbation.
Dimension-mean ratio max error is9.2983e-6. This explains the observation; it
does not change the historical gate or qualify chunk2.

## Original source CUDA oracle

Native diagnose --skip-gradients completed exit0. Original action-only source
commitf9449d55: ODE max difference0; action-SFT loss exactly0.001596096670255065
in both paths. Three1024x576 views. Behavior/recompute ratio exactly1; full
independent-reference KL0; serial/parallel official rewards and candidate
ordering invariant. Future-label isolation and reference-contamination
negative checks passed. Gradient tests were intentionally not executed by
this invocation; their absence is not a PASS.

## Two-node eight-GPU runtime comparison

Actual two-update native job: rl-zt4 GPU0..3 plus rl-zt2 GPU0..3, ZeRO2,
BF16 stored weights with observedFP32 gradient accumulation/Adam, global16,
accumulation2, G16/K10/chunk1. Both reward/reference overlap and full-chain
inner-probe reuse enabled. Completed exit0 in464.47s. The comparison source
was the prior single-node eight-GPU pilot, so topology also differs.

All initial chains/rewards/advantages/reference statistics are exactly equal;
reused complete ratio statistics equal the former explicit post-update probe.
Zero-tolerance comparison nevertheless FAILS:260/672 gradient tensors in
update1 and116/672 inupdate2 contain unequal elements. Largest per-parameter
relativeL2 is about1.04e-5 inupdate1. All update1 forward tensors are identical;
update2 has15 unequal forward tensors. Adam state differences are retained
fully in cross_topology_comparison.json. Do not claim bitwise equivalence
across these layouts. A same single-node eight-GPU comparison is running to
isolate the hardware-reduction confound. No tolerance was broadened.

Four-GPU independent runtime proofs remain in the adjacent overlap/probe-reuse
reports: all672 gradients, complete Adam/master/forward states and exact
inner-boundary resume were bitwise equal. This does not certify every topology.

Main64-update research arms use their original code; these diagnostics do not
modify their running policies or reward. PDMS/EPDMS outcomes are reported
separately, not inferred from engineering tests.
