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

## Completed single-node eight-GPU follow-up

Actual training-vla-zt2 GPUs0..7, same world8/accum2/global16 recipe, two
updates completed exit0 in424.42s. Full zero-tolerance comparison against the
original single-node eight-GPU pilot **PASS**:672/672 optimizer-received
gradients at each update, every Adam moment/master/forward tensor, and complete
reused ratio statistics are identical. The cross-topology FAIL above remains.
Single-node equality isolates that the two runtime changes need not perturb
this supported diagnostic computation layout; it is not an all-topology claim.

The two update spans total146.70s versus185.78s in the earlier pilot; whole
job424.42s versus477.47s. These jobs ran on different physical hosts with
different CPU contention, include gradient observers, and are not a controlled
claim of21% steady-state throughput gain. The removed intermediate full probe
is real; the adjacent same-GPU reward-overlap trials isolate that phase.

Same single-node checkpoint1 restored to update2 completed exit0 in285.28s.
All26 stored files (full model, all optimizer partitions, RNG/cursors, pending
behavior and scheduler state) match the continuous boundary exactly. Both
inner epochs retain the original behavior chain, advantages and old logprob.

Additional completed evidence: independent CUDA AdamW calculation against
actual eight-rank reduced gradients/moments/master/forward weights PASS for
all672 trainables (cross-topology run; original bounds unchanged). Original
VLAAgent export validation PASS:989 saved tensors identical, fixed-noise
ODE output max difference0. All own-visual/reference/frozen hashes remain
unchanged. Actual per-rank dtype inventories are included separately from
the declared numerical profile.

Native regression remains265 passed/4 skipped, with31 overlapping targeted
probe tests in the adjacent report. No native/model path was edited during
these final observer/report additions, so the full suite was not redundantly
rerun. The added read-only observer compiled and executed on real CUDA tensors.

These results validate specific runtime transformations and resume, not the
whole new algorithm for formal training. The completed gamma0.6 step64 result
regresses both devPDMS andEPDMS; source-matched formal release evidence is
not complete. **NOT_READY** remains. The native budget rejects unqualified
long training; no READY artifact or automatic long job is created.

Reproduce the completed bounded controls from this worktree (fresh output
directories/specs required; existing completed checkpoints are protected):

```bash
python -m scripts.cluster_flow_grpo.cluster run runs/fast_world8/single8_spec.json
python -m scripts.cluster_flow_grpo.cluster run runs/fast_world8/single8_resume_spec.json
python -m scripts.analysis.checkpoint_efficiency \
  --on /mnt/project/DriveDreamer-Policy-paired/runs/denoising_credit/checkpoint_on_attempt2 \
  --off runs/fast_world8/single8 --difference runtime_pipeline \
  --output NEW_comparison.json
python -m scripts.cluster_flow_grpo.boundary_evidence \
  --continuous runs/fast_world8/single8/checkpoints/update_000002 \
  --resumed runs/fast_world8/single8_resumed/checkpoints/update_000002 \
  --output NEW_resume.json --cpu-threads 4 --streaming-load
```

Use /root/miniconda3/envs/ddp/bin/python, PYTHONPATH=$PWD/navsim:$PWD,
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1. The actual saved
specs, resolved config, source identity and receipts accompany this report.
