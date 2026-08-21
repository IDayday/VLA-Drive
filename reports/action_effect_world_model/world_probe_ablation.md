# World-probe ablation status

Status: **implemented and run through Gate 3** on the scene-disjoint
`pilot_small` split. The complete three-seed table, channel controls, calibrated
risk metrics, and scene-bootstrap intervals are in
[`phase6_probe_ablation.md`](phase6_probe_ablation.md).

All methods use the same 2,643,316-parameter probe, 192-dimensional effect
latent, consequence decoder, nine-channel structured decoder, 1,200 steps,
16-scene batch, and seeds 20260821/20260822/20260823. Pair distances use only
L2-normalized latents. Factual-only supplies its one unique expert candidate per
scene; multi-candidate methods draw four candidates and average within scene
before averaging scenes.

## Main attribution

- Multi-candidate absolute supervision is the clear data benefit: relative to
  factual-only, mean per-scene Effect Alignment rises from 0.2513 to 0.3108 and
  calibrated false-safe rate falls from 0.5699 to 0.2579.
- Unweighted AEE does not beat absolute supervision: alignment is 0.3046,
  false-safe rate is 0.3855, and held-out-family alignment is 0.2529.
- Against global separation, AEE retains 96.9% of Action Gap and improves the
  separation ratio, but reduces Equivalence Leakage by only 12.3%, below the
  predeclared 20% gate.
- Confidence-AEE reduces false-safe errors relative to unweighted AEE, but its
  alignment and safety-AUPRC improvements are not significant under paired
  scene bootstrap. It does not beat absolute supervision on alignment or the
  held-out perturbation family.
- The structured target is learnable and action-dependent: drivable-area and
  lane SDF both beat scene-only, train-mean, and zero controls, and within-scene
  action shuffling significantly degrades them. Gate failure is therefore
  AEE-specific rather than a broken action path or action-invariant target.

Gate 3 is **FAIL**. No shared Qwen/DiT parameter was trained.
