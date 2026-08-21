# Topic decision at the Gate-2 stopping point

## Decision

**Proceed next with Direction A: AEE-WM probe ablations.** This is a concrete
decision for the next research stage, not a claim that the final paper direction
has already been validated by planning metrics.

The evidence supporting this decision is:

- Gate 1 passes: 89.8% of pilot scenes contain both equivalent and divergent
  pairs, 2,095 safety-boundary pairs exist, and geometric/consequence distance
  correlation is only 0.375.
- The 64-scene NAVSIM-v2 traffic-assumption subset is sufficiently stable for a
  first AEE test: candidate hard agreement is 99.48% and pairwise ranking
  agreement is 98.45%. This does not establish true causal counterfactuals.
- The three-seed factual consequence probe is learnable but exhibits the target
  collapse signature: near-zero action shuffle gap, low Effect Alignment, and
  high false-safe rate on unseen local candidates.
- The structured future objective is also learnable versus its fit-only mean
  prior, but scene-only slightly beats scene-action and candidate Effect
  Alignment is only 0.107. Unweighted map MAE does not improve, so Phase 6 must
  retain both the balanced objective and raw map error rather than optimizing a
  favorable metric alone.
- A trajectory-only probe is more action-sensitive but has worse factual error,
  indicating that factual supervision rewards scene priors while failing to
  bind candidate geometry to scene-dependent consequences.

## What would change the direction

- Choose Direction B if the larger IDM coverage or Phase-6 confidence weighting
  reveals materially more LR/IDM conflict than this hashed 64-scene subset.
- Choose Direction C only if Phase-6 world metrics improve but the later
  Qwen+DiT auxiliary loss produces persistent negative FM/world gradient cosine
  and no planning gain.
- Choose Direction D if multi-candidate absolute supervision cannot improve
  structured action sensitivity without relying on invalid/OOD candidates.

The final Direction-A publication criteria remain untested: AEE must beat
absolute-only/global separation and then yield a stable safety/dynamic planning
gain with the world probe removed at runtime.
