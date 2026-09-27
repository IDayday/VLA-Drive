# P2: measurement-driven hypotheses (not yet trained)

P0 establishes both no-object collapse and severe centre concentration, rather than an exclusively metric-induced failure. No old 200-step label experiment is reused as evidence.

Initial V1.1 configuration to implement and test before any optimizer update:

1. Fixed, observation-independent spatial references spanning the original ROI; encode references into agent queries. Predict normalised residuals and decode to metres. References do not depend on GT count or boxes. Geometric training cost must not be dominated by class swaps over several metres.
2. Unmatched-slot supervision eligibility evaluated at fixed current-observation references, not learned predicted centres. Matched slots remain supervised; ignore unknown FOV and overflow appropriately. This removes the escape through out-of-ROI predictions without calling unobserved space empty.
3. Motion regression against `gt_future - gt_current`, with invalid points masked before arithmetic. Predicted future = predicted current centre plus learned displacement, but motion loss does not move the centre through this addition. Evaluate future and stationary-centre baselines on identical detection matches.
4. W_PRE and W_POST use the same Reader, fixed K, reference encoding, structural head size, seed, data order, and exposure. W_PRE reads projected Reader output; W_POST reads frozen Qwen output. No DiT, visual or Qwen updates. Start with first 16 original overfit-manifest scenes, then the unchanged full64. Report full-GT coverage, not only assigned targets.
5. Per variant max1000 updates, effective batch8. True loss-sum/count normalisation across accumulation; do not retain eight full-Qwen autograd graphs. Log classification histograms, slot dispersion, metres, task gradients, clipping and parameter deltas.

Only if W_PRE learns and sufficiently trained W_POST fails may one LoRA experiment be considered. No claim of original action fidelity after shared-weight changes.

Status: NOT_RUN. No claim that proposed parameterisation has recovered learnability.
