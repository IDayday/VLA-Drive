# Convergence and interpretation

The completed graph pair used7284scenes,58272presentations,8passes,3642optimizer updates per variant with a warmup/cosine schedule. The downstream image planning pair used the same7284scenes,29136presentations,4passes,1821updates each. Extra training is matched across each pair; reaching either limit is not a convergence criterion.

| Graph variant | Holdout ego ADE,6passes | Holdout ego ADE,8passes | Holdout matched-agent ADE,6passes | Holdout matched-agent ADE,8passes |
|---|---:|---:|---:|---:|
| Random whole-actor subset |1.2848|1.2061|6.4688|6.5556|
| All futures hidden |1.2090|1.0550|6.5963|6.7043|

These64scenes come from59 excluded logs. Ego error still changes between the last two milestones; agent error does not improve monotonically. Earlier checkpoints give different rankings. The curves do not establish a stable joint optimum. Matched-agent motion point coverage is only11.46%, and stationary current-position predictions have2.4374m ADE, below either generated graph's~6.6m. This current perception/forecasting limitation prevents a broad world-model-quality claim. Every GT row and unmatched instance is retained in the object CSV.

The originalDiT planning bridge's fixed64training-scene ADE decreases from0.07716m to0.06759m(randommask)/0.06806m(control), with changes continuing from the2-pass checkpoint. These are training imitation diagnostics, not heldoutPDMS. Full1696development planning scores are93.1446baseline,93.3270randommask,93.2144allmask. Masked-minus-control is+0.1126points, with16-log cluster95%CI[-0.0052,+0.2198]. The scene interval does not cover training-seed uncertainty. The current finding is INCONCLUSIVE, neither proof of a stable benefit nor evidence that the research direction fails.

The completed BEV pair used a separately fixed4-pass/1821-update schedule and identical fresh initialization, graph weights, optimizer and ego noise. Current occupancy, tracked displacement, pair geometry and auxiliary all-hidden graph FM were enabled together in the task-on arm at fixedweight0.1. All1/2/4-pass task-head evaluations and final planning are retained. There was no PDMS-driven retuning, checkpoint selection or extension.

| BEV task-on metric on1696 scenes |1pass|2passes|4passes|
|---|---:|---:|---:|
| Current occupied IoU |0.134855|0.168006|0.161006|
| GT-cell future motion ADE(m) |5.266427|4.910957|4.669068|
| Pair geometry loss |0.061149|0.057074|0.054934|

Task-off final IoU is0.029327, GT-cell motion ADE6.813422m; the stationary dense comparison is5.380311m. These dense metrics use GT current occupied cells and are not end-to-end detection/forecasting. Motion continues improving at the last milestone while occupancy is nonmonotonic. This is evidence of learnable tasks, not established convergence or planning benefit.

Full1696 final BEV planning is93.312266 without auxiliary tasks and93.195079 with tasks. Task-on minus task-off is−0.117186percentage points,95%log-clusterCI[−0.325411,−0.001443]. The interval is slightly below zero for this fixed recipe and training seed; the auxiliary objectives did not improve planning. It is not a cross-seed result or evidence that the general research direction is invalid. These exploratory intervals have no multiplicity correction. Frozen upstream perception has only13.26% geometric recall and13.38% motion-point coverage on this full split; matched-agent motion is worse than the same-predicted-centre stationary comparison. Those limitations and continuing task curves support overall INCONCLUSIVE, not POSITIVE.

The exact-one-actor mode is separately implemented and has a real16-step engineering run. The long research pair used aBernoulli subset of whole trajectories, with50%all-hidden scenes; it is not an exact-one experiment. The16-step run supplies no convergence or comparative evidence.

Combined ledger cap is24000updates/48GPU-hours, including previous2596updates and failed checks. Final usage is21835updates/7.169199GPU-hours, leaving2165updates. A further matched1821×2 seed pair cannot fit; it remainsNOT_RUN under this budget. Shortening only the second-seed comparison to claim stability would change the exposure/schedule and is not done. One of two allowed convergence-repair hypotheses was used (predicted-centre residual coordinates and agent5m noise scale); all experiments and unsuccessful checks remain recorded. Earlier V1 work is a separately sealed campaign and is not erased by this accounting.
