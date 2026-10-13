# Actual expert and scorer training

The fixed experiment completed1,000 expert and2,000 scratch-scorer updates.
Expert best: `expert/best_step_000500.pt`; terminal recovery:
`expert/step_001000.pt`. Scorer best: `scorer/best_step_000750.pt`; terminal
recovery: `scorer/step_002000.pt`. All are under
`/mnt/project/iqe-internal-scorer-20261012/own_wa_v1`.

The first500 scorer updates used128 uniformly sampled natural scenes for
SmoothL1(beta0.1)+0.2×component BCE. The final1,500 used64 natural scenes for
pointwise supervision and64 gap-distribution draws for0.2×ranking. The same
AdamW1e-4/wd0.01, warmup100 and continuous2,000-step scheduler served both
phases. Dropout remained enabled; no external checkpoint initialized the scorer.

There were160,000 natural draws (19,992 distinct scenes) and96,000 comparison
draws (328 distinct pairs). Natural/component and comparison losses have
separate denominators; comparison samples are not mixed into value regression.
q already contains absolute-gap weighting, so ranking does not multiply it
again. The q distribution and two consumed cursors are checkpoint dependencies.

Every recorded expert/scorer loss and gradient norm is finite. Query, Decoder,
trajectory heads and all scorer projections/encoder/heads have actual gradient
evidence in TRAINING_AUDIT.json. Base parameters/buffers stay fixed; cached head
training contains no VLM execution. The real loss/update/freeze evidence is
distinct from scientific specialization evidence in RESULTS.md.

Shared-file latency during early scorer steps was reduced by preloading the
same verified raw tensors into a bounded32GiB CPU reader. No features were
recomputed, precision changed, ordering changed or random state consumed. The
64-real-scene byte/RNG comparison passes. Resume at342 preserved both sampling
streams, optimizer and the original scheduler. The complete bank remains
validated before the workflow can restart at the scorer stage.

Across all steps, recorded compute-update wall mean/p50/p95 is
0.59145/0.52998/0.97696s for the expert and0.36447/0.13498/1.50707s for the
scorer. The latter deliberately includes the earlier slow reader. These are
cached head training costs, not image inference costs or full job elapsed time.
Validation/scoring/cache preparation and CPU preload are additional costs.
The final1,500-update resumed segment reports305.79GPU-seconds; that field is a
segment total and must not be mislabeled as the entire scorer experiment cost.

The full chronological gradient, validation, sampled-exposure and segment
evidence is linked and hashed by EVIDENCE_INDEX.json. No training extension or
post-dev recipe adjustment was performed. Training gain did not translate to
positive held-out selection; the negative result is retained.
