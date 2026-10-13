# Dataset-level comparison objective

The new baseline does not reuse the old per-scene-normalized ranking objective.
For two candidates it defines `w = abs(R_expert - R_base)` only when both
official labels are valid and `w > 0.01`. The target is
`sum(w * softplus(-sign(delta) * predicted_delta / 0.1)) / sum(w)`.

`iqe/internal_baseline/sampling.py::TwoStreamSampler` samples the comparison
stream with replacement using `q = w/sum(w)`. Its ordinary pair-loss mean is an
unbiased estimate of this dataset objective. No additional factor `w` appears
in that stream's loss. The natural stream alone supplies value/component loss.
The comparison distribution is frozen by bank hash and q hash, not balanced by
which expert wins.

The reference gradient test holds prediction difference and ranking direction
fixed, using true gaps **0.95 and 0.02**. Its gradient ratio remains **47.5**.
The sampled unweighted pair loss has equal per-occurrence gradient; the sampling
probabilities have ratio 47.5. Multiplying by the gap again would incorrectly
make that ratio 2256.25. Per-scene division by the gap would instead erase it.

Tests in `tests/iqe/test_internal_baseline.py` cover sign, ties, invalid masks,
zero denominator with graph-connected backward, no pointwise leakage from the
comparison stream, SmoothL1(beta=.1), full-step microbatch denominators, sampler
peek/consume and restoration. `test_internal_ddp.py` checks both Gloo and NCCL
with unequal rank denominators and a completely empty-label rank against the
single-process gradient. The existing trainer scales local numerators by
world-size/global-denominator before DDP averages gradients.

Actual evidence:

- Full CPU/Gloo regression: 291 passed, 2 gated GPU tests skipped
  (`final_regression.log`, before the additional NCCL test was added).
- New comparison-objective Gloo and NCCL executions: 2 passed
  (`internal_ddp_nccl.log`).
- Existing two-rank NCCL exact-resume/unequal-label test: passed
  (`nccl_resume_final.log`). Existing NCCL terminal-resume test also passed.
- Exposure reconstruction verifies the actual sample IDs, rather than inferring
  exposure from configured epoch counts (`EXPOSURE_AUDIT.json` in artifacts).

The initial optional GPU invocation was refused by the resource guard while
pressure workers occupied the devices. After explicit user authorization those
identified pressure workers were paused. A second guard refusal observed the
previous test's transient 11% utilization with zero allocated memory; the test
passed when rerun alone after the GPU became idle. Neither refusal is presented
as an executed numerical failure or silently removed from the logs.
