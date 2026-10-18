# Same-request teacher-channel prediction diagnostic

The native numerical result is complete and hash-bound by the child accounting
envelope (return code 0). **The supervisor status remains FAILED_NO_AUTO_RETRY**:
`OWNED_DESCENDANTS_AFTER_LEADER_EXIT`; cleanup reports
0 live owned processes
after verifying 10 identities.
The following is descriptive analysis of completed artifacts, not a rewritten
successful execution receipt or independent artifact-adoption approval.

The 256-channel endpoint has **lower common-coordinate error** on this fixed
8-scene training subset: 1.53782842 versus 1.54403718 MSE on the exact same
first 256 PCA coordinates (+0.402% relative reduction).
This is one read-only endpoint comparison, not a convergence or planning result.

| Diagnostic block | PCA256 model | PCA384 model |
|---|---:|---:|
| Same first 256 coordinates: MSE | 1.53782842 | 1.54403718 |
| Same first 256: SSE / zero-prediction SSE | 0.47966716 | 0.48160374 |
| Extra coordinates 257–384: MSE | Not predicted | 0.49399274 |
| Extra coordinates: SSE / zero-prediction SSE | Not predicted | 0.96959683 |

The PCA384 full-384 MSE is 1.19402236. It is recorded for
bookkeeping only: comparing that number directly with PCA256's 256-dimensional
MSE changes the coordinate set and is not the aligned comparison. The tail's
zero-prediction MSE is 0.50948262; zero is a
diagnostic reference, not a new model. PCA256's missing 128 coordinates are not
filled with zeros and are not reported as zero error.

There are 6/8 scenes with lower common
error for PCA256 and 2/8 for PCA384.
Per-scene differences and all 384 reported component summaries are preserved in
the CSVs. These request positions are correlated within scenes; they are not
2,304 independent scientific trials.

## Checked execution boundary

- Both real endpoints completed 5,000 updates with 159,992 scene presentations,
  matching the registered spatial recipe except teacher output channel dimension.
- Selection was score-free and fixed before running: eight train scenes from the
  previously fixed 64-scene pilot, with eight named log segments. This is not a
  full-population, dev, navtest or independent generalization sample.
- Same three views, 18×24 native teacher grid, 96 positions per view: 2,304 unique
  requests, 2304 valid; 589,824 common and 294,912 tail
  scalar elements. Same request IDs and masks were reused across both endpoints.
- Native PCA256 targets are bitwise equal to the PCA384 target prefix; max
  absolute difference is zero. The common train-only PCA basis identity is
  `268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46`. Original-channel mean/std normalization occurs
  before PCA; projected components are not whitened. Different component energy
  remains, so tail MSE alone does not measure greater or lesser usefulness.
- Both runs used deterministic FP32 eval with TF32 off and unchanged RNG. Each
  ran eight `encode_inputs` and eight sparse readouts. No training forward, Flow
  Matching, action generation, optimizer step or backward occurred. Finite-output
  checks passed at all valid requests. The report reads JSON only, not tensors.
- Runtime/source and input hashes are in SUMMARY.json. The native result is
  COMPLETE; the original supervisor failure is preserved as stated above.
  The failed v1 attempt remains separate and was not relabeled successful or
  included as completed comparison evidence. No rerun or retry is performed.

## What this says about the hypothesis

The aligned common-coordinate comparison directly tests whether removing the
extra 128 supervised output coordinates coincides with better reconstruction of
the retained coordinates after this fixed training budget. Its observed direction
is evidence for this small subset and endpoint only. It does not identify whether
the cause is gradient interference, optimization difficulty, regularization,
finite-budget dynamics or sample variation. No gradient conflict was measured.

The observed common-coordinate difference is only about 0.4%, so this endpoint
provides at most a weak directional signal for easier alignment of the retained
coordinates. The extra 128 coordinates reduce their zero-reference SSE by only
3.040% on these requests, compared with
51.840% for the PCA384 model's common coordinates.
That supports checking whether the low-energy tail is weakly predicted; it does
not establish that its training gradients harm the useful channels. No new
whitening, loss reweighting or smaller-channel training run is implied.

The teacher input resolution stayed 384×288 in both arms; student input, world
queries (48 per view), sparse sample count, loss weights and the 20k learning-rate
time axis were held fixed. This diagnostic cannot show that higher teacher
resolution helps or that one channel dimension improves driving. Such claims
need the separately registered held-out planning evaluation and, for the proposed
mechanism, direct gradient/optimization evidence. Do not substitute these FP32
eval MSEs for the original BF16 training loss, or infer asymptotic convergence
from a single 5k endpoint.

The per-arm elapsed times include strict checkpoint verification/reconstruction
and model loading. They are not per-update training throughput or isolated
readout latency. No new teacher cache or model optimization was performed.
