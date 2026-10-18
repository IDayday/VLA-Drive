# Rollout scale TRAIN evidence — implementation for independent review

Status: author implementation and synthetic tests complete, not independently
approved or dispatched. No actual compact feedback output was read by the author.
The running/completed primary analyzer and its fixed plan are unchanged.

This is a dedicated descriptive post-processing call for the fixed primary
RESULT, not a training implementation. The proposal is
`ROLLOUT_SCALE_PROPOSAL.md` §4 at `b1a8cf49b98a718e1b509dc7000c7db3e3c8fb94`,
SHA256 `a32d9e405316bdc257ce283343e23b6fb05d3ed54086d42a0e2243dcce82d69c`.
The primary input pins, supplied by Lead, are:

- Plan SHA256 `2d50faf1c690bd8ebc487504a32ecae9637432e4742c561a9ea5f6442a995b06`.
- Primary/helper source SHA256 `30bb3237019a463e68bb5c0686d902521d814ee99bec228a94de7b537f92f4ca`.
- Primary RESULT SHA256 `0a6a84d3652122aeb69760b5237a56c71fec269c16b4066e78a0ddb05876d8cc`,
  327728 bytes. The author did not read this actual RESULT.

The helper bytes are verified before compiling those same bytes. Only its
standard-library JSON/checksum/quantile/atomic-write helpers are used; no
Torch/NumPy, CUDA, model, scorer or original RESULTS.pt is loaded. The call
verifies COMPLETE, all eight shard receipt hashes, every compact ledger hash
and original COMPLETE/result pins, unique boundaries 0..4095, and exactly
4096 ledgers / 131064 presentations / 524256 candidates. Nonzero C1 groups
also reconcile against the primary count, rather than substituting a hardcoded
expected result. All eight worker locks and the primary merge lock are acquired
read-only and held through the compact read phase, so active workers or merge
cause refusal. Outputs outside primary/original run are checked before writes.

For each update, stored float32 A values must remain exactly representable as
float32; the descriptive statistics promote those values to float64. N includes
every original G4 candidate and zero group, including the 24-scene tail. Sum,
mean, centered two-pass M2, correction=1 std, energy and max-absolute-A are
recorded. Epsilon is fixed at 1e-10 outside sqrt. The numerator is never globally
centered again and no normalized A is generated. Allzero updates have std=0,
gain=null and a separate category. Zero-energy rounding ratios are null.

Numerical flags identify N<2, nonfinite values/statistics, std=0,
nonzero-A/std=0, positive std<=epsilon, and constant nonzero A groups. Nonfinite
compact input fails the invocation and preserves an error. The conservative
rounding comparison is N*delta²/sum(A²), where delta is the stored maximum
float32 rounding residual; ratio>=1 means UNRESOLVED, not proved dominance.
The available compact receipt does not separate original float64-score casting
from float32-centering residual, which is stated in the output.

The fixed all/nonzero/allzero/safe-EP-only/event-present/metric-discriminative
strata preserve overlapping D/C/T flags. Safe EP-only requires at least one
active group, every active group EP-only, and all D/C/T event counts zero.
The original four policy windows are unchanged. Each stratum/window reports
update/scene/candidate presentation counts, unique scene/log unions, nonzero
presentation/unique counts, exact pooled per-update quantiles, and both summed
energy and summed rounding bound with a ratio of sums. Per-update distributions
remain separate from that energy-weighted ratio. The compact update table gives
the original numerical values and fixed flag definitions, without scene-ID dumps.

No reward, gradient, coefficient or approval is generated. Across-update scale
would change relative update weighting and weighting against BC, including
EP-only versus event-containing updates. Within-batch scaling arithmetic does
not establish safety, benefit or a whole-training gradient ratio; repeated
scene/log presentations are not iid.

This once-only invocation has its own EXECUTION identity and fresh 60-second
deadline. It never resets the primary EXECUTION deadline or counters. Before
any per-ledger read, it verifies actual primary cumulative logical reads
(eight final budgets + merge budget + original plan metadata) plus the whole
64 MiB derivative allowance are strictly below the original 32 GiB. Lead
reported primary logical reads of 29148722313 bytes, so the expected remaining
allowance is sufficient; the executable rechecks receipts rather than trusting
that message. No partial-counter reuse or resume is supported.

Every input read is preceded by an in-memory size+1 admission under 64 MiB.
The bounded bootstrap plan is read once and charged; the 128 KiB source allowance
covers the new source and pinned helper. The initial receipt reserves the full
64 MiB. Counters flush every 64 reads, on success and on handled failure. After
SIGKILL a checkpoint can lag at most 63 completed reads; the full allowance
remains reserved, cannot be reclaimed from that lag, and no COMPLETE result is
published. This avoids the primary merge's costly per-record fsync pattern while
retaining all hash checks. Shared-storage throughput for this call is unmeasured;
the 60-second limit is not a prediction or an automatic retry authorization.

One assigned CPU, CUDA visibility empty, 60 CPU seconds, 60 wall seconds,
sampled RSS <=2 GiB, and total retained outputs <=2 MiB are enforced. Read/compute
boundaries check CPU/RSS; a process signal guard covers bootstrap and analysis.
The output limit reserves 16 KiB for execution/budget/error controls and does not
silently change exact quantiles if exceeded. Abrupt process termination can leave
partial atomic output; such a call has no successful RESULT and cannot resume.

Validation: 10 synthetic-only CPU tests passed in 1.172 seconds, including
4096 synthetic updates within the output envelope, exact two-pass/correction1
arithmetic, the no-second-centering counterexample, tail96/allzero/nonfinite/
constant/small-std controls, overlapping risk strata, four windows, unique versus
presentation counts, input hash failures, original cumulative budget rejection,
active-lock/INCOMPLETE refusal, precharge and batched flush, unchanged primary
files, strict --help/invalid CLI and pre-write output isolation. No real compact
output, original payload, DEV, model or GPU was used by these tests.

After independent review, Lead alone may dispatch one call:

```text
python tools/st_world_policy/analyze_feedback_scale.py \
  --primary-plan /path/to/the/frozen/PLAN_CANDIDATE.json \
  --output-dir /new/independent/derived/output/directory \
  --cpu ASSIGNED_CPU
```

The CLI rejects abbreviated/unknown parameters and exits nonzero on failure.
An existing output directory is refused. No actual derived call was dispatched
as part of implementation or author validation.
