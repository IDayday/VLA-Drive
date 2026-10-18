# Complete TRAIN feedback scale analysis

The fixed 4,096-update C1 ledger contains 131,064 scene presentations,
101,592 unique scenes and 1,176 logs. All updates have nonzero feedback;
80,933 scene presentations have a nonconstant four-candidate score group.
The analysis read compact, checksum-bound evidence without reloading a model,
raw rollout payloads or running another score. No optimizer update was made.

The proposed denominator is the FP64 sample standard deviation (correction 1)
of the original FP32 scene-centered advantages across one complete global
optimizer rollout. All four candidates and zero groups remain in its denominator.
The numerator is not globally re-centered; epsilon remains the independently
specified `1e-10` outside the square root.

| Update population | Updates | Median std | Median multiplier |
|---|---:|---:|---:|
| All | 4,096 | 0.00109936 | 909.620 |
| Safe, progress-only | 2,406 | 0.000932583 | 1,072.290 |
| At least one discriminative D/C/T metric | 1,119 | 0.0730517 | 13.689 |

The all-update multiplier range is 6.793–2,963.679. None of the predeclared
numerical flags fired: there was no nonfinite statistic, zero denominator,
epsilon-dominated positive standard deviation or unresolved rounding bound.
The largest conservative rounding-energy bound divided by advantage energy is
`5.067594008e-8`. This is a bound, not a measured causal attribution to rounding.
All four fixed policy-version windows retain nonzero and safety-discriminative
support. Exact counts and statistics are in `RESULTS.json`.

The approximately 78.33 ratio between the two strata's median multipliers is a
material risk: this adaptation changes relative weights across optimizer updates
and against BC. It preserves relative gaps only within the same rollout. It is
not a lossless fix, safety guarantee, or evidence of improved planning. The
strata are diagnostics, not a new sampler or reward. Different strata may overlap;
repeated scenes and logs are not independent observations.

Actual execution used 15.750 seconds wall, 6.787 seconds CPU and 98.4 MB peak RSS.
It read 40,911,941 JSON bytes, with 41,047,129 bytes conservatively precharged
including source/bootstrap allowance. The immutable result is 1,109,160 bytes.
The separate 64 MiB reservation and independent timeout were respected; no
original analysis deadline or counter was reset. Controls and exact artifact
hashes are recorded beside the result under the campaign audit directory.

This report establishes available TRAIN signal and numerical scale only.
Implementation review and a bounded full-pipeline profile remain necessary
before any formal comparison. No DEV or navtest data were used here.
