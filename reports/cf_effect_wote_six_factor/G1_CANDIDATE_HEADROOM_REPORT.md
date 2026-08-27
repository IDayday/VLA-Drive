# G1-R2 Candidate Headroom Report

This is the frozen **WoTE base-anchor selector**, not a full WoTE leaderboard result.
Trajectory offsets are disabled and every score is an independently recomputed
six-factor 4-second label.

| Selector | Selected score | Oracle score | Gap | Better-scene fraction | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| WoTE base-anchor selector | 0.486507 | 0.873557 | 0.387049 | 0.965000 | NOT_APPLICABLE_TOO_FEW_ZERO_SCENES |

The scene-paired bootstrap 95% interval for oracle gap is
`[0.352878, 0.420254]`
using 2,000 resamples and seed 20260827.

DDC diagnostics: selected mean `0.635000`, oracle mean
`0.952500`, and no-DDC/full-oracle reversal fraction
`0.160000`.

Gate: `PASS`. Final verdict: `WOTE_CANDIDATE_BANK_VIABLE`.
The action-effect hypothesis remains `UNTESTED`; no effect scorer, forward model,
inverse model, WoTE training, trajectory offsets, or extra candidates were run.
