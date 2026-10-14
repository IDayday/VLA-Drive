# Matched objective endpoints

Original S00 (SmoothL1) and R00 (soft-target BCE) have identical complete random
initial state hashes and identical scene IDs at every one of 6,000 optimizer
updates. The recorded comparison population, weights and probabilities match.
The global/microbatch, optimizer, continuous learning-rate schedule and candidate
pool are unchanged. Both use semantic context and dropout zero. This is a
read-only analysis of existing results, with no additional training.

The new study validates at 1k/2k as well as 3k/4k/5k/6k; original S00 validates
only at the latter four. Therefore selected-best results alone do not isolate
the objective change. The shared endpoint comparison below avoids that mismatch.
PDMS is percent, differences are percentage points.

| Update | SmoothL1 | Expected-score BCE | Difference | High-to-zero old/new |
|---:|---:|---:|---:|---:|
| 3000 | 90.5919 | 91.6769 | +1.0850 | 15/11 |
| 4000 | 90.9231 | 91.1533 | +0.2302 | 8/14 |
| 5000 | 91.2377 | 91.2717 | +0.0339 | 12/12 |
| 6000 | 90.5425 | 91.3452 | +0.8027 | 16/11 |

BCE has a higher observed mean at each shared endpoint in this seed, but severe
errors do not improve monotonically. At 6k it remains below the same Base
(91.3915) and has 11 high-to-zero scenes. These observations support retaining
expected-reward estimation as a plausible direction; they do not establish
robustness, statistical superiority, or development/test generalization. Neither
old nor new model selection is changed by this retrospective comparison.
