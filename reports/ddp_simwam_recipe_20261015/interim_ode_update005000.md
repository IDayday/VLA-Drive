# Fixed update5000: complete single-ODE resampling

Both registered variants completed their first fixed5000 evaluation on all
1,696 RL_DEV scenes /16 logs and the fixed512 TRAIN diagnostic scenes /343
logs. Each uses original raw-observation VLAAgent inference, one trajectory per
ODE grid, seed2101 and canonical v1.1 PDMS. No search, scorer-based selection,
teacher selection or candidate-bank trajectory is deployed. Every shard passed
the direct original-deployment output check. E_NATIVE and E_SHIFT5 are separate
protocols, never a best-of-two selection.

These are intermediate checkpoints, not final20k or Navtest results. The
endpoint selection remains fixed at20k for both variants. All later registered
observations remain required. The DEV population was unseen by RL updates,
but may have been seen by SFT; it has very few SFT TTC failures. Confidence
intervals resample whole logs with scene weighting and do not include training-
or inference-seed uncertainty.

| Complete DEV1696 | SFT | R_SOURCE 5k | R_CANONICAL 5k |
|---|---:|---:|---:|
| E_NATIVE PDMS /100 | 93.6192 | 94.0763 | 94.3121 |
| E_NATIVE change vs matched SFT | — | +0.4570 | +0.6929 |
| E_NATIVE change, log-bootstrap95% CI | — | [0.0299,0.9903] | [0.4522,0.9925] |
| E_SHIFT5 PDMS /100 | 93.5747 | 93.9778 | 94.3214 |
| E_SHIFT5 change vs matched SFT | — | +0.4030 | +0.7466 |
| E_SHIFT5 change, log-bootstrap95% CI | — | [-0.0866,1.0067] | [0.3509,1.2863] |

The native deployment gain is present at this intermediate budget. The source
shift5 confidence interval includes zero. Neither result supports a claim
about final Navtest reaching92: the matched SFT DEV baseline already exceeds92,
and DEV is not interchangeable with the historically observed Navtest baseline.

The direct canonical-minus-source native difference is+0.2358 points with
log-bootstrap95% CI[-0.2238,0.5949], which includes zero. The shifted difference
is+0.3436 [0.1359,0.6564]. Thus the observed native mean ranking is not a
statistically resolved winner on this16-log population. The paired contrasts
are retained in [CANONICAL_minus_SOURCE_update005000_DEV.json](CANONICAL_minus_SOURCE_update005000_DEV.json).

| E_NATIVE DEV safety / quality | SFT | R_SOURCE 5k | R_CANONICAL 5k |
|---|---:|---:|---:|
| NC /100 | 100.0000 | 100.0000 | 100.0000 |
| DAC /100 | 99.2335 | 99.3514 | 99.5283 |
| TTC /100 | 99.9410 | 99.0566 | 99.9410 |
| EP /100 | 85.8065 | 87.6816 | 87.1155 |
| Comfort /100 | 99.8821 | 99.8821 | 99.8821 |
| DDC /100 | 98.8797 | 98.8797 | 98.8797 |
| NC newly violating / recovered scenes | — | 0 /0 | 0 /0 |
| DAC newly violating / recovered scenes | — | 4 /6 | 1 /6 |
| TTC newly violating / recovered scenes | — | 16 /1 | 1 /1 |
| New joint violation among1,683 fixed SFT-safe scenes | — | 20 | 2 |
| Nonzero→zero / zero→nonzero scenes | — | 4 /6 | 1 /6 |
| Final zero-score scenes | 13 | 11 | 8 |
| Positive score-gain sum, raw0–1 units | — | 18.0609 | 13.1365 |
| Negative score-degradation sum, raw0–1 units | — | 10.3098 | 1.3856 |
| Any negative score change | — | 209 | 201 |
| Score decrease of at least0.005 raw units | — | 20 | 2 |

Zero-score count decreases for both, while source TTC worsens. Reporting only
the zero count would conceal that regression. Source10:5:2 increases EP more
but has a larger negative-score tail and more new safety violations. Canonical
has the higher native mean with fewer new violations at this fixed stage.
It still creates two new violations on the fixed SFT-safe population, so it
does not establish per-scene safety preservation.

Source native simulated mean forward displacement rises from17.4757m to17.7973m
(+0.3216m), and simulated mean segment speed from4.5329 to4.6134m/s. These
measure actual simulator consequences. Requested trajectories, simulated
states, raw progress, arrival-time description and jerk diagnostics are stored
separately. The saved-batch gradient analysis in
[interim_training_10k.md](interim_training_10k.md) finds partial PG/anchor conflict,
particularly at the last stochastic step, but cannot isolate the weak
longitudinal anchor as the cause of these population changes.

## TRAIN diagnostic is a separate population

| Fixed TRAIN512, directed diagnostic | SFT | R_SOURCE 5k | R_CANONICAL 5k |
|---|---:|---:|---:|
| E_NATIVE PDMS /100 | 80.3316 | 83.1492 | 84.4715 |
| E_NATIVE gain | — | +2.8176 | +4.1399 |
| E_SHIFT5 PDMS /100 | 79.1099 | 82.2102 | 83.9018 |
| E_SHIFT5 gain | — | +3.1002 | +4.7919 |
| Native NC new / recovered | — | 2 /0 | 0 /0 |
| Native DAC new / recovered | — | 10 /25 | 8 /27 |
| Native TTC new / recovered | — | 17 /4 | 6 /5 |
| Native SFT-safe new joint violation | — | 24 /434 | 13 /434 |

This preselected difficult diagnostic is not an unbiased full-TRAIN estimate
and is not an unseen evaluation population. Both native and shifted policies
absorb useful changes here, with safety costs much larger for source reward.
The complete DEV result supplies independent RL-update-unseen evidence of a
smaller mean gain; the final matched Navtest remains necessary.

The supported interim interpretation is that the transferred complete recipe
can change original ODE deployment behavior, rather than only improving the
random sampler. The source reward's stronger progress weighting is associated
with an adverse TTC tradeoff in the matched comparison. This is evidence for
the specified reward contrast, not attribution of the complete recipe's gains
to LoRA, global std, the partial window or the MSE anchor separately. No
additional algorithm or hyperparameter variant has been launched.

Machine-readable scores and costs:
[R_SOURCE DEV](R_SOURCE_update005000_DEV.json),
[R_CANONICAL DEV](R_CANONICAL_update005000_DEV.json),
[R_SOURCE TRAIN](R_SOURCE_update005000_TRAIN_DIAGNOSTIC.json),
[R_CANONICAL TRAIN](R_CANONICAL_update005000_TRAIN_DIAGNOSTIC.json).
Each complete-population summary verifies all per-scene payload hashes and
retains the full denominator. Timing excludes model-loading time from the
per-worker scoring loop and must not be presented as total job wall time.
