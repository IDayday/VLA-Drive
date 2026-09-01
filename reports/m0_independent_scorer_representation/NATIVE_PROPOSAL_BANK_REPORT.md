# M0 and DrivOR native 64-proposal comparison

## Scope

This comparison uses each public checkpoint's **own native proposal generator and
own native scorer**.  Every bank contains 64 trajectories with 8 poses.  All
12,146 Navtest scenes and 136 segment logs (44 physical logs) completed official
offline PDM scoring with zero invalid scenes.  Best-of-64 is an offline oracle
upper bound and is never used during deployable inference.

DrivOR checkpoints and representations are used only for this analysis.  They
are not inputs, initialization, or cached features for the independently trained
M0 scorer.

| Native bank | Selected PDMS | Best-of-64 | Mean candidate | Median candidate | Top-5 oracle mean | Scorer regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 public | 0.909594 | 0.984112 | 0.795276 | 0.835676 | 0.972880 | 0.074518 |
| DrivOR original, 25 epochs | 0.936907 | 0.993342 | 0.797153 | 0.849673 | 0.985742 | 0.056436 |
| DrivOR scaled, 134k simulation data | 0.945829 | 0.994094 | 0.804264 | 0.861690 | 0.988241 | 0.048265 |

## Where the selected-score gap comes from

For two banks A and B,

```text
selected(A) - selected(B)
= [oracle(A) - oracle(B)] - [regret(A) - regret(B)].
```

| Comparison (A - B) | Selected delta | Oracle-ceiling delta | Regret delta | Selected-delta 95% physical-log bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| M0 - DrivOR original | -0.027313 | -0.009230 | +0.018082 | [-0.032187, -0.021896] |
| M0 - DrivOR scaled | -0.036235 | -0.009982 | +0.026253 | [-0.043491, -0.028429] |
| DrivOR original - scaled | -0.008923 | -0.000752 | +0.008171 | [-0.013860, -0.004320] |

The original DrivOR/M0 candidate-mean difference is only +0.001877 and its
physical-log bootstrap interval crosses zero.  Nevertheless, DrivOR's top tail
and oracle ceiling are substantially higher.  More importantly, 0.018082 of the
0.027313 selected-score gap is reduced scorer regret.  With the scaled weight,
0.026253 of the 0.036235 gap is reduced regret.  Thus better selection explains
the majority of the final gap, while a better proposal tail supplies a smaller
but real upper-bound advantage.

Scaling DrivOR from the original to the 134k-data checkpoint adds 0.008923
selected PDMS, of which 0.008171 comes from reduced scorer regret; its oracle
ceiling changes by only 0.000752 and that oracle delta is not clearly separated
from zero at the physical-log level.

## Selected factor differences

Values below are DrivOR minus M0.  Positive values favor DrivOR.

| Selected factor | Original - M0 | Scaled - M0 |
| --- | ---: | ---: |
| no-at-fault collision | +0.008151 | +0.008851 |
| drivable-area compliance | +0.016713 | +0.019019 |
| driving-direction compliance | -0.000370 | -0.000947 |
| time-to-collision within bound | +0.025111 | +0.027169 |
| ego progress | +0.014709 | +0.031234 |
| comfort | +0.000165 | +0.000082 |

The selection advantage is concentrated in TTC, drivable-area compliance,
progress, and collision avoidance.  Driving-direction compliance and comfort do
not explain the gap.

## Geometry and complementarity

The banks are locally similar but not identical.  Mean nearest cross-bank ADE
is 0.228/0.250 m for M0/original DrivOR and 0.221/0.257 m for M0/scaled DrivOR.
However, the two oracle trajectories differ by 1.197 m ADE for the original
comparison and 1.061 m for the scaled comparison.  A union best-of-128 reaches
0.995658 (M0 + original) or 0.995892 (M0 + scaled), showing that the best tails
remain complementary.

An important nuance is that M0's average proposal is safer on collision and TTC
than the scaled DrivOR average, while scaled DrivOR has better average road
compliance, comfort, and progress.  DrivOR's native scorer then selects a much
safer-than-average member of its more varied bank.  This is consistent with a
system that benefits jointly from proposal-tail diversity and a scene-aware
scorer, rather than from uniformly improving all 64 trajectories.

## Artifact lineage

- M0 checkpoint SHA256: `7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d`
- DrivOR original checkpoint SHA256: `e1a678f201e4f1ab93d117caad42782cd7ead293bdced2b5f80212bc92426ae3`
- DrivOR scaled checkpoint SHA256: `617e22c5ebcf8b24c542d42a470514b09c3cadce1a2630071d49f2d422d76672`
- M0 proposal submission SHA256: `d37a20fa258fef4f68ca7bbb37aff1f6cb6ac968ab4b543eb3c48fcb26935f6d`
- M0 official candidate-score matrix SHA256: `9ecfbea30f3bc51bcf59b8c6145e4fd87c4c36e9f5935cccc44b7aa7050f33e4`
- DrivOR original proposal submission SHA256: `63ab306a5e0632595f2a8eb404e7055fe43855f0653eab9d758e37d4c4410a9b`
- DrivOR scaled proposal submission SHA256: `cde9fed1aaabfa912d2669928a54ff96fe0c10e2554ef870e400b83d9cb7c977`

Machine-readable results and per-scene comparisons are in the sibling
`m0_vs_drivor_original_native64_v1`, `m0_vs_drivor_scaled_native64_v2`, and
`drivor_original_vs_scaled_native64_v1` directories.
