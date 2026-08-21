# Unified Gate-2 result table

`N/A` means not measured in this delivery. In particular, no probe was attached
to the planner, so copying a PDMS/EPDMS value from another experiment would
break attribution. Probe metrics are three-seed means from the quick consequence
target unless noted otherwise.

| Method | Factual Prediction Error | Action Shuffle Gap | Action Gap | Equivalence Leakage | Effect Alignment | False-Safe Rate | PDMS | EPDMS | NC | DAC | TTC | EP | Inference Latency | Trainable Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original Qwen+DiT (frozen feature source) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |
| Constant mean control | 0.3151 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |
| Random untrained probe | 0.4821 | 0.0001 | 0.0278 | 0.0175 | 0.1875 | 0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 903,065 |
| Scene-only factual probe | 0.1686 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 903,065 |
| Trajectory-only factual probe | 0.1789 | 0.0065 | 0.1214 | 0.0561 | 0.3625 | 1.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 903,065 |
| Scene-action factual probe | 0.1660 | 0.0003 | 0.0114 | 0.0062 | 0.2083 | 1.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 903,065 |
| Shuffled-action factual probe | 0.1669 | -0.0000 | 0.0029 | 0.0018 | 0.2216 | 1.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 903,065 |
| Same-parameter no-action | 0.1686 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 903,065 |
| Multi-candidate absolute | Not run (Phase 6) | — | — | — | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Global separation | Not run (Phase 6) | — | — | — | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| AEE | Not run (Phase 6) | — | — | — | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Confidence-weighted AEE | Not run (Phase 6) | — | — | — | — | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

The single-seed structured scene-action probe has balanced factual objective
0.7077, map MAE 0.2347, shuffle gap -0.000046, Action Gap 0.0139,
Equivalence Leakage 0.0061, Effect Alignment 0.1066, false-safe 0.5056, and
2,637,166 trainable parameters.
