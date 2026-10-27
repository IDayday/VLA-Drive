# Latest Qwen-Flow Navtest — 2026-10-27

All three full12146-scene/136-log evaluations completed. Canonical NAVSIM v1.1; single-candidate source shift5 ODE K10; inference seed2101. No best-of-N or training change.

| Checkpoint | Updates | Full Navtest PDMS |
|---|---:|---:|
| IL098000 | 98,000 | 88.004393 |
| RL77_FINAL050000 | 50,000 | 90.613716 |
| DINO_FRESH024000 | 24,000 | 79.268453 |

| Matched paired comparison | Net PDMS points | 95% log-bootstrap interval | New zero / recovered zero |
|---|---:|---|---:|
| RL50k_vs_own_IL77 | +2.468771 | [+1.960880, +2.948727] | 102 / 372 |
| RL50k_vs_RL24_6k | -0.076177 | [-0.298173, +0.128694] | 92 / 48 |
| IL98k_vs_IL80k | -0.153487 | [-0.385589, +0.084165] | 91 / 73 |
| DINO24k_vs_DINO14k | -3.631527 | [-5.592950, -1.552585] | 1114 / 569 |

NC/DAC/TTC new violations/recoveries, G+/G− and components are in the paired JSON. Intervals exclude seed and Test checkpoint-selection uncertainty. The completed50k RL endpoint is compared with its exact IL77 parent and its own24.6k intermediate; old IL53-parent models are not treated as matched controls. DINO is fresh three-frame IL, distinct from the paused warm-start model.

Logical single candidates: 36,438; physical including retries: 36,438. Compute owner wall: 2.185 h; summed shared worker occupancy: 18.665 h. Export/native parity and historical calibration costs are separate job receipts; they are not free data.

Frozen registration: [registration.json](registration.json). Full data: [results.json](results.json).
