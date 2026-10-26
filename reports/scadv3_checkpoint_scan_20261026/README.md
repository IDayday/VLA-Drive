# SCADV3 intermediate and latest-checkpoint Navtest

Full12146 scenes/136 logs; canonical NAVSIM v1.1; single-candidate shift5 ODE K10; seed2101. All continuation workers exited and exact scene membership/seals passed. The former vla-zt3 admission is closed. No training, checkpoint choice or reward configuration was changed.

| SCADV3 optimizer update | Full Navtest PDMS |
|---:|---:|
| 1,300 | 88.993109 |
| 5,000 | 89.453491 |
| 10,000 | 90.199255 |
| 12,900 | 90.285245 |
| 20,000 | 90.378165 |
| 25,000 | 90.208996 |
| 26,900 | 90.435625 |
| 30,000 | 90.307258 |
| 32,000 | 90.098149 |
| 35,000 | 90.320953 |
| 40,000 | 90.379557 |
| 45,000 | 90.474315 |
| 50,000 | 90.324199 |

Highest observed SCADV3: **45,000, 90.474315**. Fixed50k endpoint: **90.324199**. This is exploratory Test checkpoint scanning; a higher intermediate score is not an independent confirmation.

| Additional checkpoint | Updates | Full Navtest PDMS |
|---|---:|---:|
| IL_LATEST | 86,000 | 87.886957 |
| RL77_LATEST | 24,600 | 90.689893 |
| DINO_FRESH_LATEST | 14,000 | 82.899980 |
| IL77_CONTROL | 77,000 | 88.144945 |
| IL_080000 | 80,000 | 88.157880 |
| IL_087000 | 87,000 | 87.931899 |

Paired analyses preserve strict source, protocol, population, precision and parent checks. IL77-parent RL is compared with its newly reevaluated exact-source IL77 control, not with IL53. DINO14k is compared with fresh DINO5k, not the paused warm-start experiment.

| Paired comparison | Net PDMS points | 95% log-bootstrap interval | G+ | G− | Nonzero→zero / zero→nonzero |
|---|---:|---|---:|---:|---:|
| best_SCADV3_vs_fixed50k | +0.150116 | [-0.081990, +0.379570] | 1.170795 | 1.020679 | 88 / 81 |
| best_SCADV3_vs_CANONICAL50k | -0.038849 | [-0.344757, +0.251171] | 1.417838 | 1.456687 | 137 / 53 |
| best_SCADV3_vs_own_IL53k | +3.458290 | [+2.841994, +4.055019] | 4.995584 | 1.537294 | 128 / 450 |
| RL77_latest_vs_exact_IL77_control | +2.544948 | [+2.051573, +3.047949] | 4.164746 | 1.619798 | 78 / 392 |
| IL86k_vs_IL77k | -0.257988 | [-0.495676, -0.037380] | 0.995481 | 1.253469 | 124 / 83 |
| DINO14k_vs_DINO5k | +8.612728 | [+7.550240, +9.657886] | 12.880140 | 4.267413 | 418 / 1227 |
| IL_080000_vs_IL77k | +0.012935 | [-0.208908, +0.234895] | 1.008714 | 0.995780 | 93 / 90 |
| IL_087000_vs_IL77k | -0.213046 | [-0.448262, +0.008775] | 0.980353 | 1.193399 | 111 / 83 |

NC/DAC/TTC new violations and recoveries are recorded separately in each linked paired JSON; net zero-score counts do not establish safety improvement. Uncertainty is paired log-cluster resampling only, with one training and one inference seed; it does not include seed or checkpoint-selection uncertainty.

New logical single candidates: 170,044; physical including retries: 170,044. This owner elapsed: 1.566 h; summed shared worker occupancy: 68.532 h. Export/parity and historic reuse costs remain separate in native receipts.

Machine-readable details: [results.json](results.json); frozen scope: [registration.json](registration.json). All scores use the same official metric; no oracle, best-of-N, seed selection or protocol switch.

The previous vla-zt3 transport stopped with released actor/worker locks. Original invocation records remain unmodified and their exact process exits cannot be remotely confirmed. Only missing scenes were resumed on fixed hosts under the same rank locks/source/seed. Recorded physical costs are lower bounds because lost-host tail time is unknown; it is not filled with zero. The prior attempt is retained as COMPLETE_WITH_FAILURES, not relabeled as successful.

Full elapsed campaign span, including original failed attempt, handoff and continuation: 4.664 h. This wall time includes idle/lost-host delay, and is not GPU compute time.
