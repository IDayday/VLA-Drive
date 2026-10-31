# ABCD intermediate full Navtest — 2026-10-31 UTC

New complete result: **B IL10k PDMS80.692819**. All12,146 scenes, invalid0, single candidate, clean-x K10shift5, seed2101, canonical NAVSIMv1.1. Native/export parity PASS; all8worker shards COMPLETE and owned workers exited. This is Test, not training reward or DEV. Test has been repeatedly observed historically.

| B optimizer updates | PDMS |
|---:|---:|
| 4000 | 62.047152 |
| 5000 | 66.551068 |
| 7000 | 80.557548 |
| 10000 | 80.692819 |

B10k−B7k = **+0.135271 points**. The four point estimates increase, but the latest increment is small. They do not establish a final architecture ranking, a higher ceiling, or training-seed stability. All four registered100k IL budgets and own-parent DEV selection remain unchanged.

| Metric | B10k,0–100 |
|---|---:|
| NC | 96.579121 |
| DAC | 90.614194 |
| TTC | 90.820023 |
| EP | 76.338951 |
| Comfort | 99.415445 |
| DDC | 97.801745 |

A13k is also complete: **PDMS80.007834**, versus A9k80.156844 (**−0.149010 points**). All12,146 scenes/invalid0, native parity PASS and all workers exited. This small decrease is a checkpoint point estimate, not an architecture ceiling or final failure.

Current completion state is listed below; no partial-scene means are published. Exact existing asset paths, score/protocol/token seals and B history are in `results.json`. Source actor and canonical scoring were not changed for this readout.


## Latest completed requested checkpoints

| Architecture | Updates | Full Navtest PDMS |
|---|---:|---:|
| A | 13000 | 80.007834 |
| B | 10000 | 80.692819 |
| C | 12000 | 85.386918 |
| D | 7000 | 80.862939 |

Pending: none. Unequal intermediate budgets do not establish a final architecture ranking.
