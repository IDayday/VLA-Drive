# Prefix-Aware Soft Contrastive Label Audit

- Prefix-only construction: **PASS** (future-after-horizon used: `False`)
- All probability rows sum to one: **True**
- Same-prefix/different-tail examples: **62**, behavior pass rate **100.000%**

| Horizon | Mean GT weight | Effective positives | One-hot false negatives |
|---:|---:|---:|---:|
| 0.5 s | 0.1346 | 9.651 | 7.047 |
| 1.0 s | 0.1350 | 9.853 | 6.906 |
| 2.0 s | 0.1414 | 10.097 | 6.641 |
| 4.0 s | 0.1459 | 10.230 | 6.625 |

Candidate-consequence `Q` combines dimension-aware `C_environment_only` distance with a mask-aware, stable-track-aligned actor distance. Binary risk fields use unit distance; continuous fields use robust physical scales. Invalid/missing actor slots never contribute as zero-valued actors.
