# Frozen EpisodeDrive Candidate Selection

## Gate C3: NOT RUN

| Selector / upper bound | Mean offline official score |
|---|---:|
| Random retained proposal | 0.8270 |
| Original frozen EpisodeDrive scorer | 0.9626 |
| O5 logged-future direct-risk ranker | 0.9337 |
| O8 logged-future oracle ranker | 0.9210 |
| Best of retained K=16 | 0.9842 |

- Scenes/logs: 2,378/1,192
- Original-scorer best-of-K headroom: 0.0216
- O8 oracle ranker beats original scorer: False
- Predicted shared/direct model: NOT RUN

Official scores above are offline evaluation labels. No inference selector called
the official scorer, and no deployable planning-gain claim is made.
