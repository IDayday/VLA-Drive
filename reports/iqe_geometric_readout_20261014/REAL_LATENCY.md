# Actual image-to-trajectory cost

NVIDIA A800-SXM4-80GB, batch1, original three-view image input and locked Base
runtime; no precision change. Each mode has3 warmups and32 measured iterations.
Image load/decode/processor and output CPU copy are included. The file cache was
prevalidated/warm in all modes. This is not a cached-feature inference figure.

| Mode | Full mean ms | p50 ms | p95 ms | Scorer mean ms | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| Base, K1 | 737.24 | 735.33 | 760.93 | — | 8.44 |
| Base + three experts, K4 | 794.36 | 794.36 | 815.44 | — | 8.62 |
| K4 + global spatial reference | 791.06 | 790.39 | 821.98 | 7.54 | 8.63 |
| K4 + geometric Scorer | 791.21 | 791.00 | 817.01 | 11.28 | 8.63 |

Geometric Scorer-only p50/p95:11.04/13.10ms. It costs about3.74ms more than the
matched global spatial reader in these event measurements. Differences of a few
milliseconds in whole-pipeline windows are timing variability: the table does
not mean adding a Scorer accelerates the four-candidate system or is free.

Every actual forward had one visual call and one language call. K4 used one Base
Decoder and three independent private Decoder calls; each scoring mode had one
semantic context projection and one visual reader call. No training or official
score computation occurred in this benchmark.

Base parameters:2,147,752,352. Each private-reader expert:15,708,576.
Both global and geometric scoring heads:4,898,311 parameters. The old readout
reference was constructed from the same random initialization for cost only;
no old/external Scorer weights were loaded. The geometric head used this trial's
complete step2 checkpoint. These timings test architecture cost, not the quality
of that early model or the eventual selected6000-step endpoint.

Full numeric records, checkpoint/source hashes, runtime and commands are in
`REAL_LATENCY.json` and `REAL_LATENCY_PROVENANCE.json`. Full windows synchronize
CUDA before and after each sample; Scorer timing uses CUDA events within the
same forward. Parallel training on other GPUs is disclosed by the launch's
shared-host scope; no other workload was stopped.
