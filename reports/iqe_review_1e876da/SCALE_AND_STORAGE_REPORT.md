# Scale, measured costs and storage

All timings below are measured boundaries, not full-run speed claims. No full incremental candidate bank was built. The later full Base navtest is a separately authorized evaluation.

| Stage | Measured qualification cost | Scope |
|---|---:|---|
| actual FeatureBundle reads for retrieval | 0.289253 s |32 scenes,512-vector dimensions |
| old reference retrieval | 0.007204 s |32 queries over32 scenes |
| fixed index build + queries | 0.001853 +0.002483 s |identical IDs and exact distances |
| FeatureBundle read / candidate generation |0.019093 /0.011517 s|2 scenes, CPU2 threads, cached Query head |
| independent official candidate scores |3.278848 /2.908826 s|2 real scenes |
| independent official GT scores |2.926872 /2.944776 s|same2 contexts |
| private expert optimizer checkpoint+checksum |0.535054 s;131,741,528 bytes|real private-head state |
| Scorer validation predictions |0.037993 s|4 scenes, CPU2 threads |
| actual Base image inference |127.521584 s|4 scenes, CPU8 threads; excludes official scoring/model construction |
| Base checkpoint read/validation boundary |30.506846 s|22.816GB complete checkpoint; total evaluator elapsed is larger |

Real workflow wall times: round1 **256.092 s**, selector-only round2 **177.744 s**, expansion round3 **323.545 s**. Per-stage timings including manifest/plan, frozen checks, scoring, training, calibration and dev/fit evaluation are saved in `evidence/round_*_costs.json`. Round1 score bank reused88 labels; round3 reused88 and freshly scored44, **78.731645 s with2 CPU workers**. Do not divide reused-label counts by elapsed time to advertise fresh-label throughput.

A purely illustrative projection for100k scenes, K=2 at mean3.094s per fresh independent reference call is **171.9 CPU-worker hours** for candidates, plus81.6 worker hours for100k GT scores, before feature generation/validation/I/O. Two workers would ideally halve wall time, which is not a measured guarantee. Full frozen feature extraction would dominate on CPU; cached head timing cannot be called image-to-trajectory latency. Retrieval stays exact O(HND) with bounded query/library chunks; the32-scene timing does not establish full-library memory-bandwidth scaling.

Navtest's persistent import backend reuses Python imports only. Each candidate still reloads its own cache and creates a fresh official simulator/scorer, with the same paired seed. It is enabled only after4 independent subprocess scores and9 persistent reordered/repeated calls match exactly. This removes interpreter/import overhead without joint-candidate normalization, approximation or reduced data. The full12146-scene evaluation has its own measured export/score times and protocol in NAVTEST_REPORT.

Checkpoint capacity: one actual Base checkpoint **22,816,333,071 bytes**, save every1000,100000-step plan. At13000,87 additional saves≈**1.985 TB**, one atomic-write checkpoint plus500GiB safety reserve gives **2.544708 TB** required. The capacity snapshot had4.426788TB available; the later live shared-volume snapshot had4.325742TB. The17:50 snapshot had4.290821TB available. This is sufficient for this job's projected writes at those snapshots; it does not reserve space against other jobs. Current resume, best, scientific milestones and all existing files remain protected. No history cleanup was performed. `capacity-plan` emits a dry-run and new training fails on insufficient capacity; running legacy Base was not hot-modified.

[Cost evidence](evidence/COST_PROBE.json), [retrieval equality](evidence/SCALE_PROBE.json), [capacity](evidence/CAPACITY_PLAN.json). Default deterministic metric repetitions is1 and higher settings are explicitly rejected. Bootstrap repetitions/seed and minimum-log gates are consumed; nondefault settings and replay masking have behavior tests.
