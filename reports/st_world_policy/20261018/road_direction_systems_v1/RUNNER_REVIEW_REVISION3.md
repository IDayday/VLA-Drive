# Fixed32 v3: independent resource delta review

**APPROVE_ROAD_DIRECTION_EXECUTION**, resource scope only. The exact JSON review
binds registration `3bb6805165186638ac69e01c43e94655a4ee412955a761ab9844c8657366a3a8`
and frozen runtime `0d0d19dd5ce2d2fd583d2cc265289e47649ca594`. Lead admission and the
separate mathematical correctness review remain required. This reviewer neither
authored the road runner/time-grid change nor dispatched the experiment.

The delta changes the epoch timestamp computation to the official multiplication
and retains detailed fixed-size geometry errors. `Budget`, bounded LZMA decoding,
admission, bounded input loading, verified context reuse, output scan, process
supervisor, final summary and child entry have identical ASTs to the previously
approved `83679937` implementation. The confirmed owned-worker cleanup and the
original orphan-failure evidence remain preserved.

The selected 32 TRAIN scenes, candidate0, four directions, 0.005 perturbation,
numerical tolerances and metric context are unchanged. Geometry output consists
of fixed 9/41 timestamps, 41 poses and four corners per pose, not an unbounded
map serialization. A conservative metadata allowance below 128 KiB per audit,
even replicated three times per scene, stays below 12 MiB for 32 scenes; this is
a size estimate, not measured v3 output. Actual writes retain the 1-GiB budget,
pre-write accounting, parent checks and reserved terminal-summary capacity.

Budget remains one CPU worker/thread, 1,800 seconds, 8 GiB aggregate RSS/address
space, 256 MiB asset reads throttled to 32 MiB/s, 64 MiB decompression per cache,
and at most 128 charged scoring calls. Originals are reused: the code permits
only three fresh variants per scene, so at most 96 v3 calls. Together with the
12 prior v2 calls, the possible cumulative count is at most 108. There is no
automatic retry or source/context mixing. v2 and the nine-scene parity replay
remain immutable separate evidence.

Fresh policy `fed0bae5c595088a5628d2861b43302912e3d8ff2d05292329d1ce44c0b058bb`
allows the local host. The new old-SimWAM hold does not stop ST consequence work.
The v3 output did not exist at review. Fresh host metadata showed 128 CPUs,
roughly 0.9 TB available RAM and 4.1 TB free shared space. The fixed CPU-only
work does not allocate the GPU used by the concurrent Lead gradient diagnostic;
no pressure, real or unknown process was signalled.

Independent current runner fixtures: **23 passed in 8.76 s**, subprocess
10.41 s, 10.06 CPU seconds, peak child RSS 898,605,056 bytes. The initial audit
hook accidentally rejected NAVSIM's `metric_cache.pyc` source import: 22 tests
passed and one was blocked by the review harness. That record is preserved.
The corrected hook permits only frozen-runtime `.py/.pyc` sources and continues
to reject real cache, checkpoint, rollout and official-feedback payload paths.
Both attempts together used 20.88 CPU seconds; no real assets or GPU were used.

The existing nine-scene parity receipt was read as a 22,562-byte JSON only. It
reports 9/9 geometry passes, zero official scores and zero GPU use; this reviewer
did not rerun those real cache operations. This approval does not claim driving
improvement, complete/defined directions for all32, or formal training readiness.
