# Fixed-nine timestamp replay

All nine original v2 geometry failures pass after changing only the epoch timestamp conversion to official `TimePoint.time_s` multiplication. The preregistered `1e-9` tolerance was retained. XY and vehicle-corner maximum errors are zero in all nine; periodic yaw error is at most `4.440892098500626e-16`. The old division path reproduces corner errors from `9.064096957445145e-7` to `8.656643331050873e-6` metres.

These are the original nine failures from the fixed 32 TRAIN scenes, not a replacement population. Original failed results remain intact. The replay uses actual original metric caches and trajectories, and runs neither model inference nor official scoring. It establishes geometry compatibility only, not policy or PDMS improvement.

Execution: immutable runtime `0d0d19dd5ce2d2fd583d2cc265289e47649ca594`; plan SHA256 `a82a805375ba3d10a6ac0a6efafc8e91eb07fbc40c1d423b2a0eaecad928e43c`. Independent correctness review `51494473882730108d38bd265b4f67e46cdf7b720ef8e1e41a8c94e5d8c0673f`. Lead admission `22cd6cd437105909a6c7276759396d984dfef6d3e5402ea08ba31cb7e41d8837`. Outer timeout: TERM at115s, KILL after5s.

Measured inner execution: 1.902s; 2,846,223 logical verified-asset bytes; 843,673,600 peak RSS bytes; 0 scoring calls; CUDA not initialized. Import/startup is outside this inner timer but inside the external120s bound. Exit code0 and all nine explicit parity booleans were separately verified.

Next: complete the same fixed32 direction diagnostic using the corrected immutable source, separate output and narrow independent reviews. This does not widen tolerances, change perturbations or select scenes using outcomes.
