# Actual fixed-four no-update calibration launch

The independently reviewed loss-gradient diagnostic was actually launched on one freshly idle local A800 (GPU0), CPUs2/3. The immutable runtime is `09ca42cb`; current and future targets reuse the existing real64 bank. No optimizer is created, no checkpoint is saved and no new teacher tensors are written. Parameters/buffers and RNG are checked by the diagnostic.

The one invocation is capped at 3600 seconds plus 40 seconds cleanup,96GiB aggregate RSS,256GiB logical reads and16MiB output. The snapshot and exact control/source/review/process identities are in `LAUNCH.json`. This is a launch record, not a measured gradient, throughput or scientific-weight result. Actual terminal status must be read from the isolated output; no automatic retry or formal configuration update is authorised.

Existing C training and the original TRAIN future-metadata writer continue unchanged. The latter stores file/time metadata, not a multi-TB teacher tensor cache.
