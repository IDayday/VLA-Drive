# Cache decision and current execution

Observed 2026-10-17 09:59 UTC. This records the user's disk constraint for
the active ST world/consequence campaign. It does not restart held IQE tasks.

The chosen implementation reuses the existing immutable current-frame DINO
targets through an index. Missing future targets are computed with the same
frozen FP32 teacher and retained only in a bounded RAM LRU. No new full-population
future teacher tensor bank is being built. The student retains the verified
DriveDreamer-Policy execution recipe: BF16 execution, FP32 master weights and
Adam state, and ZeRO2. The campaign's already registered objectives and training
timelines are unchanged.

Disk allocation is limited to metadata, indexes and control records. Their
combined reservation is 5,620 MiB within an 8 GiB ceiling, including prior
reservations and temporary space. This is an upper-bound reservation, not the
current size of completed artifacts. Existing caches and checkpoints are not
deleted. Checkpoint space and the free-space reserve are accounted separately.

## Completed development metadata

The full development population contains 1,696 scenes from 16 logs. Its
metadata build completed in 161.435 seconds, with 921,694,208 bytes observed
peak aggregate RSS. Final root allocation, including the retained failed
attempt and successful result, is 30,265,344 bytes. Teacher extractions and new
teacher tensor bytes are both zero.

- COMPLETE SHA256: `5418ed85260c449bee5df626b14e123064b883488816c6103bf4904a4e365180`.
- Request identity: `61f2c9400dec518e33b60e222d14d6c096826b61dc865ee64638c60c55afa149`.
- Successful result SHA256: `889a108501ddac3b269ecb5274406bb4d01a69bdd12d6e4a3b51d96321fe013f`.

The original failed attempt remains recorded. The corrected controller avoids
writing transient counter files while the builder inventories its directory.
Independent tests and the actual retry closure are in
`full_build_admission_v1/dev_actual_complete_v1/` and the referenced reviews.

## Training metadata in progress

The original 101,592 training scenes / 1,176 logs remain fixed. The single
approved build started at Unix time 1792230754.7872064. Its START record is
`/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_v1/CONTROL_BUILD_STARTED.json`.

- Controller SHA256: `a866ae4261cad36312355028cdfba3b24f52f64b174433635d0473420ddc84bf`.
- Admission SHA256: `e7cd20b754b4f8fab6b3045d1ab7f70e79b1e4299966ce80a0096761e5cc7389`.
- Limits: two CPU cores, 4 GiB aggregate RSS, four hours plus bounded cleanup,
  512 GiB charged logical reads, 2 GiB output-root allowance, no GPU and no
  teacher extraction.

Independent first observation at 09:58:55 UTC found 17 complete log shards
covering 3,470 scenes, 33,931,264 allocated bytes and 1,021,157,376 bytes of
aggregate RSS. The owned child was reading the next log's source images.
These are partial progress figures, not a COMPLETE receipt or a guaranteed ETA.
The remaining index work will use actual completion identities.

The preceding 32-scene equality checks and 64-update PROFILE support the
small bounded path only. Full-population supply rate, multi-rank behavior and
the formal temporal training admission still require their scoped checks.
No multi-terabyte cache, training restart, data reduction, precision change,
formal ACTIVE update or push was performed for this decision.
