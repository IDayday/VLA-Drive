# Corrected camera-learning preflight (not the full experiment)

The original803M-parameter DiT and generic Qwen/Wan/PPD framework are running actual camera training. No historical driving checkpoint initializes these runs. Formal A/B/C42 and B/C43 are registered for100000updates each; the formal campaign has not started in this snapshot.

The initial two-GPU fits were invalid: a DeepSpeed int32 tensor-size overflow made the optimizer a no-op. Their logs/costs remain archived with the validity overlay. Fresh corrected fits bound optimizer groups and verify changed FP32 masters at every counted step. The provisional20m head-output rescaling was not selected; head scale remains1.

Fixed64 training-scene FP32 inference (sampling seed42, one sample, zero export failures):

| Arm | Updates | Ego ADE m | Ego FDE m | Current vehicles matched within2m | Joint-selected matched vehicles |
|---|---:|---:|---:|---:|---:|
| A |64|7.3170|11.8966|N/A|N/A|
| B |64|10.0373|13.1757|0/333|0/333|
| C |64|10.7075|12.6457|0/333|0/333|
| B |128|6.2760|10.2516|1/333|0/333|

These early errors show ego learning; vehicle detection remains inadequate at128updates. One matched vehicle is not evidence of good motion prediction. Training later has nonzero predicted-graph vehicle supervision and C role tasks. The matched512-update diagnostics continue; no conclusion about MASK or planning follows from these rows.

The actual4GPU/globalbatch32/microbatch4 startup completed4verified updates, averaging12.4s/update after the first, peak41.56GiB. Five full runs extrapolate to6889.6GPUh of training at this initial ego-only graph workload. The cap is8000GPUh including prior failures/loading/diagnostics/evaluation, with300GPUh reserved before further training. This is an extrapolation, not a completion-time commitment.

Resume evidence is mixed and retained explicitly. Continuous4 versus independent2+2 has every RNG state equal but FAILS the predeclared full-master tolerance (max4.95e-6). Repeating the final2updates from the identical parent checkpoint passes the same tolerance on100000sampled master elements (max1.18e-6), again with all RNGs equal. This narrower diagnostic does not turn the failed full comparison into a pass, and no bitwise resume guarantee is made.

A real-output audit over256camera predictions confirms final ego is decoded from the same sample's slot0 (max2.38e-7 rounding difference), and unmodeled channels remain exact zero. The independent evaluation labels reproduce all core box/track/future targets for64scenes; the training-only supervision grid is intentionally not rebuilt. One source log has a temporal gap:507/512ego label points align with the nominal timestamps, so relative ego/vehicle diagnostics apply an explicit timing mask. Existing original-framework ego labels/normalization are unchanged. Full dev evaluation labels cover1696scenes,8624ROI/FOV vehicles,0failures. These labels are never inference inputs.

Full development PDMS and Navtest model results: **NOT_RUN**. No released or historical score is substituted. The immutable full-campaign registration and machine-readable evidence are beside this file.
