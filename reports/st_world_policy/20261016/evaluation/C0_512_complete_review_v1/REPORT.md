# C0 step 512: complete full-dev diagnostic

Independent Evaluation audit: **PASS_COMPLETE_SINGLE_ARM_DIAGNOSTIC**. This is the prescribed 512-update diagnostic of the formal C0 continuation. It is not a C1−C0 comparison or the common 4096-update endpoint.

All **1696 scenes from 16 logs** matched the fixed dev manifest exactly, with no missing, extra, duplicate or foreign-log rows. CSV checksum, finite official [0,1] values, all saved feedback validity flags and the complete denominator passed. All 46 valid zero scores remain included. The independently recomputed aggregate and every log summary match `score_world/COMPLETE.json`.

| Official metric | Full-population mean |
| --- | ---: |
| PDMS | 0.909893518688 |
| NC | 0.991155660377 |
| DAC | 0.981132075472 |
| TTC | 0.963443396226 |
| EP | 0.848622435333 |
| Comfort | 1.000000000000 |
| DDC | 0.987028301887 |

Zero PDMS: **46/1696 (2.7123%)**, spread over 9 logs. There are 1225 scenes with PDMS ≥0.9 and 425 between the registered zero and high thresholds. Those are C0 strata only; high-score degradation and zero-score transitions require a matched treatment at the same checkpoint.

Canonical component values below `1−1e−8` occur in NC 16 scenes (14 zero, 2 half-score), DAC 32, TTC 62, DDC 28 (16 zero, 12 half-score), and Comfort 0. NC/DAC/TTC overlap: their union is 91/1696 (5.3656%). These are descriptive component flags, not mutually exclusive failure causes or a new risk score.

The preregistered stopping proxy flags **0/1696**, including **0/1590** moving-reference scenes. It requires reference progress ≥5 m, executed raw progress ≤1 m and mean executed speed ≤0.5 m/s. Mean executed progress is 18.558747 m; fixed-reference progress 20.572293 m; executed speed 4.551061 m/s. Canonical EP is zero for 46 scenes; this must not be interpreted as 46 physical stops. The absence of stopping-proxy flags does not establish general safety or noninferiority.

Offline original-GT trajectory diagnostics cover the same 1696 scenes: **ADE 0.667219688 m; FDE 1.403338678 m**. These use all eight command XY waypoints in the original ego frame; they do not substitute tracker states for the prediction.

The frozen 6714 `load_population`, `load_run` and `trajectory_diagnostics` readers verified every prediction receipt/payload and scored trajectory hash, every saved feedback value and reference-v2 execution array, the original GT ego-label identities, formal registration/approval lineage and metric-context metadata. The small native COMPLETE receipt binds the exact 512 tag and file map. Large model, image and metric-cache payloads were not opened or rehashed.

Actual scoring used **1696 admitted calls of 1728**, 424 completed request groups, one invocation, zero retry groups and no failed/unresolved attempts. The audit itself made no score call and initialized no CUDA context. It took 19.640 seconds, used at most 2 CPU threads, and peaked at 1,569,472,512 bytes RSS.

Exact execution binding:

- Runtime and training source: `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`.
- Native tag: `step_000512_1792206493515965061`; training identity `484fc7d2db6ceba5fb531e4f42423471a2730df4a2afc8ce8ce711dbfc435aa9`.
- Native COMPLETE SHA256: `ed68d260feaa95bbeced6ea0bea78bfca9e8f0431aaba18d30d39f1024934a12`.
- Export identity: `5f8d968ca090532e868013f62582b597f662faf20dd4d29b6480c6e71c4c90fb`; world1, seed42, original Euler10, one trajectory, FP32 masters/TF32 off.
- Score identity: `4dc489ae0b4cdbaaf9b3d0a262ae35c3f1d51abcbdba4e8fa47d17e9959b1bad`.
- SCENES.csv SHA256: `063de1251ec3d87df0f4e406c80548d554b09d972858e949d81fb2cfb8483787`.

No matched C1 artifact was consumed. There is no treatment gain, paired confidence interval, transition table or regression estimate in this report. The registered 2000 paired-log PCG64 draws with seed20261017 remain unchanged for later comparisons. All C arms retain the fixed 512/1024 diagnostics and 4096 endpoint without scientific early elimination; Navtest remains unread and unrun.

[Machine-readable review](REVIEW.json) SHA256 `a0780231374425a6e1bb6831c22964a5c7be46e861e08d4c3b2a71a2c29a5035` includes the exact evidence pins. The private audit directory is `/mnt/project/st-world-policy-20261016/audits/evaluation/C0_512_complete_review_v1`; its `AUDIT.py` reproduces the read-only checks, while `BY_LOG.csv` and `SCENE_RISK_DIAGNOSTICS.csv` retain the complete derived denominators. No training, scoring or export command was dispatched by this audit.
