# C0 step 1024: complete full-dev diagnostic

Independent Evaluation audit: **PASS_COMPLETE_SINGLE_ARM_DIAGNOSTIC**. All **1696 scenes from 16 logs** match the fixed development population. There are no missing, duplicate, extra or foreign-log rows, invalid metrics, or unresolved scores. Every CSV value and its checksum matched saved valid feedback; all legal zero scores remain in the denominator. The independently recomputed aggregate and all 16 per-log summaries match the canonical COMPLETE receipt.

| Official metric | C0 at 512 | C0 at 1024 | Descriptive change |
| --- | ---: | ---: | ---: |
| PDMS | 0.909893519 | 0.910766471 | +0.000872952 |
| NC | 0.991155660 | 0.990566038 | -0.000589623 |
| DAC | 0.981132075 | 0.980542453 | -0.000589623 |
| TTC | 0.963443396 | 0.958726415 | -0.004716981 |
| EP | 0.848622435 | 0.855318067 | +0.006695632 |
| Comfort | 1.000000000 | 1.000000000 | +0.000000000 |
| DDC | 0.987028302 | 0.986438679 | -0.000589623 |

At 1024, **48/1696 scenes have zero PDMS (2.8302%)**, across 9 logs. NC, DAC and TTC values below `1−1e−8` occur in **17, 33 and 70** scenes; their overlapping union is **98/1696 (5.7783%)**. NC has 15 zero and 2 half-score values. DDC has 18 zero and 10 half-score values; Comfort remains 1 for every scene. Component counts are descriptive canonical metric flags, not a new risk score or mutually exclusive causes.

The preregistered stopping proxy remains **0/1696**, including **0/1590 moving-reference scenes**. It requires reference progress ≥5 m, raw progress ≤1 m and mean executed speed ≤0.5 m/s. Mean raw progress is 18.848768 m, fixed-reference progress 20.572293 m and executed speed 4.629107 m/s. The 48 canonical zero-EP values do not imply 48 physical stops. An absence of proxy flags does not establish general safety or noninferiority.

Original-GT, command-waypoint errors at 1024 are **ADE 0.728314768 m** and **FDE 1.421504157 m**, with the full 1696-scene denominator. Relative to 512, ADE increases 0.061095080 m and FDE increases 0.018165479 m. These use the original eight XY waypoints and ego frame, not tracker-state substitution.

The optional 512→1024 progression is descriptive within the same C0 run. Every scene/log, current input hash, metric-cache identity, canonical protocol, inference environment and fixed reference-progress value matched. Differences were calculated from joined scene rows, not aggregate reconstruction. The small PDMS increase (+0.000872952 in official 0–1 units) accompanies more zero scores, eight additional TTC-deficient scenes and larger trajectory errors; it is not an unqualified improvement.

| C0 at 512 → C0 at 1024 | Scene count |
| --- | ---: |
| Zero → zero | 42 |
| Zero → nonzero | 4 |
| Nonzero → zero | 6 |
| Nonzero → nonzero | 1644 |

The zero→nonzero conditional denominator is 46; nonzero→zero is 1650; every full-population rate uses 1696. Among 1225 scenes with 512-step PDMS ≥0.9, 35 decline by more than 1e−8, 10 decline strictly more than 0.05 and 2 become zero. Six scenes have EP declines greater than 0.05; no scene simultaneously has such an EP decline and an NC/DAC/TTC gain. These are fixed-threshold descriptive transitions across diagnostics, not a registered reward-treatment contrast.

The reused frozen 6714 readers verified formal registration and independent approval lineage, complete prediction receipts/payloads and trajectory hashes, every saved valid feedback value and reference-v2 execution array, original GT ego labels and all metric-context metadata. Native tag/file-map bindings were checked from the small COMPLETE receipt. No large model, image or metric-cache payload was opened or rehashed.

Actual scoring consumed **1696 of 1728 calls**, across 424 completed request groups in one invocation, with zero retry groups or failed/unresolved attempts. This read-only audit used at most 2 CPU threads, took 26.549 seconds and peaked at 1,569,472,512 bytes RSS; CUDA remained uninitialized.

Exact execution binding:

- Runtime/training source: `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`.
- Native tag: `step_001024_1792207671595855958`; training identity `484fc7d2db6ceba5fb531e4f42423471a2730df4a2afc8ce8ce711dbfc435aa9`.
- Native COMPLETE file SHA256: `789ce37769e1d7c0cc550ebc3cb427eb5ec06be654f6c06e1b7d5b0acdcf0df3`; canonical receipt identity: `e8d92893269eccd2dbaf2d49f57bb88e327b347092f422977502c47a2b78080e`.
- Export identity: `772649659e2628098b68775e75d6029ff5a5ce8ce005edb29f2f5f6c061149d8`; world1, seed42, original Euler10, one trajectory, FP32 masters/TF32 off.
- Score identity: `742c52f648190dba18d2dc83cc120c811daf406bf4acc21e0f308e38232d5bab`.
- SCENES.csv SHA256: `8deabaa9d8e469c10f7662636014498d9be3154a45e55d6551d7ecb8aa9b5b73`.

No C1–C4 artifact was consumed. No reward-treatment gain or best-checkpoint selection is supported. The registered **512/1024 diagnostics and 4096 common endpoint remain unchanged**, with no scientific early elimination. The registered 2000 paired-log PCG64 bootstrap with seed20261017 remains unchanged for fixed arm contrasts; no inferential interval was added to this descriptive progression. Navtest remains unread and unrun.

[REVIEW.json](REVIEW.json), SHA256 `114c0cf6717daf88ef0ed584452e46227e3daaa4fa100fab66907f0f38ecd529`, pins the full audit evidence. [DESCRIPTIVE_PROGRESSION.json](DESCRIPTIVE_PROGRESSION.json), SHA256 `7b1397e4b75c606ed4632c05b6f0eae7a326a73011056df6aa98b3c1127bd0cc`, records the exact joined-row differences and prior evidence. Private artifacts are under `/mnt/project/st-world-policy-20261016/audits/evaluation/C0_1024_complete_review_v1` (`AUDIT.py`, `AUDIT_REUSE.json`, `BY_LOG.csv`, `SCENE_RISK_DIAGNOSTICS.csv`). Only these new reports were added; no trainer, runtime, scorer or preregistration changed.
