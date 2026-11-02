# Executable performance configurations

All algorithms independently start from the same complete IL70 and fresh Adam/RNG42. No qualification weights are formal initialization. Qwen is frozen; actor-attention LoRA and scene/predictive memory are trainable. Original W3 mean anchor β50 remains; it is neither token KL nor an extra BC loss.

| ID | algorithm | starting B / G / inner M | action / memory peak LR | PG matching MC | planned rescue/history |
|---|---|---|---|---|---|
| S0_LARGE_SAFETYEP | safetyep | 128 / 16 / 4 | 5.0e-05 / 5.0e-06 | N/A | main@epoch0.0:B128, rescue=False, history=False |
| S0_LOW_LR | safetyep | 128 / 16 / 4 | 3.0e-05 / 3.0e-06 | N/A | main@epoch0.0:B128, rescue=False, history=False |
| S1_NOMINAL_RESCUE | nominal_rescue | 128 / 16 / 4 | 5.0e-05 / 5.0e-06 | N/A | main@epoch0.0:B128, rescue=True, history=False |
| S2_CONSERVATIVE_AWFM | awfm | 128 / 16 / 2 | 3.0e-05 / 3.0e-06 | 4 | main@epoch0.0:B128, rescue=False, history=False |
| S3_LARGE_FPO | fpo | 128 / 16 / 2 | 3.0e-05 / 3.0e-06 | 2 | main@epoch0.0:B128, rescue=False, history=False |
| S4_TEMPORAL_RESCUE | temporal_rescue | 128 / 16 / 4 | 5.0e-05 / 5.0e-06 | N/A | main@epoch0.0:B128, rescue=True, history=False |
| S5_PERFORMANCE_MAX | performance_max | 64 / 16 / 4 | 5.0e-05 / 5.0e-06 | N/A | coverage@epoch0.0:B64, rescue=False, history=False; absorb@epoch1.0:B128, rescue=True, history=False; retain@epoch4.0:B128, rescue=True, history=True; refine@epoch8.0:B128, rescue=True, history=True |

M is optimizer reuse; it is not MC. S2 uses MC4/inner2 with conservative signed velocity-MSE and rollout regularization. S3 uses MC2/inner2 with per-MC CFM-energy PPOclip.01; no endpoint-density/full-FPO++ claim. Nominal-rescue CFM uses separate auxiliary MC2/sigma[.1,.9]; it is not source PG matching MC. S4 uses the explicit weighted indicator+actual-time cost in ALGORITHM_DESIGNS.md, with strict feasible precedence. The inherited source clip.005 is not S3’s replacement PG clip.

Schedule clock is valid scene exposures/N: warmup .08 epoch, hold to .75, cosine to4 epochs then .05 floor; stage LR bases may differ. Unlike the older I WKC schedule, this is an exposure clock, not an assumed fixed Adam clock. Target6/minimum3 epochs and later8/12/20 are configuration plans, subject to actual progress, explicit review and host authorization.

Evaluation epochs: .25/.5/1/2/3/4/6/8/12/20; canonical single-B1 ODE, K10/shift5/seed2101, full NAVTEST12146/invalid0. Partial, qualification, TRAIN and oracle outputs are not formal deployment scores. `analyze.py` leaves unobserved gains/zero repairs/long-training holds null.

## Independent S6 verified-quality route

S6 uses the same complete IL70 with fresh Adam0, B64/G16/four inner updates,
peak action/memory5e-5/5e-6 and the same mature exposure schedule. Actual B1 is
queried from epoch0. A teacher must pass NC/DAC/TTC/Comfort=1 and score≥.9. If
B1 is not strict-high, gain≥.02 uses alpha.05; an already strict-high B1 may
also absorb gain≥.005 with alpha.02. It replaces the original beta50 share by
.75 calibrated CFM + .25 actual ODE-last supervision. It does not use Control
trajectories or NAVTEST labels. First evaluation epochs are .0625/.125,
followed by the original schedule; actual launch/update status is read from
the live registry and journals, not this design document.

Frozen source `runtime/verified_quality_v2@8067e13` passed actual full-IL70
world8 continuous4/split2/resume4 qualification. All weights, Adam, scheduler,
calibration PG EMA, eight-rank RNG/stream and per-step loss/gradnorm/LR matched
bitwise. The first64 TRAIN scenes naturally activated20 alpha.02 and12
alpha.05 teachers. Qualification weights never initialize formal S6.

The v1 probe exposed a cold-start weakness in IL-only gradient calibration.
V2 uses `max(norm(gIL), .04*EMA(norm(gPG)))`, decay.9 and first-positive-PG
initialization. Initial PG/EMA-zero slots retain minimum scales without
consuming the100/500 eligible calibration/measurement budget; later PG0 slots
can use historical EMA. The existing S1 V1 path remains unchanged. Actual
four-step last-slot targets were387–22870 times the local IL gradient norm.
These component norms establish a material change in injection scale; weighted
products and triangle estimates are not directly measured joint BF16/DDP
gradient norms or evidence of PDMS improvement.

Evidence: `P/qualification/S6_VERIFIED_QUALITY_NATIVE_V2/COMPLETE.json` and
`P/operations/S6_NATIVE_QUALIFICATION_LANE_V2/GRADIENT_READOUT.json`, where
`P=/mnt/project/qwen-world-performance-rl-20261103`. S4 and LOWLR preserve
their native states while resources are assigned to S6; this disposition does
not declare either algorithm ineffective.

## TRAIN GT discovery is separate from S6 and NAVTEST

The complete GT audit used2048 existing NAVTRAIN diagnostic scenes. Thirteen
whole trajectories had timestamp gaps and were not scored or treated as zero;
2035 original physical-FP32 GT candidates received new CPU canonical scoring,
with zero model forwards and zero training updates. All15 B1-zero cases had a
strict4 GT;11 also reached score≥.9. Across2035 legal scenes,245 GT candidates
were strict-high with gain>.005, including67 with gain≥.02. However255 GT
trajectories reduced score by>.005; among1606 already-strict-high B1 cases,
197 improved and170 worsened by>.005. One GT failed Comfort.

This is evidence for verification/gain gates, not a new policy score. GT is
privileged offline supervision, has no on-policy probability/advantage, and
was not admitted to S6, another bank or an active algorithm. A good GT does
not establish that same-version G16 lacks a successful candidate. Codec
round-tripping changes FP32 requested bits, so any future normalized GT
teacher needs fresh endpoint scoring. The existing input pool includes40
old DEV scenes; NAVTEST tokens/physical logs remain disjoint. Compact source:
`P/analysis/VERIFIED_TRAIN_GT_TEACHER_AUDIT/RESEARCH_READOUT.json`.

The completed real TRAIN19 probe adds targeted sampling evidence: on15 fixed
B1-zero scenes, G16/G32 each found strict-high in12 scenes, with union13. Both
groups found a strict-high candidate on all11 scenes whose GT was strict-high.
The remaining two neither-high cases had safe GT scores.857625/.890610, below
the.9 teacher threshold. Thus this panel has no GT-only strict-high success,
while two GT trajectories can safely repair zero below that threshold. This
does not generalize to the full TRAIN population or make GT universally useless.

All38 groups had nonzero original SafetyEP raw credit; all91 sampled strict-high
candidates in the zero cohort had positive rawA. In four high controls, G16/G32
still sampled8/64 and22/128 strict3-unsafe candidates; some high candidates had
negative relative credit, while unsafe-positive credit was zero. The panel used
912 actual queries, independent G16/G32 seeds/BF16 batch shapes and zero Adam
updates, so it diagnoses current sampled-support/B1 gaps, not G32 causal gains
or post-training absorption failure. It does not alter S6 or create a new bank.
Source: `P/analysis/IL70_TRAIN19_G16_G32_EXPLORATION/RESEARCH_REVIEW.json`.

## S6 first-window progress-teacher discovery

The complete75-query CPU roundtrip-GT audit found that all seven S6 first-window
cases lacking both main/extra strict-high witnesses already had strict4 B1 and
strict4 GT trajectories. Their GT scores were.802890–.876222: safe but too low
in progress for the.9 teacher gate, not seven safety failures. GT supplied no
new eligible disabled slot. This remains separate from the earlier TRAIN2048
audit and its different sampled population.

A separately completed112-query CPU diagnostic constructed explicit smooth
progress proposals along the current B1 and offline TRAIN GT paths. The original
same-seven extra-G16 pools had0/112 strict-high candidates; the synthetic panel
had35/112 strict4 candidates with score≥.9 and B1 gain≥.02, covering7/7 scenes.
B1-path proposals contributed16/56 eligible candidates across5 scenes; GT-path
proposals19/56 across6 scenes, with overlap. Thirty-two synthetic candidates
failed at least one strict condition: construction alone never certifies safety.

These are privileged TRAIN teacher-discovery results, with no new GPU forward,
Adam update or post-update B1 evaluation. Generation/cost differs from the SDE
pool; this is not a fair observation-only policy ablation, deployment score or
proof of learned improvement. S7 is a separate development route, not activated
by this diagnostic; no teacher from it is silently inserted into S6. Sources:
`P/analysis/S6_FIRST_WINDOW_GT_FALLBACK_AUDIT/RESEARCH_READOUT.json` and
`P/analysis/S6_FIRST_WINDOW_SYNTHETIC_PROGRESS_AUDIT/SUMMARY.json`.

Config files:
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s0_large_safetyep.yaml` (SHA256 `199b56b6a1e1cfc3285051a5877b3b3c6a6ebfb549ff0e6fd5ed2b0fefc53364`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s0_low_lr.yaml` (SHA256 `ee8b64b72c46fc4b9309fced59616db523178d5f28910a8a660a49709284c162`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s1_nominal_rescue.yaml` (SHA256 `1d7b54771eb36e917ac29a48ea3ba57b779ddbb38845bf017e4cd63615c9dfd5`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s2_conservative_awfm.yaml` (SHA256 `df79fa7fdd47afde90f871ed869dee851b0b51913fbd26db136457784623365e`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s3_large_fpo.yaml` (SHA256 `c47a027ae6c6d5a96945a86c3dfea93ef51ddf57d6e9910b1e6d4bf43987b1a2`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s4_temporal_rescue.yaml` (SHA256 `e6b0a2281432a9c568f46618adf68ce739713dc45618a63b3639fe650d586731`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s5_performance_max.yaml` (SHA256 `248a4491189476ba3d5947ed50ad059a69f904f076f73db6e2f2cf96f6d2a082`)
