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

Config files:
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s0_large_safetyep.yaml` (SHA256 `199b56b6a1e1cfc3285051a5877b3b3c6a6ebfb549ff0e6fd5ed2b0fefc53364`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s0_low_lr.yaml` (SHA256 `ee8b64b72c46fc4b9309fced59616db523178d5f28910a8a660a49709284c162`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s1_nominal_rescue.yaml` (SHA256 `1d7b54771eb36e917ac29a48ea3ba57b779ddbb38845bf017e4cd63615c9dfd5`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s2_conservative_awfm.yaml` (SHA256 `df79fa7fdd47afde90f871ed869dee851b0b51913fbd26db136457784623365e`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s3_large_fpo.yaml` (SHA256 `c47a027ae6c6d5a96945a86c3dfea93ef51ddf57d6e9910b1e6d4bf43987b1a2`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s4_temporal_rescue.yaml` (SHA256 `e6b0a2281432a9c568f46618adf68ce739713dc45618a63b3639fe650d586731`)
- `/mnt/project/VLA-Drive-qwen-world-performance-rl-20261103/configs/qwen_world_performance_rl/s5_performance_max.yaml` (SHA256 `248a4491189476ba3d5947ed50ad059a69f904f076f73db6e2f2cf96f6d2a082`)
