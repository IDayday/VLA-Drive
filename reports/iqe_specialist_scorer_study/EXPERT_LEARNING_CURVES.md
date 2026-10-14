# Specialist training evidence

| Cell | Endpoint | H_train scenes | Base target IL | Expert target IL | H_train PDMS delta | H_val count | H_val delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E00 | raw | 125 | 0.0561041 | 0.0099076 | 0.729751 | 1 | -0.00157691 |
| E00 | ema | 125 | 0.0561041 | 0.00657447 | 0.737387 | 1 | -0.000872189 |
| E01 | raw | 125 | 0.0561041 | 0.0101283 | 0.705671 | 1 | -0.00339988 |
| E01 | ema | 125 | 0.0561041 | 0.00539533 | 0.736917 | 1 | -0.00214885 |
| E10 | raw | 125 | 0.349038 | 0.0178544 | 0.791076 | 1 | 0.0136796 |
| E10 | ema | 125 | 0.349038 | 0.0165815 | 0.786191 | 1 | 0.0135703 |
| E11 | raw | 125 | 0.349038 | 0.0105624 | 0.79403 | 1 | 0.00963681 |
| E11 | ema | 125 | 0.349038 | 0.00578033 | 0.794176 | 1 | 0.0105214 |
| EXT_interaction | raw | 116 | 0.194031 | 0.00884751 | 0.432722 | 9 | 0.137626 |
| EXT_interaction | ema | 116 | 0.194031 | 0.00478397 | 0.436318 | 9 | 0.137539 |
| EXT_safe_progress | raw | 393 | 0.341941 | 0.00867083 | 0.112979 | 11 | 0.0831346 |
| EXT_safe_progress | ema | 393 | 0.341941 | 0.00725586 | 0.113269 | 11 | 0.0838832 |

Pure specialty supervision, no anchor/distillation, 5,000 updates, global batch128, unchanged Query IL/dropout. One update counts128 actual sample presentations; repeated exposure is not new observations. Core cells use identical sample sequence. Private-reader cells update only their own copied compression/query/ego/action branch. The shared VLA is absent from head-only training.

Already selected pool endpoints, independently rescored on their entire fixed H_train (no further updates or selection):

| Expert | Step/endpoint | Scenes | Logs | Base PDMS | Expert PDMS | Direct gain | Base IL | Expert IL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| road_direction | 3000/ema | 125 | 89 | 0.182356 | 0.952608 | 0.770253 | 0.349038 | 0.0193226 |
| interaction | 3000/raw | 116 | 90 | 0.556183 | 0.996114 | 0.439931 | 0.194031 | 0.0181648 |
| safe_progress | 1000/raw | 393 | 231 | 0.863756 | 0.974038 | 0.110282 | 0.341941 | 0.0422105 |

These selected-endpoint diagnostics are distinct from the fixed5000 raw/EMA factorial rows above. They measure training-support fit, not unseen-scene generalization. Source files and checksums are in SELECTED_EXPERT_H_TRAIN_EVIDENCE.json. The diagnostic performs no optimizer updates, does not modify the pool, and reads no test data.

Generated 2026-10-14T17:45:14.300013+00:00. Implementation `1e40377d247a106ccd8cd94e33c9c1a4f7a9e853`. Base step 70000, SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`. All current scores use NAVSIM v1.1 fixed official reference; internal score unit [0,1]. Artifact root: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
