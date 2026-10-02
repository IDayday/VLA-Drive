# Frozen C1@100k video future probes

Source41c8b8195fd12706a3c6ab9c71d225e6a62d260e. Fixed4096 training scenes, complete1696 development scenes/16 logs, no log overlap. Each lightweight head completes2000 updates, batch16,32000 training presentations, same seed42 and initialization/data rule. The old Qwen and execution action model remain frozen; main-model optimizer updates0. Two missing clips are retained in the1696-scene denominator and yield no auxiliary metric;1694 valid clips,0 failures.

| Input to future head | Normal normalized MSE | Permuted W | Permuted GT action | Trainable head parameters |
|---|---:|---:|---:|---:|
| V0: W, no action | 0.30244818 | 0.54299629 | 0.30244818 | 8413184 |
| V1: W + GT ego action | 0.30212075 | 0.52355874 | 0.30919475 | 8943104 |
| V_ACTION_ONLY: GT ego action, no W input | 0.37251329 | 0.37251329 | 0.47019124 | 7894016 |

All heads have the same512-wide/two-cross-attention readout structure; the explicitly enabled conditioning encoders determine the reported trainable parameter difference. Only the same V-JEPA2.1 target is compared here: three views, four real tubelet positions,9x12 spatial grid,1024 channels, channel LayerNorm and scene/view mean. Raw DINO and video errors are not put in one ranking.

V1−V0=-0.00032743 normalized MSE,16-log paired95% interval[-0.00083427,+0.00001517],10000 bootstrap draws/seed73. The improvement is small and the interval includes0. This fixed-budget probe does not establish a reliable benefit from adding GT ego actions to W.

V1−action-only=-0.07039254,95%[-0.07432064,-0.06389716]. W remains useful beyond the action condition in this probe. The action-only control was independently trained with no W input, not obtained by deleting W from a previously trained W+action head. W/action permutations are sensitivity interventions and can be out of distribution.

Training-only mean and same-video-encoder current-repeat references are being evaluated separately. They are artificial diagnostic references and never replace missing future clips or enter formal model inputs. The frame-sequence F0/F1/action-only comparison is queued behind the complete matching native DINO cache. These results are frozen-representation probes, not new-method planning PDMS or cross-training-seed stability.

Formal S0–S4 continue from common public Qwen/random driving modules; the frozen100k weights and these lightweight heads do not initialize formal runs. Actual matched query rows, losses and model weights stay in /mnt/project/action-video-foresight-artifacts/20261002/frozen_C1_100k_future_*_v1. Compact identities, target structure, all requested/valid/failure counts and paired intervals are in FROZEN_VIDEO_PROBE_RESULTS_V1.json.
