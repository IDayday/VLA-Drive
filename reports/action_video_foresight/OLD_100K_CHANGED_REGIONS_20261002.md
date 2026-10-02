# Frozen100k changed-region diagnosis

C1 covers all1696 development scenes/16logs; C0 is the previously fixed128-scene/16-log limited cross-check. Both completed with0failures and0optimizer updates on source2df13419467a2275aed2376aef1db7fab3f16bda. FP32 masters/readouts, TF32off. Thresholds are per-view/horizon upper training quartiles of mean-channel squared current/future feature change, fitted only on the fixed4096 training queries. They include camera and appearance change and are not semantic moving-object labels.

| C1 region | Horizon | Model MSE | Copy actual current MSE | Copy predicted current MSE | Scene-shuffled W MSE |
|---|---:|---:|---:|---:|---:|
| changed_upper_train_quartile | 1s | 0.02000401 | 0.04492898 | 0.03195645 | 0.05625058 |
| changed_upper_train_quartile | 2s | 0.02231892 | 0.05830882 | 0.04253090 | 0.05696595 |
| changed_upper_train_quartile | 4s | 0.02505548 | 0.06902115 | 0.05242925 | 0.05775994 |
| other_change | 1s | 0.01254970 | 0.01181658 | 0.01347667 | 0.05010196 |
| other_change | 2s | 0.01353410 | 0.01768868 | 0.01582592 | 0.04916047 |
| other_change | 4s | 0.01528984 | 0.02488542 | 0.01986378 | 0.04766362 |

At greater feature-change positions, the old C1 future prediction beats copying current features and shuffledW at all three horizons. In the remaining lower-change positions at1s, it is slightly WORSE than copying actual current features (.01254970 versus.01181658); this limitation is retained. Thus auxiliary learning exceeds a static reference in important change regions, but is not uniformly better at every position/time.

C0 also improves over its own static-copy reference in the fixed limited cross-check. Do not rank raw C0 and C1 MSE: their teacher resolution/pooling and evaluation populations differ. This is within-target diagnostic evidence only. It does not prove new PDMS gains, precise object motion reconstruction or counterfactual dynamics.

Per-view denominators and compact aggregate source/checkpoint/cache identities are in `OLD_100K_CHANGED_REGIONS_20261002.json`. Full per-scene rows remain in the independent artifact directory; no raw images or teacher weights are uploaded.
