# Complete-method development milestones, 2026-09-29

Two formal eight-GPU runs remain active on the fixed100000-update plan. This report uses all1696 development scenes/16logs with train/log isolation, training seed42 and fixed inference seed42. It does not report Navtest or a completed configuration choice.

| Updates | Configuration | PDMS points | Ego ADE m | Ego FDE m | Zero-score fraction | Failures |
|---:|---|---:|---:|---:|---:|---:|
| 5000 | C0 | 78.39 | 1.802 | 3.455 | 12.79% | 0 |
| 5000 | C1 | 79.89 | 1.736 | 3.322 | 11.38% | 0 |
| 10000 | C0 | 84.20 | 1.147 | 2.172 | 9.20% | 0 |
| 10000 | C1 | 84.45 | 1.061 | 2.120 | 8.90% | 0 |

Both configurations use all four losses. C0 has128×96 DINO input without pooling; C1 has256×192 DINO input with2×2 feature pooling. Both have144 W queries and the same supervision grid/student architecture, generic/random initialization, data order, batch32, optimizer/schedule and shared frozen MAE teacher. The input resolution difference belongs to the offline teacher targets, not Qwen input. Neither is an Action-Only ablation.

PDMS uses FP32 master-weight export with auxiliary heads stripped, W retained,10FMsteps and one executed trajectory. Official full traffic/route environment is retained. Reference-progress and full-precision map behavior follow the earlier audited NAVSIMv1 protocol. No candidate scorer or oracle is used.

Runtime verification discovered local NumPy1.26.4/SciPy1.13.1/Shapely2.0.7 versus remote NumPy1.26.4/SciPy1.11.4/Shapely2.1.2. Original score tables remain unchanged. C1 was rescored on the same audited local environment as C0, using identical immutable predictions and metric caches. Both1696-scene reruns had0failed scenes and exactly0maximum difference in every PDM submetric. This is observed parity for these two artifacts, not proof that every future trajectory is insensitive to library versions.

New CPU-only observer source68c1bb7 performs future paired reports under one exact evaluator identity, without modifying or stopping the immutable trainers/controllers. It reuses compatible completed scores and rescores incompatible runtimes separately. Nine focused tests passed; changed predictions/caches/logs/failures cannot be described as runtime-only parity. The observer is live and bounded to216000seconds; its launch and resume arguments are preserved in CANONICAL_OBSERVER.json. All extra CPU scoring is ledgered with gpu_count0.

At5000updates, C1−C0=+1.5044PDMSpoints, log-cluster95% interval[+0.2240,+2.2899]. At10000updates, C1−C0=+0.2491points, interval[−0.9747,+1.7150]. The latest interval includes both gains and losses. We cannot distinguish the configurations reliably at this stage. This is one training seed, with only16log clusters; no cross-training-seed stability is established. The early milestone is not selected in place of the common endpoint.

Both models improved from5000to10000 in PDMS and ego ADE/FDE. This is evidence of learning under the full protocol, not evidence of improvement over a separately trained Action-Only baseline. W_ACTION_ONLY/native and task ablations are still deferred under the user's fastest-main-result priority. Future-feature copy-current diagnostics remain NOT_RUN for these formal checkpoints; auxiliary loss decrease alone does not prove learned dynamic or interaction knowledge.

Full future/interaction weights have been active since update1000. Recorded FP32 master gradients and actual parameter changes at1000/2000/5000/10000 show nonzero W, sampled first/last Qwen attention projections and ego action-decoder updates. Independent original per-loss gradient tests remain in the earlier source2bde06f evidence; the live observations measure combined-loss training, not per-loss causal attribution.

The latest training progress, exposure, resource charges, learning curves and timing estimates are in TRAINING_SNAPSHOT.json and live external ledgers. Both models continue to25000 and then the remaining common milestones through100000. At this snapshot, approximately45hours of optimizer/save time remained per model, plus remaining milestone exports/scoring. Runs are parallel, so their wall times are not added. ETA is conditional on current throughput and no failures. Training source remains d1d40854299b9599b2accc382bcfc4b676dd7623; original controller remains fb474a7. This report/evaluation observer has a separate source identity.

Scene-level and log-level paired CSVs are included. Raw images, feature caches, model checkpoints and private scene visualizations are not committed. Planning conclusions, complete Navtest, final five-sampling-run analysis, remaining C2-C5, second training seeds and task ablations remain unfinished.
