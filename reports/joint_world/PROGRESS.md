# Initial joint graph evidence

Implemented: fixed ego+64 predicted actors, whole-actor random-mask continuous flow, relative-geometry interaction attention, current Qwen context, strict prediction-only sampler and zero-gated bridge to original DiT. GT future context is confined to auxiliary training; at least50% of scenes hide all futures. No RL, scorer, PDMS training or future image inputs.

Real current-only image caches:128 scenes,0 failures. Two matched graph-only pilots:64 scenes,1000 updates each, batch8,8000 presentations/125 passes, seed42. Frozen Qwen, DiT, vision and previous world Reader/heads. No original policy update. Train source dfe8122; intermediate500 and final1000 preserved. Frozen world pretraining previously used these64 training scenes; original base-model dev exposure is inherited and not claimed unseen.

Randommask vs all-hidden control, final1000:

| Split | Ego ADE, m | Ego FDE, m | Agent ADE, m | Stationary ADE, m | Motion coverage |
|---|---|---|---|---|---|
| train randommask |2.712|3.513|3.223|2.456|49.57%|
| train control |3.313|3.679|3.815|2.456|49.57%|
| log-holdout randommask |3.270|4.443|4.904|2.437|11.46%|
| log-holdout control |3.673|4.495|5.397|2.437|11.46%|

All evaluations:64/64 scene successes. Matched surrounding errors use independent geometric matching on current predictions; ALL GT objects appear in object CSV, misses carry absent ADE/FDE. Holdout follows the previously selected disjoint-log64 manifest, never adapted. End-to-end motion coverage remains poor because the inherited detector generalizes poorly. Randommask holdout dynamic ADE6.442m vs stationary6.209m; static4.428m vs1.377m. These results do not establish interaction understanding or planning benefit. Graph ego error is not original DiT planning performance. Paired uncertainty, additional seeds and context-use diagnostics are pending.

Real integration: one actual scene, full Qwen/image/DiT path. Gate0 action output exactly equals original, poisoning extraneous target/future fields changes neither graph nor action, saved modules restored into new instances exactly. Two optimizer updates show nonzero ego gate gradient at step1 and nonzero ego-to-graph gradient at step2. Magnitude of graph gradient4.77e-10 is small and does not prove useful planning optimization. Original model stays frozen; peak12.19GB allocated. No convergence claim from the update test.

Failures retained: first integration failed BF16 input/FP32 linear boundary; second and third found identical conditions but2.00e-5 action drift caused by changed initial-noise dtype. Corrected by explicitly preserving originalBF16 ego noise while using FP32 residual conditions under the originalFP32 autocast. No tolerance relaxation. Old action sampler default path is preserved; optional initial_noise is used only by the new bridge. Head-entry RNG hash differs because explicit ego noise is drawn immediately before entry rather than inside; final outputs are exact.

Remaining: explain static drift and low detector coverage, bounded targeted repair if warranted, conditional-context usage tests, BEV occupancy/motion/interaction supervision and real BEV integration, matched bridge planning training, original1696 paired evaluation, visualizations and checked resume. Full goal ACTIVE; engineering PARTIAL, interaction INCONCLUSIVE, planning NOT_RUN, new BEV objectives NOT_IMPLEMENTED.
