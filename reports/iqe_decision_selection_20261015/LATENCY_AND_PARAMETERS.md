# Real-image inference cost

| Mode | Samples | Mean ms | p50 ms | p95 ms | Head mean ms | Peak allocated GiB | Decision params | Fallbacks |
|---|---|---|---|---|---|---|---|---|
| Base | 100 | 891.74 | 897.77 | 967.32 | 0.00 | 8.441 | 0 | 0 |
| K4_R00 | 100 | 910.66 | 916.82 | 966.89 | 5.35 | 8.627 | 3301639 | 0 |
| M_scene | 100 | 876.98 | 887.98 | 933.27 | 6.86 | 8.629 | 3299588 | 0 |
| K4_M_set | 100 | 918.74 | 920.29 | 972.44 | 7.38 | 8.630 | 4090881 | 0 |
| M_scene_margin | 100 | 891.31 | 893.43 | 934.18 | 6.97 | 8.629 | 3299588 | 0 |
| K4_M_set_margin | 100 | 915.40 | 919.30 | 964.57 | 7.51 | 8.630 | 4090881 | 0 |

One A800 80GB; batch1; 10 warmups and the same 100 real stage_val images per mode. Includes observation verification/read, image decode/preprocessing, visual and language forward, action branches, selection and CPU trajectory copy. CUDA synchronized wall times and separate CUDA events for the head are retained. No feature-cache latency is called end-to-end. Model parameter counts and every call are in evidence/BENCHMARK.json. All modes retain original Base precision and frozen generator weights. Allocated and reserved peaks are both recorded. Modes run sequentially on the same GPU. Small timing differences are not a controlled claim of a speedup: storage/page-cache state, GPU clocks and shared-host load can vary. The shared VLM dominates the end-to-end path. Margin fallback to Base does not constitute learned selection success.

Parameter inventory (actual loaded tensors):

| Module | Parameters |
|---|---|
| Base_planner_and_action | 15708576 |
| entire_Base_adapter | 2147752352 |
| generator_trainable | 0 |
| shared_language | 1720574976 |
| shared_visual | 406957056 |
| expert_1 | 15708576 |
| expert_2 | 15708576 |
| expert_3 | 15708576 |

The Base breakdown reports its principal visual/language/planner modules; remaining embeddings and other Base parameters are included in entire_Base_adapter. Private expert counts include their condition readers and complete action branches. All generator parameters remain frozen.
