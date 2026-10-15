# Real-image call and retention audit

| Model | Batch | Visual | Language | Context projection | Decoded samples by ID | Fallbacks | Max reported error |
|---|---|---|---|---|---|---|---|
| M_scene | 4 | 1 | 1 | 1 | {"expert_0": 2, "expert_1": 1, "expert_2": 0, "expert_3": 1} | 0 | 3.814697265625e-06 |
| M_set | 4 | 1 | 1 | 1 | {"expert_0": 4, "expert_1": 4, "expert_2": 4, "expert_3": 4} | 0 | 0.0 |
| M_scene | 4 | 1 | 1 | 1 | {"expert_0": 1, "expert_1": 0, "expert_2": 0, "expert_3": 3} | 0 | 2.288818359375e-05 |
| M_set | 4 | 1 | 1 | 1 | {"expert_0": 4, "expert_1": 4, "expert_2": 4, "expert_3": 4} | 0 | 0.0 |
| M_scene | 4 | 1 | 1 | 1 | {"expert_0": 1, "expert_1": 1, "expert_2": 0, "expert_3": 2} | 0 | 3.814697265625e-06 |
| M_set | 4 | 1 | 1 | 1 | {"expert_0": 4, "expert_1": 4, "expert_2": 4, "expert_3": 4} | 0 | 0.0 |
| M_scene | 4 | 1 | 1 | 1 | {"expert_0": 0, "expert_1": 0, "expert_2": 1, "expert_3": 3} | 0 | 3.814697265625e-06 |
| M_set | 4 | 1 | 1 | 1 | {"expert_0": 4, "expert_1": 4, "expert_2": 4, "expert_3": 4} | 0 | 0.0 |

M_scene chooses before any action branch and groups only selected samples; M_set executes each K4 branch once. Synthetic negative tests additionally force mixed selections and invalid selected outputs to verify grouping, order restoration and an explicitly counted Base fallback. Real-image tests use actual Query70k and three trained experts, not fixture weights. Trained bundle fresh-process outputs are separately checked. Base/expert parameter and buffer hashes remain unchanged; shuffled/deleted labels do not alter context or trajectories.

Online/cache numeric comparison replays attested original feature-extraction batches. It does not claim BF16 language kernels are bitwise invariant under arbitrary batch regrouping. This qualification and batch1 deployment timing are distinct protocols. No GT or metric-cache dependency is passed to deployment inference.
