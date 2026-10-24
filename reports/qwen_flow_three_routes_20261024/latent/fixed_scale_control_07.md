# Fixed root scale 0.7: first completed deployment control

This preserves the initial control-analysis snapshot. Later completed-arm and correction status is in [README](README.md); pending items below describe this snapshot rather than live worker state.

This is a **DEV_SELECT control**, not a trained-policy result and not a Navtest score. The complete registered scale comparison also contains scale 1.3, which is still pending. No scale has been selected from this partial result.

The frozen CANONICAL50k Qwen-Flow generator is identical in both arms. L0 samples one root from `N(0,I)`; L0-S samples one root from `N(0,0.7²I)`. Both use one source shift5 ten-step ODE trajectory per scene/seed, with no scorer-based output selection. The same 256 pre-registered DEV_SELECT scenes, eight complete logs, and inference seeds 2101–2105 give 1,280 individual outputs per arm. Seeds are averaged within each scene before paired, scene-weighted log bootstrap (2,000 replicates).

| Quantity | L0, scale 1.0 | L0-S, scale 0.7 | Difference |
|---|---:|---:|---:|
| Canonical v1.1 PDMS | 92.143105 | 92.180332 | +0.037228 |
| NC | 99.218750 | 99.218750 | 0 |
| DAC | 99.453125 | 99.531250 | +0.078125 |
| TTC | 98.281250 | 98.281250 | 0 |
| EP | 83.940326 | 83.920297 | −0.020029 |
| Comfort | 100 | 100 | 0 |
| DDC | 100 | 99.960938 | −0.039063 |

All table values are on the 0–100 reporting scale. The PDMS difference has a paired log-bootstrap 95% interval of **[−0.103974, +0.181690] points**. The observed +0.037-point difference is below the pre-registered 0.5-point useful-gain scale and does not provide evidence that lowering root temperature improves deployment quality. The interval concerns the fixed inference and training seeds; it does not estimate their uncertainty.

Positive scene differences contribute +0.187572 PDMS points and negative differences contribute −0.150345 points, giving the reported +0.037228 net. There are 68 scenes with any positive difference and 103 with any negative difference, but only two improve and four deteriorate by at least 0.005 raw score (0.5 PDMS points). Tiny floating-point or behavioral differences must not be described as severe failures.

Compared at matching scene/seed pairs, scale 0.7 introduces one new zero-score output and recovers two. It introduces two joint NC/DAC/TTC violations and recovers three. NC has no new or recovered violations; DAC has one new and two recovered; TTC has one of each. The absence of new NC events in this finite evaluation does not establish zero true risk.

Any-seed zero-score scenes remain 4 versus 4. All-five-seed-zero scenes change from 2 to 3, and exact-zero scene means from 2 to 3; these are different events from the per-output zero rate, which decreases from 1.328125% to 1.25%.

The independent fixed-reference stratification contains 169 stable-high scenes, two stable-zero scenes, four low-nonzero scenes, and 81 other/unstable scenes. Stable-high mean PDMS changes by −0.025988 points, with no newly zero scene and no paired decline of at least 0.5 points. The two stable-zero scenes remain zero. The low-nonzero subgroup has only four scenes from two logs, so its apparent improvement is too weakly resolved to support a separate claim. None of the 246 scenes passing all reference NC/DAC/TTC samples has an evaluation violation here; that finite observation is not an all-noise safety guarantee.

Each arm uses 1,280 new canonical candidate scorings, 12,800 action velocity forwards, 256 Qwen prefills, and zero optimizer updates. Worker wall is 1,290.63 seconds for L0 and 1,290.87 seconds for scale 0.7 on shared GPUs. These are worker occupancy times, not exclusive GPU compute times or a combined end-to-end duration. Choosing between scales incurs these evaluation costs.

The common analyzer verifies the complete scene/seed denominator, canonical submetrics and simulated consequences, exact schema/protocol/source-base identity, and the native scene-bound CPU FP32 root RNG. It independently reproduces every saved root. The latent evaluator and merged-adapter evaluator use different readout names but are compared only after matching this actual deployment contract. Earlier feedback evaluations with a different root namespace remain auxiliary.

**Current judgment: evidence insufficient for a useful fixed-scale improvement.** This is a control for L1/L2/L3, not evidence for scene conditioning. Those three training arms must be evaluated before assigning a route-level judgment. In particular, the conditional MLP currently moves its distribution more than the global 48-parameter Gaussian under the same learning rate; a weak L2 result alone would not isolate the value of conditioning from unequal policy displacement.

Private aggregate and paired-scene evidence: `/mnt/project/qwen-flow-three-routes-20261024/analysis/L0S_07_v1/`. Public aggregate: [fixed_scale_control_07.json](fixed_scale_control_07.json). No private trajectories or scene identifiers are committed.

Executed analysis command:

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.qwen_flow_exploration.analyze_evaluation \
  --protocol /mnt/project/qwen-flow-three-routes-20261024/common_protocol.json \
  --evaluation /mnt/project/qwen-flow-three-routes-20261024/evaluations/L0S_07 \
  --reference /mnt/project/qwen-flow-three-routes-20261024/evaluations/L0_sample \
  --reference-strata /mnt/project/qwen-flow-three-routes-20261024/common_bank/reference_strata.json \
  --output /mnt/project/qwen-flow-three-routes-20261024/analysis/L0S_07_v1/summary.json
```
