# D46 AUGUstus regression diagnosis and corrective implementation

The original FULL and LoRA runs are preserved; neither is resumed with silently changed mathematics. All scores below are complete NAVTEST B1/K10/shift5/seed2101, 12,146 scenes, 44 physical logs, invalid=0. Parent D IL46 is 89.4955209759. NAVTEST_SELECTED=true; no scene labels or priorities enter training.

| Original run | Adam updates | PDMS | Delta versus own IL46 | Repaired zeros | New zeros |
|---|---:|---:|---:|---:|---:|
| AUG1 FULL | 100 | 81.2731397755 | -8.2223812003 | 130 | 1234 |
| AUG1 FULL | 200 | 83.8037928650 | -5.6917281108 | 130 | 1046 |
| AUG2 LoRA | 200 | 89.3801404327 | -0.1153805432 | 49 | 73 |
| AUG2 LoRA | 400 | 89.1605578447 | -0.3349631311 | 22 | 92 |

Confirmed engineering defect: both runs reached and sealed 0.5 exposure epoch (FULL404, LoRA807), then rank0 failed publishing the first evaluation request because its parent directory did not exist. The failure is independent of the earlier degradation. Fix creates the request directory before atomic publication, with a first-gate regression test. Native states and Adam are preserved.

Confirmed optimization facts, not isolated causal attributions:

* The old training grid [1,2/3,1/3,0] differs from D46's native ten-step shift5 deployment. Initial parameter parity alone cannot validate this transfer.
* M1 always evaluates fresh ratios exactly1; PPO clipping therefore does not constrain the single Adam step. It is not evidence of a broken log-prob formula, but removes a hoped-for trust mechanism.
* FULL reference coordinate KL grew from0 to0.899 at100 and2.774 at400 on changing TRAIN batches. Fixed coefficient0.001 is weak for this migrated full scope. These are not paired fixed-panel KL measurements.
* Real TRAIN collections contain positive canonical advantages for TTC0 candidates despite a strict-safe score1 alternative. Existing SafetyEP credit resolves this ordering counterexample without changing official PDM scores.
* Vendored kernel/reduction/direction tests match the pinned upstream; no probability-sign defect has been found. FP32 parameters/Adam, independent frozen reference and source freshness are checked. Do not relabel these tests as performance evidence.

Corrective independent recipes:

| ID | Scope | B/G/M | LR | Reference coordinate KL beta |
|---|---|---|---|---|
| AUG3_FULL_ALIGNED | same full trunk/interfaces |128/8/2|2e-7|0.1|
| AUG4_LORA_ALIGNED | same LoRA/interfaces |64/8/2|1e-6|0.1|

Both keep official lambda_t JaT mean-coordinate probabilities, share the deployment seed2101 root, execute seven exact native deterministic steps, and explore only the final three native shift5 transitions. Credit is hard safety then EP with group sample-std; official scorer remains unchanged. Both start from the identical verified IL46 and empty Adam, not old AUG RL weights. Warmup/hold/cosine exposure clock is unchanged. These are local D46 adaptations, not complete official AUGUstus reproductions.

Multiple mechanisms change together for performance; this is not a clean attribution ablation. Stronger KL and lower FULL LR are registered initial choices, not proven optima. Targeted CPU tests:12 passed; actual8-CPU/Gloo native checkpoint/resume tests:4 passed. Real GPU qualification, fixed TRAIN K3/K10 diagnostics and complete official evaluation must determine whether the correction helps. No new higher score is claimed.

Live assets: `/mnt/project/d46-priority-rl-20261103/analysis/AUGUSTUS_REGRESSION_20261103/`.
Live HTML: `/mnt/project/qwen-world-performance-rl-20261103/report.html`.
