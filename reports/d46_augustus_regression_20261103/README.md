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

Actual fixed TRAIN64 follow-up (already exposed scenes, not validation or NAVTEST):

| Policy | Native K10 PDMS | Old K3 PDMS |
|---|---:|---:|
| IL46 |89.05098830|88.98664650|
| FULL100 |85.72099889|85.72859137|
| LoRA400 |89.68292224|88.13420037|

FULL drops under both grids: grid mismatch alone is insufficient to explain the collapse. FULL TTC decreases98.4375%→82.8125%, while EP increases81.847%→88.230%. LoRA's training-panel improvement does not transfer to complete NAVTEST. This strengthens the priority of safety credit and update stability; it does not isolate their causal contributions.

The actual32-GPU corrected FULL first update, native pending-inner audit and exact cached-condition native B1 parity passed all32ranks. Official single-B1 qualification uses zero output difference, not a permissive score proxy. Native own-resume and LoRA16 qualification follow. Initial resource refusals were prechild: D66 IL evaluation acquired otherwise idle GPUs; keep those evaluations intact and use measured authorized sharing. Failed receipts remain, and no optimizer reset is hidden.

32 combined science/runtime/operations CPU tests passed after consolidation; optional physical-covariance tests2 and report notifier tests3 also passed. Both corrected real multi-node pending-native resumes completed at2 updates with all48ranks, frozen reference and RNG equality. The two independent formal runs have been dispatched using qualified source0951451; the single background owner validates formal native0/empty Adam and evaluates saved checkpoints automatically.

Nonzero FULL100 and LoRA400 native-actor versus deployment-loader **all parameter/buffer tensors** match exactly on CPU, with genuinely nonzero learned deltas. This is not a new GPU-output parity or score. Detailed algorithm audit, remaining B1-objective gap and the newly found upstream per-time quantile versus D-global normalization mismatch are in [ALGORITHM_IMPLEMENTATION_AUDIT.md](ALGORITHM_IMPLEMENTATION_AUDIT.md). Full103k TRAIN exploration calibration is prepared but **not enabled** in the current two formal runs.

## 2026-11-04：适配版首轮完整结果

各自只曝光6,400个TRAIN场景（0.06196E），相同IL46=89.4955209759。两点均为完整NAVTEST12,146场景、44物理log、invalid0、真实B1/K10/shift5/seed2101。

| 适配版 | Adam | PDMS | 相对IL46 | 44log差值CI95（PDMS点） | 修复零分 | 新增零分 |
|---|---:|---:|---:|---|---:|---:|
| FULL |100|89.11003825|-0.38548272|[-0.6999,-0.0816]|56|139|
| LoRA |200|89.45854342|-0.03697756|[-0.1450,+0.0798]|16|24|

尚无净提分。LoRA基本持平，FULL仍小幅退化；FULL的EP+0.9995pp而TTC-1.4243pp，NC/DAC也下降。原高分队列在全人口的贡献FULL为-0.6680、LoRA为-0.1459，仍抵消零分修复收益。只有早期0.062E，不作最终上限判决；下一点FULL200/LoRA400为0.12393E，才对应旧FULL100/LoRA200的曝光，不能把不同曝光点的差值解释为纯适配收益。

另已CPU检查每组前6,400场景的全部1,600份rank rollout材料。组std下完全安全组优势L1系数占66.1%/67.3%；同材料改batch std反事实为16.4%/17.3%，安全混合组约83%。这提示跨场景安全/EP信用被组std重加权，值得作为下一适配重点；它不是实际参数梯度占比，也不是已经训练出的收益。`audit_training_normalization.py`可复算到独立文件，原始证据保留。当前两组并未改用batch std，仍按原配置继续训练与自动评测。
