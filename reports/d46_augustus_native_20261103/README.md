# D46 AUGUstus native RL — current research

Two independent formal algorithms are running on 48 GPUs: AUG1_FULL (four nodes /32 GPUs, B128/G8/M1) and AUG2_LORA (two nodes /16 GPUs, B64/G8/M1). Both start from the same verified D IL46 policy, PDMS **89.49552098**, fresh Adam update0. The actual native0 payloads have empty optimizer state, scheduler0 and all-rank cursor0. Qualification weights never initialize formal training. Older D routes retain their code, complete results and native state; automatic restart is fenced.

Current D-series complete NAVTEST best remains **90.32032058**, prior FPO at0.5 scene epoch, +0.82479960 over D46 IL; zero scenes repaired235/new73. No new AUGUstus deployment score is claimed. The historical IL70 SafetyEP91.45359756 uses a different parent.

Live HTML: `/mnt/project/qwen-world-performance-rl-20261103/report.html`. Current runs/results go first. Updates every30minutes and immediately on a complete result/best improvement. Background native checkpoint events trigger evaluation; no manual progress polling.

| Route | Scope | Trainable parameters | Fresh scenes / candidates per update | Initialization |
|---|---|---:|---:|---|
| AUG1_FULL | All28 text transformer layers/norm, existing action/time/P/ego/role interfaces |1,472,810,499|128 /1024|sameIL46 /emptyAdam|
| AUG2_LORA | Text r16, last8 FiLM low-rank, action I/O/position |7,010,307|64 /512|sameIL46 /emptyAdam|

Both use the pinned official [AUGUstus JaT implementation](https://github.com/haha-yuki-haha/AUGUstus/tree/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/recipe/transfusion/algorithms): three lambda_t transitions, independent Gaussian innovations with per-dimension std multipliers0.05/0.05/0.02, a shared initial root within a group, coordinate-mean ratio/KL, clip0.2, fixed reference beta0.001 and canonical group-relative sample std+1e-6. The vendor files retain their Apache notice and exact upstream hashes. Unlike the previous D0/FPO engine, the target is the original JaT action objective.

These are explicit D/Qwen3-VL action-only adaptations. Vision and input embeddings remain frozen for the frozen-observation cache. No textual CoT branch is claimed. Training uses the explicit linear three-step grid; deployment remains the original B1 deterministic clean-x K10/shift5/seed2101, with no reranking or scorer input. The training/deployment grid mismatch needs real B1 diagnosis. Reference parameters do not alias any current trainable parameter.

First genuine 32/16-GPU qualifications and own-native resume to update2 completed. All48 saved private RNG streams reproduce the exact next three-step draw sequence; full source/material/Adam/scheduler transactions were restored. Tiny8-CPU/Gloo tests separately checked parameter/Adam/RNG exact roundtrip and M2 pending material. This is not a separate uninterrupted full-GPU parameter-equivalence experiment.

The dedicated two-node FPO execution benchmark improved full-cycle wall32.872→23.364seconds (1.40693×; matched window1.41027×), including real sampling/scoring/two Adam updates. Fusion changes BF16 grouping and is not gradient-bitexact. The new JaT routes reuse bitexact prefix/invariant-cache improvements but have no FPO MC objective; the FPO speed ratio must not be advertised as their measured throughput.

A meaningful reward conflict already appears in a bounded formal prefix: in FULL3 of512groups,9 TTC-failing candidates received positive canonical advantages despite a strict-safe score1 alternative. This is a real objective tradeoff, not a density bug or a proven deployment regression. See the complete finite diagnostic with hashes; no hot mutation of the running objective was made.

Training uses all103288 legal NAVTRAIN scenes, uniform shuffled coverage; candidate counts never multiply the number of independent scenes. At G8 a valid epoch costs826304 candidate scores before padding. Registered target6epochs is a resource envelope, with continuation/revision driven by complete real B1 results. All algorithms have independent actor/Adam/scheduler/RNG/checkpoint namespaces.

Early complete NAVTEST gates use native FULL100/200 and LORA200/400 (~0.124/0.248 valid scene epochs), followed by native0.5/1/2/3/4/6 gates. Only complete12146scene/44log/invalid0 results are published. NAVTEST is used for checkpoint selection (NAVTEST_SELECTED=true), never for training priorities/labels/teacher trajectories. Evaluation runs only on authorized spare GPUs and does not occupy the six training nodes.

Scientific runtime: `runtime/augustus_science_v1`, commit78cea8f14907762f790bed87684cf87612a42456. Shared native trainer/actor/data/scorer/checkpoint machinery is reused; no full Trainer copy. Independent parent/hash/initialization, actual resume, complete previous evaluation, first real diagnostic and execution benchmark evidence are in `results/`.
