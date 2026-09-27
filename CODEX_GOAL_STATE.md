# V3 structured mechanism campaign — COMPLETE

Branch: experiment/joint-local-scene-v3-mechanism-20260927
Reviewed base: afa599762de491d4a9f87310aae41a4fc8d5e76b
Locked training source: 6dd1ab3001940d9e22e6615a955308d498316d81
Last verified implementation/analysis commit: b7d97b3122cf7fd1b42b4ee912fe6f522fccda44
Final report revision: the commit containing this completed state; verified remote SHA is also recorded in the local artifact DELIVERY.json after push.

## Completed scope

Real actual-model startup4vs2+2 PASS; small64 J_ALL/J_MASK512updates each; fresh seed42 AND43 full paired64epochs/14592updates/466176presentations per arm; all100 registered milestone evaluations; all4 endpoint K8/relation ablations; paired whole-log bootstrap; exact paired data/noise/initialization verification and load/re-evaluation parity.57CPU regression tests PASS; separate6 analysis-focused tests overlap that suite. Single joint generator, executed ego remains joint slot0, neighbor yaw canonical0, no Qwen feature input, unused condition projection frozen.

Read reports/joint_local_scene_v3/mechanism_training/RESULTS.md and CONCLUSIONS.json. Both seeds show basic learning and within-model conditional completion. J_MASK all-hidden ego/neighbor is worse than J_ALL at the common64 endpoint. Neighbor strong-vs-weak condition sensitivity is supported; ego evidence remains inconclusive. Convergence: UNCONVERGED. Visual deployment/Navtest/PDMS NOT_RUN. No private V1/V2 trained baseline weights used; fresh initialization per seed.

## Budget and artifacts

Independent artifact root: /mnt/project/v3-mechanism-artifacts/20260927
17 ledger runs complete, no active campaign training/diagnostic process.59408real optimizer updates,0synthetic;1881344real training sample presentations. Total GPU-hours6.011867536041396/48, remaining41.988132463958604; real update cap61464, remaining2056. Remaining budget is not authorization to add experiments beyond this completed plan.

Old review ledger SHA unchanged:06464b57b2709b28a6df9bd4bd74467e93a9ffe0b9d15c7c0c4c0c94f82008dc; real0/synthetic16 remains sealed. Old V2step1887/controller not restored. Original code workspace/data/cache/checkpoints/reports untouched. Locked training worktree /mnt/project/VLA-Drive-v3-mechanism-run-6dd1ab3 remains clean at6dd1ab3.

Public metrics: reports/joint_local_scene_v3/mechanism_training/results/endpoint_queries_anonymized.csv (1944rows), condition_ablation_queries_anonymized.csv (1944rows, including308NOT_APPLICABLE), scene/log paired CSV, all milestone/gradient/role/cost summaries and learning plots. Private checkpoints, joint sample banks and8fixed real scenes x4models remain local; see LOCAL_ARTIFACT_INDEX.json. No private scene pictures, raw data or large weights uploaded.

## Resume/reproduction boundary

All formal runs COMPLETE64; do not resume them to more epochs, rerun the launch sequencer, or rerun campaign preparation over this directory. The previous continuation state ready_for_results_review has been handled by the final report/export; it does not mean training is waiting for review.

CPU-only reproduction of existing experiment statistics, using a NEW output directory:

```bash
cd /mnt/project/VLA-Drive-v3-mechanism-20260927
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python -m tools.joint_local_scene_v3.analyze_campaign --campaign /mnt/project/v3-mechanism-artifacts/20260927 --output /mnt/project/v3-mechanism-artifacts/20260927/analysis_reproduction
```

Actual GPU endpoint re-evaluation and exact original training/resume commands are in REPRODUCTION.md and commands.jsonl. No blocker remains within this campaign. Next separately authorized phase would need bbox assignment/same-track motion labels, Reader/head/Qwen joint gradients, predicted graph transition and camera predict_action with joint ego slot0. Full visual training, new provider, Navtest/PDMS, RL/scorer/second execution head and oldV2 remain outside this completed scope.
