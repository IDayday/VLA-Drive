# Masked Joint Trajectory World Model — bounded work complete

2026-09-27. Engineering READY for the explicitly supported fixed-upstream single-GPU production path; research INCONCLUSIVE. All training, inference, task evaluation, official development PDMS, diagnostics and final reports are complete. No experiment jobs remain. Final branch publication is verified against remote SHA in the delivery response.

## Workspace and identities

- Development: /mnt/project/VLA-Drive-masked-trajectory-world-20260927
- Branch: feature/masked-trajectory-world-20260927
- Remote: git@github.com:IDayday/VLA-Drive.git
- Latest valid code/results commit before this final documentation: 9a5ad9d. Current exact delivery commit: git rev-parse HEAD.
- Immutable scientific sources: fed64aa graph training; 7bd8901 image planner transfer; b820f4c BEV transfer; cea37ea final BEV evaluation; 1eda084 final learned online checks.
- Baseline compatible code0ecd2ae; original training SHA UNKNOWN. Original checkpoint9445f9da…; inherited post-Qwen world checkpointfe9aef18… . Full hashes and six final checkpoint identities: reports/joint_world/FINAL_MANIFEST.json.
- Original workspaces, raw logs, existing caches and checkpoints were not overwritten. Weights/data/images remain in private artifact directories.

## Completed evidence

7284 scenes/978 logs, matched graph pair3642updates/8passes each; four planner transfer variants1821updates/4passes each. Fixed milestones retained, no PDMS-based checkpoint selection. Long masked trials use Bernoulli whole-actor subsets plus50%all-hidden, not exact-one masking. Exact-one mode has a16-step real engineering check only.

Full1696-scene officialv1 PDMS: baseline93.14457488, randommask93.32698483, allmask93.21436495, BEV taskoff93.31226551, BEV taskon93.19507936. Allfive0failures. Random-minus-control+0.11262points,95%log-clusterCI[-0.00517,+0.21984]; BEV taskon-minus-off−0.11719,CI[-0.32541,−0.00144]. Single seed; no stable positive claim. Complete small-score CSV: reports/joint_world/planning_all1696/scene_metrics.csv (8480 rows). All paired comparisons and proposal hashes retained.

BEV dense GT-cell motion ADE improves5.2664→4.9110→4.6691m over1/2/4passes; occupancyIoU nonmonotonic. Fullset detectionrecall13.26%, futurepointcoverage13.38%, generatedagentADE5.70–5.84m versusstationary2.38m. Neither stronger auxiliary metrics nor reaching the training cap establishes world-model quality, convergence or planning benefit.

29 related tests pass. Original1696 trajectories and all7 PDMS factors match archived baseline exactly. Actual trained image and BEV policies on4real current-image scenes each pass bitwise cached/online actions and joint trajectories, poisoned-label independence and strict restore. Real GPU continuous/resume checks pass. Real2GPU NCCL graph-module accumulated gradients, unequal/empty-agent-label rank and resume pass; production fullQwen DDP training is NOT implemented/claimed. Real ego-onlyFM reaches trainableBEVencoder/fusion through fixedgraph/DiT.32real diagnostic figures remain private with published hashes. Current/BEV cache malformed-version/sensor/label fields rejected. Full onlineBEV~0.752s,image~0.520s on4scenes, excludespreprocess/model-loading.

BEV is frozen pretrained DAV2visualbackbone plus NEW calibrated geometry, not pretrainedBEV; FOVsupport is not occlusion confidence. BEV enters AFTER Qwen, through graph and originalDiT adapter. Only image-conditioned worldqueries go throughQwen. Graph then bridge are trained sequentially; originalQwen/vision/currentheads/DiT and provider remain fixed. NoRL, learnedscorer, candidate selection, futureRGB or PDMS labels.

## Budget and limits

Authoritative ledger: /mnt/project/joint-world-artifacts/20260927/budget_ledger.json, copied to reports/joint_world/RUN_LEDGER.json.
Final21835/24000optimizerupdates,7.16919871687889/48GPUhours, including sealedV1.1prefix2596updates. EarlierV1campaign is separately sealed; repository-root execution_budget.yaml belongs to that older campaign. Current budget: reports/joint_world/execution_budget.yaml. Remaining2165updates insufficient for another full1821×2seedpair.1of2repairhypotheses used.4failed earlyruns and the repaired RNG comparison failure remain documented and charged.

NOT_RUN: second full training seed pair, long exact-one mask comparison, fullQwen/graph/DiT joint fine-tuning/productionDDP, Navtest. Basepretraining development exposure UNKNOWN. No automatic additional training. Resources used localGPUs0–3 only; unrelatedGPUs4–7 tasks untouched.

## Reports and repeat command

Final self-contained report: reports/joint_world/FINAL_REPORT.md.
Current status: reports/joint_world/STATUS.json. CSV/completeness/hash audit: reports/joint_world/DELIVERY_CHECK.json.
Quickstart and full historical commands: docs/JOINT_WORLD_QUICKSTART.md; exact scientific arguments in manifests/ledger.

```bash
cd /mnt/project/VLA-Drive-masked-trajectory-world-20260927 && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /root/miniconda3/envs/ddp/bin/python tools/joint_world/report_planning.py --artifacts /mnt/project/joint-world-artifacts/20260927 --output reports/joint_world/planning_all1696 --runs planner_dev1696_baseline planner_dev1696_randommask planner_dev1696_allmask planner_dev1696_bev_control planner_dev1696_bev_tasks
```

This re-creates reports without retraining. Do not restart completed runs or reduce ledger charges. Exact resume is limited to explicitly paused, fully accounted runs with pinned code, same hardware/dtype/cache identities and original arguments plus --resume <same-output>/checkpoint_<step>.pt. No activity waits or additional evidence are pending. New scientific work requires its own bounded plan.
