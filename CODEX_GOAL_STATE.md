# Masked Joint Trajectory World Model — ACTIVE

Latest user objective replaces the previous V1.1 scope restriction against joint multi-agent generation. Fixed three current front views; VLA+WM with random actor-trajectory masking, useful BEV representation, and real planning transfer. No RL/scorer/PDMS leakage.

Branch feature/masked-trajectory-world-20260927; source96ff2ee; original worktrees and data untouched. Design and budgets: reports/joint_world/DESIGN.md and execution_budget.yaml. Previous V1.1 runs terminal,2596 updates/1.67101 GPU-hours; planned box4 repair NOT launched. Previous diagnostic artifacts remain immutable in /mnt/project/structured-world-v1p1-artifacts/20260927. New artifacts must use /mnt/project/joint-world-artifacts/20260927.

Next: implement/test joint masked flow model and prediction-only planner bridge, run actual baseline GPU integration and bounded true-data learning; add BEV task objectives; matched controls/holdout/development planning evaluation and final report. Do not describe unit tests as planning success. Future context may only appear in explicitly privileged training reconstruction, never planner or predict_action input.

Resource snapshot: local GPUs0–3 free after own previous experiments;4–7 existing unrelated workers untouched. Verify before use. No new training launched yet.


## 2026-09-27 implemented progress

New joint flow alternates temporal, relative-geometry actor and current-context attention. Fixed ego+64slots, whole-actor random masks with50% all-hidden scene probability, GT context only in explicit training reconstruction. Deployment graph.sample accepts no targets/known futures. Generated graph features feed zero-gated residual bridge to frozen originalDiT. Initial19 CPU tests passed. GPU actual one-scene gate0/poisonedlabels/independentrestore exact; first gate gradient then second graph ego gradient nonzero. Two updates charged. FP32 residual retains nativeBF16 initialego noise through optional initial_noise in original actionhead; old default unchanged. Failures before correction recorded, never excluded from resource accounting.

Frozen current image conditions extracted64 train +64 complete-log holdout. New initial graph-only matched pilots finished1000 updates each, batch8,125epochs; source pinned dfe8122 in /mnt/project/VLA-Drive-joint-runs-dfe8122. Randommask train egoADE2.712m/agentADE3.223m; allmask egoADE3.313m/agentADE3.815m. Agent stationaryADE2.456m, so no motion improvement claim; coverage49.57%. These are graph trajectories, not deployed originalDiT outputs, not PDMS.

Real bridge check: /mnt/project/joint-world-artifacts/20260927/real_pipeline_native_noise/REAL_PIPELINE.json. New artifacts root /mnt/project/joint-world-artifacts/20260927, authoritative budget_ledger.json carries sealed prior2596steps/1.67101h and all new jobs. No convergence repair used yet. Graph training checkpoints500/1000 full state exist; resume wrapper not implemented/validated yet, don't silently restart them.

Full-object train/holdout evaluation launched on local0–3 with sessions67359/97385/89001/92854, respectively randomtrain/controltrain/randomholdout/controlholdout. Verify terminal status before reuse; original4–7 untouched. Need collect summaries and advance interaction/BEV/planning work rather than claim full goal complete. All commands docs/JOINT_WORLD_QUICKSTART.md. New branch not yet pushed; prior V1.1 published branch remains intact.


All four full evaluations are now terminal,0 scene failures each. Randommask holdout egoADE3.270m vs control3.673m; agentADE4.904m vs5.397m, both worse than stationary2.437m. Holdout motion coverage only11.46%, reflecting inherited current detection failure, not omitted from denominators. Randommask dynamicADE6.442m vs stationary6.209m; static4.428m vs1.377m. No current evidence for usable motion planning. Need diagnose absolute XY/noise parameterization, persistent static drift and detector generalization; consider ONE documented anchored-residual repair, not yet launched. Keep masks,split,K and2m metric fixed; improve representation/coverage without dropping hard objects. Final paired planning and BEV work remains required.

Latest metrics reports/joint_world/PAIRED_GRAPH_RESULTS.json and full_evaluation/ with all object CSV. Current newtest runtime source98966b0; learned graph source dfe8122. Local0–3 free, no experiment handle live after confirmed terminal tools; local4–7 untouched. Budget snapshot reports/joint_world/STATUS.json; authoritative ledger in artifacts.
