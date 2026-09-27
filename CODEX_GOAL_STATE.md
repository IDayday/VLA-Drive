# Local interaction mask V2 — ACTIVE

Branch: feature/local-interaction-mask-v2-20260927
Worktree: /mnt/project/VLA-Drive-local-interaction-mask-v2-20260927
Start: 227e781dd54f623b12b41bd7f2d2a7b302e0a014
Latest verified remote: c509a6460516f6c89e5f75c628f9aa6711c0d6cf.
Binding objective: reports/local_interaction_mask_v2/OBJECTIVE.md and USER_STEERING.md.
Artifact root: /mnt/project/local-interaction-mask-v2-artifacts/20260927
Ledger: budget_ledger.json, independent48GPU-hour cap. Latest snapshot32.85GPUh; read live before allocating. All failures/startup/features count. No seed43, no Navtest tuning, no private driving foundation. Do not duplicate active jobs or mutate their pinned source worktrees.

## Completed shared public foundation

Public Qwen/Qwen3-VL-2B-Instruct revision89644892e4d85e24eaac8bacfd4f463576704203; all public bytes verified. Fresh original DiT/history/Reader/current heads; common rank8 Q/V languageLoRA alpha16 and driving embeddings. Original vision/Qwen tensors frozen. Seed42, batch32,24passes/5472updates/174816presentations. First8passes source3972cab, continuation0937532, optimizer/RNG/scheduler preserved. Source/public provenance and full curves are in BASELINE_MANIFEST.json and PUBLIC_FOUNDATION_EPOCH24.json/EPOCHS.csv/CURVES.png.
Frozen final checkpoint: formal_public_epoch24/foundation/checkpoint.pt
SHA256 ac7cfadba298e2980a5c278183ca4a93924e1792fe7a6e4342bb730fafa8a0f4.
Final native-training holdoutADE6.562679m, recall.289621.22→24FM−2.02%,cls−1.52%,box−1.17%; holdoutADE+2.35%,recall+4.95%. Finite stop, not converged or paper-equivalent. Paper reference action-only88.0/video-no-depth88.9/full89.2 uses100k scenes/100k updates; this7284scene pilot is not that reproduction. Shared A0 also has current-perception auxiliary training.

## Active processes and dependencies

Final feature pipeline: immutable source40363d4, /mnt/project/VLA-Drive-local-v2-runs-finalprep.
Remote training-vla-zt2 launcher704755: train_current7284 and holdout_current64 COMPLETE. All64finalholdout native/append-prefix parity checks passed. Shared current head runs on remoteGPU0,16passes/batch64/lr.001/max.8GPUh, epoch0-inclusive holdoutF1 selection, future labels erased. Latest epoch6, selectedepoch5 F1.222624 vs epoch0.206003. After head: train_refined/holdout_refined.
Local launcher883196: navtest_current12146 and dev_current1696 COMPLETE. Waiting for same head, then navtest_refined/dev_refined. Source/checkpoint/head identities bind FP32 smallLoRA inference; never substitute old native caches.
Status: formal_public_epoch24/pipeline_train/status.json and pipeline_evaluation/status.json.
Remote CPU launcher707149 waits for final train-role completion, then final_graph_audit7284/32privatefigures, role coverage, train_nearest/holdout_nearest. Source955097588152912eeaca72eeb4ee9f736e40b530, immutable /mnt/project/VLA-Drive-local-v2-runs-main. Status: formal_public_epoch24/audit_preparation_status.json. Review final coverage before P2 launch.
Prepared P2 exact commands: formal_graph_commands.json (NOT_LAUNCHED), remoteGPU0/1/2, ALL/MASK/NEAREST16passes, batch32,32pass scheduler, each initial cap1GPUh. Holdout specs already exist. Main source9550975 includes corrected static grouping, actual masked-task supervision statistics and paired schedule guards.
Both hosts8A80080GB are authorized. Cross-host16GPU foundation benchmark slower than8; use hosts independently. Only authorized gpu_stress.py paused; restore commands resource_actions.json when all work completes. Check occupancy before allocation; do not stop unrelated jobs.

## Fixed protocol and known limits

Train7284/978logs; trainingholdout64/59logs; dev1696/16logs; Navtest12146/136logs, log-disjoint public manifests committed. CurrentF0/L0/R0 only,8x.5s ego(t0), one candidate10FMsteps, no scorer/RL/PDMS training. Public Qwen pretraining exposure UNKNOWN. Formal QwenBF16 with small languageLoRA FP32 and original DiTFP32; stable graphseed2037/egoseed20260926. NativeBF16 trained LoRA had shape-dependent prefix drift; FP32 residual path repairs prefixexact0, explicitly a changed inference numerical path.
Nativeepoch8 audits7284each/0fail/32figures before and after current-head refinement: matchedlocal4903→5666; supported/relevant motionGT3036/26365→3445/26365; static/unknown B risk3958/28046→6415/28046. Expected eligible-neighbor task valid labels6.88%→8.31%, severe sparsity. NO threshold/matching/GT-dependent mask changes. Final head audit pending. B risk context is not motion coverage. Do not claim graph scientifically sufficient from geometric plausibility or small local ADE.
P1 realGPU online/cache, target poison, nativeprefix and gate0 errors0. Original absolute1e-5 batch test failed1.29938e-5 and retained; preregistered mixed atol1e-5/rtol1e-5 plus1mmxy passes (max9.18e-5m). Main variants share fixed batch layout/noise. Final actual trained CURRENT/ALL/MASK parity remains required.
FullQwen2GPU empty-rank/resume,8GPU commonLoRA gradients, graph/head exact resume, bridge crossGPU parity, public vision cache, raw4scene reconstruction and original action decoder checks recorded in reports. P1 common-track self-comparison64scenes/59logs/43tracks has0deltas/0fail; engineering only.

## Official evaluator resources

Use LOCAL /root/miniconda3/envs/navsim/bin/python3.9.25, NumPy1.26.4/SciPy1.13.1/Shapely2.0.7. Remote identically named environment differs; do not mix CPU shards. Bound aggregate CPU workers<=32; nestedthreads1.
Official NAVSIM branch v1.1 SHA3e8291bfa89ff247231e0227778840cd0a036896; nuPlan e9241677997dd86bfc0bcd44817ab04fe631405b, artifacts/reference_sources. Four synthetic-plan parity checks exact, not learned-model scores.
Dev full official metric cache COMPLETE1696/16logs/0fail,553.44s at8CPUworkers. Index: official_dev_metric_v2/cache_index.json. Old compactdev/QDSv2 caches are incompatible; do not use them.
Navtest readonly full cache index: navtest_cache_index.json,12146/136logs. All正式CPU scores use score_async with official source and identical numeric versions. Always merge_scores even one CPU partition; compare_pdms requires merged identity summaries.

## Next work

1. Finish and audit final current head/refreshed graph caches. Freeze full current-only dev/Navtest interaction subset registries before any PDMS using build_interaction_subsets. Registry rules already committed; P1 actual64scene check passed.
2. Run P2 ALL/MASK16, apply convergence --runs ... at common boundary; extend BOTH to32 per fixed world holdout12→16 rule. Nearest same terminal length if budget; unequal length cannot establish relation benefit. Common-track allhidden/conditional/stationary/dynamic-static and related-vs-equalweak future-condition diagnostics.
3. Equal CURRENT/ALL/MASK bridges8→matched16 by trainingholdout actualDiTADE6→8 rule. Foundation/Qwen/currenthead/graphs/DiT frozen; same projection/adapter capacity. CURRENT may start once final current cache is complete. Reserve full test budget.
4. check_online --variants mapping actual current/all/mask bridge paths verifies each trained path. Freeze all4models before full1696dev and12146Navtest export/scoring on both GPU hosts/local CPU. export_plans --control-output gives separate per-host stop/status under supervisors sharing one proposal bank. Merge complete scene CSV/subscores/failures and paired log-cluster uncertainty; analyze actualDiT vs internalgraphego joint geometry separately. No learned-model Navtest score exists yet.
5. Final four statuses GRAPH_VALIDITY, LOCAL_COMPLETION, INTERACTION_DEPENDENCE, DEPLOYABLE_PLANNING; state coverage/convergence/budget limits, exact restore commands. Commit by stage, push only task branch and verify SHA; no raw data/private figures/caches/weights. Keep goal ACTIVE while required work remains.

Live inspection:
`python -c 'from pathlib import Path; print(Path("/mnt/project/local-interaction-mask-v2-artifacts/20260927/formal_public_epoch24/head/progress.json").read_text())'`
After a supervisor has EXITED, resume exact saved command in supervisor.json with same source/data, --resume and new runID/--initial-step; preserve optimizer/scheduler/RNG. Never duplicate a live run.

## Latest: final caches complete and main training ACTIVE

Finalhead16passes/1824updates selectedepoch16, F1.23713956 vs initial.20600273; SHA2560c002ffcb58d76bcce4983048ad635068b467988244b34447844f2316dd5c6ae. All four refined caches COMPLETE21190scenes/0fail, commonidentity428fc9e88b0ebece30a3a3ee3e9caf0bd8a5d195700559fdc0fbf7b6df379bfd. Source40363d4 feature launchers are terminal; do not rerun. Full graph audit7284/0fail/32privatefigures COMPLETE. Roles: supported/relevant motion4406/26365, Bstatic/unknown7467/28046; eligible61894, withanyfuture7240, expecteduniformneighbortaskvalid10.82%. Graphqualitystilllimited, reviewaudit_review.json; no rule orGTdependentmask changes. Main P2 is a finite diagnostic under this explicit limitation.

Actual training source9550975 (immutable runs-main): remoteG_LOCAL_ALL supervisor711288 GPU0; G_LOCAL_MASK711335 GPU1, initial16pass cap1GPUh each. Same epoch0completeholdoutsummary exactly. LocalP_CURRENT supervisor896072 GPU0,8passes cap1.5GPUh; independent current-memory control overlaps P2. Prepared exact commands formal_graph_commands.json and formal_planner_commands.json. Artifact launch_prepared_jobs.py launches named initial jobs with source/resource/reserve checks; formal_launches/ records scripts/commands/PIDs. Nearest preparation remains running under remoteCPU707149; check status before G_NEAREST_MASK launch on remoteGPU2. Do not duplicate existing runIDs.

Full current-only subset registry COMPLETE: dev1696/16logs, interaction1208/ego-only371/risk1572; Navtest12146/136logs, interaction8748/ego-only2839/risk11030. Files formal_public_epoch24/{dev,navtest}_subsets.json; summary/hash reports committed next. No PDMS opened.

Evaluation sourcec66e6f86636377c575990598e4f33671e67fb495 in immutable runs-evaluation. CPUsubsetlauncher895671 has completed. New check_online --variants checks actual trained CURRENT/ALL/MASK paths and synchronized cached/full-image latencies. export_plans --control-output supports independent host supervisors; actual2GPU pause→resume test passed128engineeringexports/0fail in.0272GPUh, source78aa360 preserved. Localtestdirectory37passed; earlier fullrelatedsuite58separate. summarize_experiments rebuilt realP1 graph and bridge curve reports, rejects gaps/unequalepoch exposure.

Prepared evaluation_preparation.json, evaluation_variants.json and parity_variants.json reference futurefinalpaths; NOT model locks and NOT launched. Freeze models only after registeredgraph16→32/planner8→16 decisions, final actual onlinechecks, then officialdev/Navtest scoring. Use cached feature exports16shards acrossbothhosts, distinct control-outputdirs, localCPU4models×8workers max32. Always merge_scores. Latest verifiedremote10662f0; c66e6f8 pushsucceeded, verifySHA next. Current uncommittedreports reflect finalhead/cache/audit and initial P2 proof.
