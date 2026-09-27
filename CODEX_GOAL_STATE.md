# Local interaction mask V2 — ACTIVE

Worktree: /mnt/project/VLA-Drive-local-interaction-mask-v2-20260927
Branch: feature/local-interaction-mask-v2-20260927
Starting commit: 227e781dd54f623b12b41bd7f2d2a7b302e0a014
Last verified origin commit: 40363d413405c0f322424a18b26295601f9139d5.
Binding instructions: reports/local_interaction_mask_v2/OBJECTIVE.md and USER_STEERING.md. Preserve previous workspaces, artifacts, failed runs and immutable live code. No formal scientific conclusion yet.

## Binding protocol

Public Qwen/Qwen3-VL-2B-Instruct, revision89644892e4d85e24eaac8bacfd4f463576704203, byte-verified public files. Fresh original DiT/history/Reader/current heads, common language Q/V rank8alpha16 LoRA and driving embeddings. Original Qwen tensors/vision fixed. No private driving weights. Single training seed42; current F0/L0/R0,8x0.5s ego actions, one candidate,10FM steps, no scorer. Public pretraining exposure UNKNOWN. Train7284/978logs, holdout64/59logs, dev1696/16logs, full Navtest12146/136logs: exact disjoint public token/log manifests committed. Navtest is the final planning endpoint, never used for tuning. This finite subset campaign is not the published100k-scene/100k-update DriveDreamer-Policy reproduction. Shared A0 has current perception auxiliary training; it is not that paper's action-only ablation.

## Live runs and resources

Artifact root: /mnt/project/local-interaction-mask-v2-artifacts/20260927
Ledger: budget_ledger.json, independent48GPU-hour cap. Latest snapshot 25.184/48GPUh; read live before new jobs. Includes loading, failures, repair and extraction.
Local0–7: supervisor862274, public_foundation_continued_epoch8_to24_local8, source0937532 in /mnt/project/VLA-Drive-local-v2-runs-foundation24. Immutable epoch8parent, optimizer/RNG/scheduler/data/batch32 preserved. Epoch16 extended to24 using registered training/holdout changes; noPDMS. Continuation cap16GPUh. Latest progress {"step": 4480, "epoch": 19, "offset": 4736, "presentations": 143132}. Do not duplicate/restart/mutate this active run.
Remote training-vla-zt2: all P1 jobs completed;8GPUs verified idle before this update. Check again before allocation. Both hosts8A80080GB. Cross-host16GPU was slower (3.153s/update vs8GPU1.719); use hosts independently. Only explicitly authorized gpu_stress.py occupancy paused; restore commands in artifact resource_actions.json when done. No unrelated tasks stopped.

## Completed evidence and remaining limitations

-58 related tests passed. Real full-Qwen2GPU empty-annotation rank/resume, commonLoRA8GPU gradients, graph exact pause/resume and head exact pause/resume recorded. Original DiT gate0 error0. Public vision cache7348train/holdout+1696dev+12146test complete. Navtest current_v3 records complete; original failed_v2 retained.
-Foundation original8pass premature two-point stop preserved and corrected: training losses were still falling. Continue8→16→24, original16epoch scheduler floor after16. Epoch16 extension report committed with this stage. Reaching24 does not prove convergence.
-P1 current-head fitting64TRAINscenes: recall25.3%→79.0%, precision10.0%→64.3%; fitting-only. Full7284scene8pass current-head generalization pilot: holdoutF1 .164847→.213918, recall .239279→.297079, precision .125735→.167133; selectedepoch8. Native-adapter epoch8 diagnostic only. Final head must train anew on final FP32-adapter current features, max16passes/batch64/lr1e-3, same head for all controls, holdoutF1 including epoch0 selection, max.8GPUh.
-Full native epoch8 original/refined audits:7284each,0fail,32private figures each. Local accepted matches4903→5666; supported/relevant motion-class GT coverage3036/26365→3445/26365. B static/unknown risk associations3958/28046→6415/28046; risk context is not trajectory coverage. Expected uniform eligible-neighbor task validity6.88%→8.31%; severe supervision sparsity, cannot claim graph sufficient. First64TRAIN current-score threshold diagnostic cannot repair this without heavy coverage loss; NO selector/matcher change. Recheck final parent/head, explicitly retain invalid task rates.
-Trained native BF16 LoRA prefix shape changed actual DiT trajectory by .08165m; erased same-shape tail gives0, so numerical rounding, not leakage. Small language LoRA FP32 inference repairs native/append prefix to exact0 on64train+64holdout. Original QwenBF16/DiTFP32 retained. This is an explicit inference numerical change from training. All formal controls must use --language-adapter-precision fp32 and new caches/heads; identity binds precision and code.
-P1 FP32 head→graph→bridge→online pipeline completed.12mode×scene cache/online, visual bypass, target poison, native-prefix and gate0 errors0. Absolute-only1e-5 batch tolerance failed at1.29938e-5, failure preserved; preregistered mixed atol1e-5+rtol1e-5 plus1mmxy check passes with max9.1764e-5m. Same formal batch membership/noise across models; no arbitrarybatch PDMS invariance claim. These architecture tests reuse one engineering graph, not ALL/MASK research results.
-New metric repair: GT current centre, not predicted detection centre, defines dynamic/static groups. GT anchor excluded from model input allowlist. Common-track graph comparison passed on64real scenes with the same engineering model in both arms; no research gain claim. Role-specific coverage and expected mask supervision exported from full saved audits without rebuilding inputs.
-Portable raw licensed NAVSIM preparation:4real scenes/4logs labels/calibration/ego targets exact, speed error<2e-7. Public split IDs, dependency versions and original lightweight action decoder committed.128actual old exports decoder exact. Fresh clean install NOT_RUN.
-Official evaluator: NAVSIM branch v1.1 commit3e8291bfa89ff247231e0227778840cd0a036896, nuPlan e9241677997dd86bfc0bcd44817ab04fe631405b, artifacts/reference_sources. Prior enclosing repo SHAs are not official upstream commits. Direct4synthetic-plan subfactors exact; not learned-model PDMS. Formal scorer records complete source trees/numerical versions; use official clones and navsim Python3.9.

## Next commands and completion conditions

1. Commit/push this metric/coverage evidence stage, verify remote SHA. Stage a bounded final-feature pipeline against a new immutable source worktree; it must wait for foundation supervisor completion, freeze/hash its checkpoint, then use both hosts for final FP32-adapter features.
2. Extract finaltrain/holdout/dev/Navtest features, train shared existing current head up to16passes, refresh all caches. Audit final graph full population and32figures, assess supervision sparsity without GT-dependent selection.
3. Matched ALL/MASK graphs16passes, extend BOTH to32 only via registered world holdout rule; nearest at matching length if budget. Run common-track completion/stationary/conditional/related-vs-weak diagnostics. Track actual masked-task valid coordinates and empty rates.
4. Equal CURRENT/ALL/MASK bridges8→16 matched by training-domain holdout, upstream/DiT fixed. Final online/cache checks. Freeze four models before full1696dev and12146Navtest asynchronousGPU/officialCPU scoring, complete CSV/submetrics/failures/paired log-cluster intervals. Actual DiT vs graph ego reported separately. No current learned-model Navtest score exists.
5. Deliver GRAPH_VALIDITY, LOCAL_COMPLETION, INTERACTION_DEPENDENCE, DEPLOYABLE_PLANNING with convergence/coverage/budget limits, portable commands, ledger/CSVs and verified feature-branch push. No weights/raw/private visuals uploaded. Goal stays ACTIVE while required work remains.

Inspect live progress:
`python -c 'from pathlib import Path; print(Path("/mnt/project/local-interaction-mask-v2-artifacts/20260927/public_foundation_continued_epoch8_to24_local8/progress.json").read_text())'`

If a supervisor has EXITED, resume its exact worker command from supervisor.json using the same immutable cwd, append --resume, preserve --continue-from parent, new supervisor runID and --initial-step savedstep. Never duplicate a live run. Nearest unequal training length cannot establish a relation benefit; no seed43 runs.

## Queued final preparation

Source40363d4 immutable worktree /mnt/project/VLA-Drive-local-v2-runs-finalprep. Artifact final_feature_preparation.json defines all paths/caps,9GPUh downstream reserve, expected24foundationpasses. Remote train launcher704755 waits for foundation completion, then freezes finalparent→train/holdout current→16pass sharedhead→refreshedtrain/holdout. Local evaluation launcher waits for the same frozenparent→Navtest/devcurrent→sameheadrefresh. Check actual PIDs and formal_public_epoch24/pipeline_{train,evaluation}/status.json; wait does not allocate CUDA. Both roles have bounded4hour dependency deadlines. Do not duplicate. P1_common_track_identity_check failed before inference due supervisor-directory incompatibility; sourcebc0adce repaired it and the actual64scene/59log same-model comparison now completes with0failures,43matched tracks, all paired deltas0. Engineering identity check only. Local queued feature launcher PID883196; remote704755.

## Latest operational update

Verified origin a75044fa12bbf7bf4c96eac7e3f4e92f235dd956; current task-statistics/report update still to commit. Shared final-feature code remains40363d4 in immutable runs-finalprep; local waiting PID883196 and remote waiting704755 are alive. Their GPU children have NOT started. Do not update that live source worktree.

Official development full metric-cache regeneration runs locally, launcher885062, immutable runs-metriccache sourcea75044f, CPU8workers, outputs official_dev_metric_v2. First4scene/4log smoke completed0fail. Original remote smoke official_dev_metric_smoke_v1 failed4/4 because official Scene imports NUPLAN_MAPS_ROOT before the explicit constructor map argument; failed artifacts preserved, noGPU cost. Source fix now sets map-root before official imports. Existing QDS train caches cover only175/1696devtokens and contain a different richer schema; do not substitute them. Formal dev scoring should use the new official_dev_metric_v2/cache_index.json only after1696/16logs finish0fail.

CPU environment discovery: remote navsim is Python3.10/Scipy1.11.4/Shapely2.1.2; validated local navsim is Python3.9/Scipy1.13.1/Shapely2.0.7. Use the LOCAL validated environment for official full-cache generation and all formal CPU scoring (bounded aggregate workers), unless a separately validated identical environment is installed remotely. Both GPUs hosts remain in use as planned. Never merge scorer shards with different numeric dependency identities.

Graph trainer now additionally logs per-actual-task scene/hidden-actor/valid-actor/ego-coordinate/neighbor-coordinate/empty-scene counts. This is label-side logging only; no graph, mask, loss or RNG change. Targeted metric/accounting tests4passed; prior fullrelatedsuite58passed. Use the final committed training source, not the older prepare-only worktree, for P2.

Latest live foundation progress: {"step": 4981, "epoch": 21, "offset": 6176, "presentations": 159140}; campaign 27.303/48GPUh.

## Complete evaluator resource and final reporting preparation

Official dev metric caches COMPLETE1696/16logs,0fail,553.44s with8CPUworkers, sourcea75044f. Use artifact official_dev_metric_v2/cache_index.json with the official full-cache score_async entry; do not use the old compact dev adapter or QDSv2cache. Cache figures/GT are not pushed, only hashes/counts/scenes CSV.

New formal reporting code: current-only interaction subset registry (>=2neighbors plus neighbor edge), verified merged-summary identity guard for PDMS comparisons, machine-checked equal8pass CURRENT/ALL/MASK extension to16, and actual-DiT vs graph-ego passage-order proxies. Targeted statistics/flow/schedule tests16passed; geometry tests2passed. Formal results remain NOT_RUN. Next pin this completed source for P2/P3, run real64scene subset generation, then audit the final fresh head/cache and start matched graph runs.

## Foundation COMPLETE and final features ACTIVE (latest)

Public foundation completed24passes/5472updates/174816presentations; original3972cab first8passes plus0937532 continuation preserve optimizer/RNG/scheduler. Frozen checkpoint formal_public_epoch24/foundation/checkpoint.pt SHA256 ac7cfadba298e2980a5c278183ca4a93924e1792fe7a6e4342bb730fafa8a0f4. Public report utility9a1f20e verified complete update/exposure trace and wrote public_foundation_epoch24_report, copied as PUBLIC_FOUNDATION_EPOCH24.json/EPOCHS.csv/CURVES.png. Final native-training holdoutADE6.562679m, currentrecall.289621.22→24trainFM−2.02%,cls−1.52%,box−1.17%; holdoutADE+2.35%,recall+4.95%. Finite stop, not convergence/paper reproduction. BASELINE_MANIFEST.json records full public lineage/runtime/tokenizer/input/normalization; add selected final head later.

Live final feature parent launchers unchanged: remote704755/local883196. Source40363d4 is immutable. Train/current7284 and holdout/current64 have completed; final16pass current-head stage is active on remoteGPU0. Local0–7 continueNavtest current feature extraction, then dev, then sharedhead refresh. Query ledger/current statuses for actual completion. All64finalholdout native-prefix checks passed or head would not have started. Both hosts' GPU resource work is real; idle cards during the one-GPU head stage are not charged.

Remote CPU-only launcher707149 (prepare_final_graph_audit.py, source9550975 worktree runs-main) waits for final train-role completion, then audits7284finalgraphs/32privatefigures, produces role coverage, and prepares train_nearest/holdout_nearest caches. Monitor formal_public_epoch24/audit_preparation_status.json. No P2 run has started yet. Artifact formal_graph_commands.json contains exact prepared ALL/MASK/NEAREST16pass commands (batch32,seed42,32pass scheduler), source9550975; each initial supervised segment cap1GPUh. Review final audit for defects/coverage before launching, reserve remaining planner/Navtest budget. Main ALL/MASK terminal counts must stay matched; nearest matched if budget. Do not accidentally use older source40363d4 for main training since it lacks corrected metrics/task logging.

Actual64scene subset registry CLI check passed:52interaction-proxy,8ego-only,61risk-context-present. Registry rules use current graph only and are frozen beforePDMS; formal full dev/Navtest registries still needed after final refined caches. Latest source9550975 includes matched graph/planner extension and verified-summary paired PDMS guards. Source9a1f20e adds only foundation reporting. Current working-tree reports awaiting stage commit.
