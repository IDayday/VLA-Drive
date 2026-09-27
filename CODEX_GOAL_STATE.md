# Local interaction mask V2 — ACTIVE

Worktree: /mnt/project/VLA-Drive-local-interaction-mask-v2-20260927
Branch: feature/local-interaction-mask-v2-20260927
Starting commit: 227e781dd54f623b12b41bd7f2d2a7b302e0a014
Last verified origin commit: 44f315728006ac1b6bdfd2f1c0cc255166b04fd1 (this stage is not committed yet).
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
-New metric repair: GT current centre, not predicted detection centre, defines dynamic/static groups. GT anchor excluded from model input allowlist. Common-track graph comparison added; real GPU execution still pending. Role-specific coverage and expected mask supervision exported from full saved audits without rebuilding inputs.
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
