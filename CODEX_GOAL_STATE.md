# Local interaction mask V2 — ACTIVE

Branch: feature/local-interaction-mask-v2-20260927
Worktree: /mnt/project/VLA-Drive-local-interaction-mask-v2-20260927
Start: 227e781dd54f623b12b41bd7f2d2a7b302e0a014
Latest verified remote: 228c0b1d698ab410c6e804a146eeb271a156203c.
Binding task: reports/local_interaction_mask_v2/OBJECTIVE.md + USER_STEERING.md.
Artifact root: /mnt/project/local-interaction-mask-v2-artifacts/20260927
Final stage: formal_public_epoch24. Active evaluation output: evaluation_v2.
Do not duplicate live jobs, mutate pinned source, restart completed foundation/features, or mark goal complete before final delivery.

## Binding protocol

Independent public-Qwen origin; no old private driving foundation. Seed42 only. CurrentF0/L0/R0, calibrated transforms/navigation/allowed ego history. Train7284/978logs, trainingholdout64/59logs, development1696/16logs, finalNavtest12146/136logs. All log-disjoint within this campaign; public Qwen pretraining exposure UNKNOWN. Final acceptance is locked fullNavtest v1PDMS, one candidate/10FMsteps, graphseed2037/egoseed20260926, no learned scorer/RL. EPDMS/Hard NOT_RUN. Never tune from Navtest scores.

Independent48GPU-hour cap includes loading, failed runs and features. Live ledger: budget_ledger.json; read current usage/reservations before allocations. Last snapshot approximately36.5GPUh. Local and training-vla-zt2 eight A800s each authorized; only previously authorized gpu_stress.py occupancy stopped. resource_actions.json records restoration; restore only after all GPU work ends and devices are free. Cross-host16GPU foundation measured slower than8; hosts run independent jobs.

## Completed evidence

- Foundation24passes,5472updates/174816presentations; public Qwen revision89644892e4d85e24eaac8bacfd4f463576704203, fresh originalDiT/history/Reader/heads, common rank8Q/V LoRA/driving embeddings. Source3972cab then0937532 with optimizer/RNG preserved. Frozen checkpoint formal_public_epoch24/foundation/checkpoint.pt SHA256ac7cfadba298e2980a5c278183ca4a93924e1792fe7a6e4342bb730fafa8a0f4. HoldoutADE6.562679m, still changing near finite stop; not convergence or paper reproduction.
- Shared current-head16passes/1824updates selectedepoch16. F1.206003→.237140; precision.188555/recall.319453. Head selected.pt SHA2560c002ffcb58d76bcce4983048ad635068b467988244b34447844f2316dd5c6ae. Only532752current-head parameters trained, future labels erased. Cost.4555GPUh.
- Final refined train/holdout/dev/Navtest caches21190/0fail, identity428fc9e88b0ebece30a3a3ee3e9caf0bd8a5d195700559fdc0fbf7b6df379bfd, source40363d4. Formal QwenbaseBF16/smallLoRAFP32/originalDiTFP32. SmallLoRAFP32 repairs BF16 sequence-shape prefix drift; explicitly changed inference numerics, not oldBF16 equivalence. Do not reuse old caches.
- Full graph audit7284/0fail/32privatefigures. Supported/relevant motionGT4406/26365=16.71%; Bstatic/unknown7467/28046 separately. Eligible61894/withfuture7240; expecteduniformneighbor-task valid-label10.82%. Ego-only1223. NoGT-dependent resampling or threshold/matcher changes. Finite diagnostic under severe graph-quality limitation. FINAL_GRAPH_REVIEW.json.
- Current-only analysis groups frozen: dev1696/interaction1208/ego-only371/risk1572; Navtest12146/interaction8748/ego-only2839/risk11030. Overlapping prediction proxies, full population primary.
- P1actual fullQwen2GPU empty-rank/resume,8GPU LoRA gradients, graph/head exactresume, bridge crossGPU<=1.2e-7, raw4scene rebuilding, originaldecoder128exports, online/targetpoison/nativeprefix/gate0 tested. Initial absolute1e-5 batchfailure retained; declared mixedatol/rtol1e-5 plus1mmxy passes, maximum9.18e-5m. Final actual-trained3path online check still queued.
- NativeA0realimage→DiT4scene timing .6334/.4300/.4285/.4286s; exactcached/nativeprefix/actions. Excludesdecode/crop/resize, loadedwrappermemory includes unusedworldparams. NATIVE_A0_LATENCY.json.
- Parameters: graph949450; eachplanner17315969trainable. Full original FlowmatchingActionHead819503620frozen inP3; constructor802967040is INNER diffusion transformer only. PARAMETER_COUNTS.json.
- Current tests37passed in LATEST_TEST_RUN.json;2new combined-result targetedtests separately passed. Earlier related58tests separate. Do not sum or call freshinstall tested.

## Live controller ownership

Training source955097588152912eeaca72eeb4ee9f736e40b530 in immutable /mnt/project/VLA-Drive-local-v2-runs-main.
P2ALL/MASK completed16 and registered EXTEND_BOTH_TO32 (12→16worldchange10.65%/9.33%, agents worse thanstationary). Continuations remote supervisors717727/717775 onGPU0/1, newrunIDs *_e32, resumeinitialstep3648. Latest both23passes; nearest32continuation supervisor719359 remoteGPU2, latest19passes. Query ledger for current state. Never duplicate.
P_CURRENT8passes/1824updates complete. Holdout actualDiTADE epoch0=6.530142,6=6.528674,8=6.688454. This will trigger joint16rule only after other two bridges reach8.

Main CPUcontroller899209: continue_main_phases_v2.py, status formal_public_epoch24/main_phase_controller.json. Waits ALL/MASK32, launches localGPU1P_LOCAL_ALL/GPU2P_LOCAL_MASK8(cap1.5GPUh each), then jointly resumes allthree16ifregistered6→8rule fires(cap1GPUh each), reserve4. Reads noPDMS. Exact commands formal_graph_commands.json/formal_planner_commands.json and formal_launches/. Originalv1 had a supervisor-registration race, no model restart; failedstate/script retained, ORCHESTRATION_REGISTRATION_REPAIR.json.

Graphdiag CPUcontroller898556: diagnose_final_graphs.py, immutable evaluationsourcec66e6f8; waits main32 then localGPU4 common-trackMASKvsALL, related-vs-equalweak privileged future-condition removal forbothgraphs, nearestsame32 common-track comparison, graph_learning_curves. Eachcap.12GPUh/reserve4. State graph_diagnostic_status.json. No duplicate diagnostics.

Finaleval CPUcontroller899233: evaluate_final_locked_v2.py, pinnedsourcec66e6f86636377c575990598e4f33671e67fb495 in /mnt/project/VLA-Drive-local-v2-runs-evaluation. State formal_public_epoch24/evaluation_v2/status.json. Waits allfinalgraph/planner schedules, then localGPU3actualonlineparity(cap.2GPUh), hashlocks all4models BEFORE anyPDMS, full1696dev then12146Navtest. Eachsplit16GPUshards acrosshosts (offset0/8, batch16), distinct perhostcontrol dirs, caps .5GPUh/hostdev and1.5GPUh/hostNavtest. FourLOCAL officialCPU scorers×8workers=32aggregate consume atomic outputs asynchronously. Mergeallrows,5pairedcomparisons and frozen subsets. Originalevaluation/ contains failedwaiteronly; use evaluation_v2. Ifwaiterfails, inspectchildren/status/ledger before recovery; childrenmaystillrun. No scores begun at this snapshot.

## Official CPU evaluator

LOCAL /root/miniconda3/envs/navsim/bin/python3.9.25 with NumPy1.26.4/SciPy1.13.1/Shapely2.0.7. Remote same-namedenv differs: do not mix. GPUenvironment /root/miniconda3/envs/ddp/bin/python. Set NUPLAN_MAPS_ROOT=/mnt/navsim/maps, nestedthreads1.
Official reference_sources NAVSIMv1.1 branchSHA3e8291bfa89ff247231e0227778840cd0a036896, nuPlan e9241677997dd86bfc0bcd44817ab04fe631405b. Exact allfactor parity on4syntheticplans, not learned-modelscores. Full officialdev metriccache1696/16logs/0fail, official_dev_metric_v2/cache_index.json. Navtestfull12146/136 index navtest_cache_index.json, readonly cache /mnt/project/DriveDreamer-Policy/navsim_exp/eval_v1_1/metric_cache_navtest. Oldcompactdev/v2cache incompatible.

## Remaining delivery

1. Monitor controllers; repair genuine failures while retaining artifacts, no new seeds/variants/tuning.
2. Review P2final common-track/conditional/dynamic-static/stationary, actual per-task empty supervision and relation diagnostics. Build planner_learning_curves using summarize_experiments --phase planner after allthree terminal16.
3. Verify actualonline checks/model lock/fullGPUexports/fullofficialCPUrows/failures. Do not finalize incompletepopulations.
4. From latest mainworktree (oldc66 lacks newentry), run combine_results --merged-root "$EVAL/navtest_merged" --lock "$EVAL/MODEL_LOCK.json" --subsets "$FINAL/navtest_subsets.json" --navtest --output NEWDIR. Produce onefull12146-row4modelCSV inclsubmetrics/failures/pairedppdifferences, plus dev. Failedrows zero/invalid, not filtered.
5. analyze_joint_plans on ALL/MASK full dev/Navtest exports, keeping actualDiT vs internalgraph ego distinct. Geometryproxies not realcollisionprobability/counterfactuals.
6. Final parameter/cost/peakmemory/throughput/timing report, fullcommands/runledger, updatedquickstart and final4statuses GRAPH_VALIDITY/LOCAL_COMPLETION/INTERACTION_DEPENDENCE/DEPLOYABLE_PLANNING. Paper88.9not equivalent to ourfinite7284scene training.
7. Restore authorized occupancy when safe, commit/push only taskbranch and verify remoteSHA. Never push rawdata/privatefigures/cache/weights. Markgoalcomplete only when planned work delivered or honestbudget-ended with explicit NOT_RUNs.

Live inspection command:
`python -c 'from pathlib import Path; print(Path("/mnt/project/local-interaction-mask-v2-artifacts/20260927/formal_public_epoch24/evaluation_v2/status.json").read_text())'`

CPU-only final reporting controller900304: report_final_locked.py, pinnedsource228c0b1 in /mnt/project/VLA-Drive-local-v2-runs-reporting. It waits for allthreebridges, builds final_reporting_v1/planner_learning_curves, then waits completeevaluation_v2 and runs full dev/Navtest combine_results plus ALL/MASK actualDiT-vs-internalgraph geometry. No training/modelselection. State final_reporting_v1/status.json; do not duplicate these reports. Latestgraphs27passes, nearest23. Architecture and20requirement evidence index now documented; finalscientificreports stillpending.
