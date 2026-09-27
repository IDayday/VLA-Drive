# Local interaction mask V2 — ACTIVE

Branch feature/local-interaction-mask-v2-20260927 in /mnt/project/VLA-Drive-local-interaction-mask-v2-20260927; start 227e781dd54f623b12b41bd7f2d2a7b302e0a014. Binding objective: reports/local_interaction_mask_v2/OBJECTIVE.md and USER_STEERING.md. Preserve old workspaces, data and failed attempts. No final scientific conclusion yet.

## Binding protocol

Independent public-origin algorithm: official Qwen/Qwen3-VL-2B-Instruct revision 89644892e4d85e24eaac8bacfd4f463576704203; all public files byte verified. Fresh original DiT/history/Reader/current heads. Common rank8 language Q/V LoRA and driving tokens trained for every control; original public Qwen tensors and vision fixed. Old private VLA/current caches are historical audits ONLY. Single training seed42 (second seed canceled). Current F0/L0/R0, original8x0.5s action normalization, one candidate,10 FM steps, no scorer. Final planning endpoint complete12146-scene/136-log Navtest v1 PDMS; no Navtest tuning. Public base pretraining exposure UNKNOWN. Batch/epochs adjustable using measured cost and training-domain convergence; matched paired conditions. Independent48 GPU-hour cap remains.

## Running jobs (do not duplicate or mutate their pinned code)

Local8 GPUs: public_foundation_lora8_seed42_local8, supervisor849099, source3972cabe8470a37b6bf9b8401204329e5622d292 in /mnt/project/VLA-Drive-local-v2-runs-3972cab. Artifact root /mnt/project/local-interaction-mask-v2-artifacts/20260927. Train7284/978logs, holdout64/59logs, globalbatch32, seed42, initial8 epochs with preregistered6-to8 holdout curve gate allowing16; max17 GPU-hours. Latest2 completed passes: holdout egoADE11.03271→7.56021→6.61106m; class-filtered recall.01181→.14046→.16656. These are early learning diagnostics, not PDMS. Rolling checkpoint includes all new/shared adapters, optimizer/scheduler/per-rank RNG/offset. Read progress.json and ledger for live status.

Remote training-vla-zt2: frozen public current-vision extraction complete:7348 train+holdout,1696 dev,12146 Navtest; all eight shards per dataset exact disk-roundtrip error0. No future labels read. Remote GPUs now available for current-cache/graph engineering jobs. Only authorized gpu_stress.py occupancy was paused, commands in artifact resource_actions.json. Do not stop unrelated jobs.

Ledger snapshot 5.9503/48 GPU-hours, including all failed jobs/loading/features. This is not live; read budget_ledger.json. Phase reservations: audit3, foundation+features20, graph8, planning+fullNavtest14, diagnostics3. Supervisor enforces actual GPU-time and graceful checkpoint on cap. Both hosts have8 A80080GB. Measured8-GPU batch32 step1.719s versus16-GPU crosshost3.153s; NCCL Socket and no /dev/infiniband. Use hosts for independent jobs; no speculative16-GPU speed claim.

## Completed evidence

Current-only geometry/selection A/B/C/D, limited rooted relation/twohop graph, distinct masks, source-stable noise, fullslot matching before local extraction, conditional clamping, NaN isolation, cache identity, publicfoundation DDP and resume implemented.48 related tests pass. Real full-Qwen2GPU empty-annotation rank +checkpoint resume passed; real8GPU commonLoRA/token training gradients and holdout passed. Failed initial foundation job (0steps, BF16 noise to FP32 DiT) preserved and fixed in sourcef03b2f0, included in ledger.

P0 historical7284/7284 audit,0 failures and32 real visualizations at P0_historical_audit_3972cab; full raw606028targets, supported+relevant54411, matched selected supported+relevant1200. This low historical association coverage is a limitation, not formal public-origin evidence. No private images uploaded. Publictrained perception must be audited independently before scientific claims.

Data split actual manifests verified:7284train/978logs,64holdout/59logs,1696dev/16logs, pairwise no log overlap. Navtest fixed official12146/136 set verified; current-only records navtest_current_v3 complete0fail (v2 bad sensor-root attempt retained). Metric-cache scoring contents still need smoke verification. No new Navtest planning scores yet.

## Next work

1. Commit/push reviewed current-cache +matched graph training pipeline and stage reports. Create immutable hardlink of completed public-foundation epoch checkpoint for explicitly labeled P1 engineering check on remote; do not use partial-trained check as main result. Verify strict restore, actual currentcache parity, graph update/resume and32 publicgraph views.
2. Finish paired graph schedule controller and same-capacity CURRENT_MEMORY/ALL/MASK planner bridge. Fix/test FP32 DiT inference boundary, gate0 and online/cache equivalence. Register choices before runs. Publicfoundation remains running unchanged.
3. When publicfoundation is frozen, extract shared current-only caches, audit matching/coverage, run core single-seed G_LOCAL_ALL/G_LOCAL_MASK with matched16/32 schedule and nearest control. Run conditional/relationship diagnostics and planned frozenDiT bridge controls.
4. Freeze comparison models, full1696 development diagnostics and full12146 Navtest PDMS with bounded asynchronous GPU export/CPU official scoring, all scene CSV/submetrics/failures and log-cluster uncertainty. No score-based test selection. Publish final four statuses and recovery command only on actual evidence or budget exhaustion.

Run observation command:
`python -c 'import json; print(json.load(open("/mnt/project/local-interaction-mask-v2-artifacts/20260927/public_foundation_lora8_seed42_local8/progress.json")))'`

Resume requires same immutable run code and full recorded command with --resume, a NEW supervisor run ID and --initial-step from checkpoint, only after original supervisor has exited. Do not launch duplicate training. Latest valid committed source before this state update: dcc22911bf73f3885074844af48466bb3e2077fd. Live training pinned3972cab irrespective of later repository commits.

## Updated stage after source6149cd5

Stage commits70296db,815f7c2,5c874d4,6149cd5 pushed to origin new branch (never merged).50related tests passed, plus new statistics test. RemoteP1 cached graph and CURRENT_MEMORY bridge paused3→resumed16updates on64real training scenes; graph exact parameter/loss equality, bridge cross-GPU max1.2e-7 numerical difference. Full original DiT gate0 error0 on64scenes.12online mode×scene checks: cached/uncachedcurrentvision/targetpoison/gate0 all0; batch4vs1 max4.03e-6 within1e-5 declared tolerance. Public epoch2graph audit64/64 and32privatefigures done; 25matched local neighbors out of407selected,16supported+relevant retained out of456raw proxy targets. This is undertrained engineering evidence only. Do not claim planning benefit.

New train_planner, export_plans (boundedbatch16, atomicNPZ), score_async (boundedCPU workers, officialfull-cache PDMS, resumable rows/allfailures retained), check_online, regraph, convergence and Quickstart implemented. FormalP2/P3/Navtest remains NOT_RUN while foundation trains. Four artificial stationary-plan scoring smoke cases exactlymatch direct officialfunction and resumable output; not learned-model Navtest scoring.

Revised preregistered phase reservations3audit+29publicfoundation/features+4graph+9planningNavtest+3diagnostics=48. Initialmain3972cab jobunchanged. At16compare14/16trainingholdout relativechange>2percent; if triggered and9GPUh remains reserved after extra stage, one explicit continuation to24totalpasses(max9GPUh) is allowed. Original16epochcosine scheduler staysatfloor; no LRsearch.24isceiling,notconvergenceproof. Source6149cd5 has --continue-from with fulloptimizer/RNG/offset and parenthash plus computationalfile parity. Actualremote8GPU continuation456→458tested:firstforwardmatches exactly, BF16backward/crosshost step458loss differsmax0.000698; no bit-exact full-Qwen training guarantee. Do not misreport sampler/RNG recovery asbitwise arithmetic.

Remote P1_batched_export_remote2 source6149cd5 may stillrun; outputP1_public_epoch2_engineering/batched_export (A0/CURRENT_MEMORY64training scenes, notNavtest). Continuationcheck andonlineparity finished. No foundation continuation/main graph run has beenlaunched. Preserveactive original local8GPUfoundation.

Latest progress snapshot: {
  "step": 1668,
  "epoch": 7,
  "offset": 2304,
  "presentations": 53292
}; ledger 11.1515/48GPUh. Readlivefilesbefore nextaction.
