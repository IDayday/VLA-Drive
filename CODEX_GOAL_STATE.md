# Local interaction mask V2 — PAUSED_BY_USER

User instruction, 2026-09-27: pause development, organize design/ideas/results, push promptly, then stop experiments. Do not restart training, evaluation, reporting controllers or a new recipe. The scientific objective is unfinished; user pause is not completion or a resource block.

Branch: feature/local-interaction-mask-v2-20260927.
Worktree: /mnt/project/VLA-Drive-local-interaction-mask-v2-20260927.
Start SHA: 227e781dd54f623b12b41bd7f2d2a7b302e0a014.
Last verified remote before this handoff: 025a2a7c94afafb052c8b21b9aeef35b7a52b4fb.
Current handoff SHA: obtain with `git rev-parse HEAD`, verify against origin. Original dirty /mnt/project/DriveVLA-M0 remains untouched.

Authoritative entry: [DESIGN_REVIEW_AND_HANDOFF.md](reports/local_interaction_mask_v2/DESIGN_REVIEW_AND_HANDOFF.md), [PAUSE_STATE.json](reports/local_interaction_mask_v2/PAUSE_STATE.json), [PAUSE_RESUME.md](reports/local_interaction_mask_v2/PAUSE_RESUME.md). These supersede historical running/queued text. OBJECTIVE.md and USER_STEERING.md retain the research protocol, subject to this pause.

Artifact root: /mnt/project/local-interaction-mask-v2-artifacts/20260927; final stage formal_public_epoch24.
Budget: 38.45416863918297 / 48 GPU-hours used; 9.545831360817033 remaining. No running ledger jobs. PAUSE_RUN_LEDGER.json preserves every charge including failed work.

Completed: public-Qwen foundation24passes/5472updates, current-head refinement16passes/1824updates, all21190refined current features/0fail, ALL/MASK/NEAREST graphs32passes/7296updates each, full7284scene graph and supervision audits/0fail,32private visualizations, common-track and relation diagnostics, graph curves. Real engineering and multiGPU checks are indexed in ENGINEERING_EVIDENCE.md. No private driving foundation is used for formal runs. Public Qwen revision89644892e4d85e24eaac8bacfd4f463576704203; foundation checksumac7cfadba298e2980a5c278183ca4a93924e1792fe7a6e4342bb730fafa8a0f4. Current head checksum0c002ffcb58d76bcce4983048ad635068b467988244b34447844f2316dd5c6ae.

CURRENT/ALL/MASK all completed8passes/1824updates. Actual DiT holdoutADE6.688454/6.645932/6.708071m,64scenes/0fail per arm; these are not Navtest or PDMS. Three runs jointly continued63updates, then all saved step1887, epoch8, offset2016, presentations60288 with optimizer/scheduler/RNG/seen tokens. CPU deserialization, keys, progress and ledger agreement verified. Paused checkpoints are NOT the weights corresponding to epoch8 metrics; no new inference was run.

Training source: 955097588152912eeaca72eeb4ee9f736e40b530 in immutable /mnt/project/VLA-Drive-local-v2-runs-main. STOP_REQUESTED files remain. Three supervisors and waiting controllers899209/899233/900708 exited; states are paused_by_user. Remote graph diagnostics completed/exited. No automatic experiment queue remains.

Paused checkpoint SHA256:
- P_CURRENT: e4f6f521fa23d4066eaad092f5f7a92ee24f007feec35600de66781d263411e0
- P_LOCAL_ALL: 4fa85a451dd7b1fd0b0d338d6a7232bdb4cb45367fdd3bcdceb5164e0911ffdb
- P_LOCAL_MASK: 052c430811145564aefea76c3d0e3ff98acb4d7aff4e54c37b90397338755646

Only original authorized occupancy restored: local gpu_stress GPU4–7, vla-zt2 gpu_stress GPU0–7. LocalGPU0–3 free of campaign work. PAUSE_RESOURCE_RESTORE.json records ownership; occupancy is not an experiment or campaign budget charge.

Unfinished: actual-trained3planner online parity; full16pass planning transfer; final model lock; formal learned-model dev/NavtestPDMS; EPDMS/Hard; full4model benchmarkCSV; final planner geometry; freshcleaninstall. Prepared Navtest current features and official metric caches are not model scores. Final acceptance remains full12146scene/136log Navtest, seed42, one candidate/10steps, no test tuning.

Key correction: NAVSIM has GT boxes and future tracks.7240/7285accepted associations have futures;33191active predicted nodes were unassigned and21418assignments rejected. Shared foundation trained egoFM+currentboxes with motionhead frozen, then strict predicted graph association limited motion supervision. This differs from WCog semantic motion training. Proposed corrections are documented only, not implemented or run. Existing results are finite and do not establish convergence or planning benefit.

engineering_status=PARTIAL; research_status=INCONCLUSIVE.
GRAPH_VALIDITY=PARTIAL; LOCAL_COMPLETION=INCONCLUSIVE; INTERACTION_DEPENDENCE=INCONCLUSIVE; DEPLOYABLE_PLANNING=NOT_RUN.

Safe next command (read-only):

```bash
cat /mnt/project/local-interaction-mask-v2-artifacts/20260927/formal_public_epoch24/user_pause_verified.json
```

Only after the user explicitly resumes: inspect real processes/resources/ledger, follow PAUSE_RESUME.md for the old recipe or register a separate new design. Never repeat completed large experiments or automatically execute earlier active-state instructions.
