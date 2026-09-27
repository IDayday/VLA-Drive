# Local interaction mask V2 — ACTIVE

Worktree /mnt/project/VLA-Drive-local-interaction-mask-v2-20260927, branch feature/local-interaction-mask-v2-20260927. Start227e781dd54f623b12b41bd7f2d2a7b302e0a014. Binding OBJECTIVE.md and USER_STEERING.md under reports/local_interaction_mask_v2. Preserve old workspaces/data/checkpoints, failed artifacts and immutable live run code. Latest completed source9a15df9; origin verified0937532 before current stage. No formal scientific conclusion yet.

## Binding protocol

Public-only Qwen/Qwen3-VL-2B-Instruct revision89644892e4d85e24eaac8bacfd4f463576704203, all public files byte verified. Fresh originalDiT/history/Reader/current heads plus common languageQ/V LoRA rank8alpha16 and driving embeddings. Public pretrained tensors/vision fixed. Historical private weights/caches are not formal model inputs. One training seed42. Current F0/L0/R0 only, original8x0.5s ego action encoding, one candidate,10FM steps, no scorer. Final endpoint complete12146scene/136log Navtest v1PDMS, no test tuning. Train7284/978logs, holdout64/59logs, dev1696/16logs; log-disjoint verified. Public pretraining exposure UNKNOWN. This subset pilot is not the published100k-scene/100k-update recipe.

## Active jobs and budget

Local0–7: public_foundation_continued_epoch8_to24_local8, supervisor862274, source0937532 at /mnt/project/VLA-Drive-local-v2-runs-foundation24. Parent immutable public_foundation_epoch8/checkpoint.pt at step1824. Continue same optimizer/RNG/scheduler/data/batch32 to16passes, optional24 per documented training+holdout curve rule. Max16GPUh for continuation, total48GPUh campaign. Do not launch duplicates. Original3972cab foundation completed8passes; its premature two-point stop decision is preserved, not called convergence. Current progress snapshot: {
  "step": 2699,
  "epoch": 11,
  "offset": 6112,
  "presentations": 86236
}

Remote training-vla-zt2 GPUs0–7: launcher698023 runs epoch8current feature extraction for7284train then64holdout, code0937532; outputs P1_public_epoch8_engineering/current_train and current_holdout. Max1.5+.2GPUh. This is the declared current-head optimization repair generalization check; not the main graph comparison. Probe head training not launched yet. Read ledger/status before following work.

Artifact root /mnt/project/local-interaction-mask-v2-artifacts/20260927. Latest ledger snapshot16.4456/48GPUh includes all failed jobs and loading. Phase reservations3audit+29foundation/features+4graphs+9planningNavtest+3diagnostics. Only authorized gpu_stress.py occupancy was paused; restore commands in resource_actions.json when no longer needed. Do not stop unrelated jobs. Both hosts8A80080GB. Measured samebatch32:8GPU1.719s/update vs16crosshost3.153s/update (Socket,noIB), so use hosts independently.

## Completed engineering evidence

51 related tests passed before current stage; new strict-head tests added. Full realQwen2GPU emptyannotationrank +resume; commonLoRA8GPU gradients; publicvision cache7348train/holdout+1696dev+12146test allcomplete,0roundtriperror. Navtestcurrent records v3complete0fail; badrootv2 retained. Original0stepBF16/FP32 failure and graph FD-exhaustion failure both preserved/charged and fixed.

P1 epoch2 fixed64training scenes: graph pause3→resume16 exact parameters/loss; CURRENT bridge difference at most1.2e-7 crossGPU (notbitexact); actual originalDiTgate0 error0.12mode×scene online/cache/visualbypass/targetpoison checks0; batch4vs1 max4.03e-6 within declared1e-5. ALL/MASK share one diagnostic graph here, not a research comparison. Batched16originalDiT export64scenes0fail. Crosshost foundation continuation456→458 restores optimizer/RNG/sampler; firstforwardexact, nextlossmaxdifference.000698 from BF16 crosshost arithmetic. No bitexact claim. New trained native/append-tail assertion implemented but notyetexecuted.

Official CPU v1PDMS wrapper smoke over4artificial stationary plans exactlymatches directfunction/subfactors/resume. Not learnedmodel scoring. No learnedmodel Navtest score yet.

P0 historical7284scene audit0fail32privatefigures:606028rawGT,54411supported/relevant,1200retainedmatches. P1 publicepoch2audit64/64,32privatefigures:25/407localnodes matched. Epoch8audit64/64,32figures:36/611nodes accepted by original fullslot matching,25/456raw supported/relevant proxies retained. Do not hide low coverage.

P1 fixed existingcurrent-head training on frozenepoch8features:500updates,64TRAINscenes, recall25.3%→79.0%, precision10.0%→64.3%. This is fitting-only evidence, not generalization/planning. Report P1_FIXED_CURRENT_HEAD_PROBE.json. Current generalization repair registered before launch:7284train, fixed64holdout, batch64,lr1e-3,8passes/16schedule; select best currentF1 including epoch0. If it improves, apply same recipe up to16passes after final foundation freeze; no epoch8head transplantation to later features. Same head for all controls. Selector/fullslot matcher unchanged; futurelabels erased. New strict override/cache/online integration is under test.

## Next concrete work

1. Finish/test/commit current-head refinement stage; source-pinned remote pilot after current_train/current_holdout complete. Verify actual save/resume and online/cache head parity; reject wrong-parent head. Record holdout gain/failure. No endless repair search.
2. Finish publicfoundation16/optional24, freeze shared parent/head. Extract actualfinaltrain/holdout/dev/Navtest currentfeatures; fullpopulation coverage audit and32real figures. Public-current provider and core graph code already implemented; no new BEV/scorer/encoder.
3. Main ALL/MASK graph16passes matched, extend BOTH to32 per worldholdout12→16 rule; nearest control. Sameinit/order/features/labels, single seed. Run completion/stationary/conditional/related-vs-weak removal diagnostics.
4. Equalcapacity CURRENT/ALL/MASK bridges, upstream includingDiT frozen,8→16matched according toholdout rule. Freeze comparison models, actualDiT vsgraph joint diagnostics, full1696dev and12146Navtest asynchronousGPU/officialCPU. Preserve allsubmetrics/failedrows/full denominator, pairedlog-cluster intervals. Testscores never select models.
5. Publish4required statuses, actualresults/limits, portable publicrecipe, commands/ledger/CSVs and stagecommits. Push only thisfeaturebranch, verifyremoteSHA, no weights/cache/privateimages. Markgoalcomplete only on actual completedwork or finitebudget-end delivery, not while mainwork remains.

Live status command:
`python -c 'from pathlib import Path; print(Path("/mnt/project/local-interaction-mask-v2-artifacts/20260927/public_foundation_continued_epoch8_to24_local8/progress.json").read_text())'`

Resume only after recorded supervisor exits: same immutable cwd and workercommand from that run's supervisor.json plus --resume, NEW supervisor runID and --initial-step equal savedstep; preserve --continue-from parent. Do not mutate live pinned code or duplicate activejobs.
