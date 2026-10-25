# Post-action evaluation and canonical candidate bank

This implementation is a training-only auxiliary objective for the complete Qwen-Flow method. It is not a deployed selector and does not reuse a trained external scoring model. The source starting point is `425eb5420ce1b592786f455ceea9bd6d124d2c54`; the final integration commit belongs in the campaign manifest. Results below distinguish executable implementation, automated checks, and actual simulator evidence. The imitation/representation/post-action stage explicitly has `uses_metric_supervision=true`, `uses_rl=false`; real PDM labels are supervised auxiliary targets, not policy reward gradients.

## Information and gradient contract

`PostActionHead` reads only the current actor trunk's eight hidden tokens for one exact labelled candidate. A LayerNorm/linear bridge, six randomly initialized metric queries, two width-512 decoder layers and a shared scalar projection (8,939,521 total trainable parameters) predict NC, DAC, TTC, EP, Comfort and DDC. Candidates are independent rows; the decoder has no candidate identity, separate scene encoder or cross-candidate attention. The six queries can exchange information within the same candidate.

`replay_post_action` normalizes detached physical FP32 coordinates with the original 8×3 ActionCodec, uses clean sigma=0, selects the already prepared differentiable context, and calls `actor.forward_features`. It does not call `actor.predict_x0`, the final action projection, an ODE solver or another Qwen forward. The auxiliary hidden state and selected context remain differentiable. Candidate coordinates, labels and pair indices are detached. Deployment does not call this helper or head.

The objective is factor BCE plus 0.25 times same-scene safety-boundary pair softplus, margin 1. NC=0.5 remains a valid BCE soft label and is never silently converted into a hard pair. Only opposite exact 0/1 NC or DAC labels form pair constraints. Missing labels have explicit validity. Nonfinite predictions fail even if their labels would otherwise be masked. Each objective averages its valid elements inside a scene, then its valid scenes. Distributed training sums denominators across ranks and accounts for DDP gradient averaging. Accumulation can supply full-update global scene denominators; such microbatch losses are summed without another division by accumulation.

The bank selects half of microbatch scene positions uniformly with a separately checkpointed CPU generator, then two real candidates each. Eligible nearby hard-boundary pairs have priority, with geometric proximity measured in physical coordinates; when no boundary pair exists, two distinct candidates still train factor BCE and their pair mask is false. For a one-scene microbatch ceil selects one scene, so two auxiliary trajectories are executed; the normal microbatch-four configuration executes four.

## Candidate and storage contract

`CandidateRecord` binds the scene, physical source log, TRAIN/DEV role, timestamp, observation fingerprint, exact physical FP32 8×3 trajectory and SHA256, six original canonical factors and masks, official total score, evaluator/context identities, actor/sampling/source provenance, coordinates, codec and schema. No trainable hidden states or post-head memory are cached. The exact stored FP32 coordinates are the official scoring request and later actor replay input. FP64 values that would silently round are rejected until explicitly converted and rescored.

Candidate sources are logged GT (actually scored), coherent interventions, own native IL/RL or current-student snapshots. External scoring weights and NAVTEST candidates are not accepted. Bad but physically representable candidates remain legitimate evaluation supervision; they never become the main imitation target.

The standalone builder uses `RewardService` and the existing canonical v1.1 tracking/simulation scorer. Each requested trajectory is scored against its own fixed official reference, so candidate-batch composition does not change progress normalization or labels. It never replaces this evaluator with a geometric approximation. Coherent time-warp, lateral and progression proposals are checked for finite coordinates, headings and generous motion bounds before real simulation. This prefilter makes no safety claim.

Immutable bounded JSONL shards have exact record offsets in a SQLite index. Reads do not deserialize an entire shard. The reader has bounded open-shard and decoded-scene LRUs, verifies the sealed manifest/index, hashes each shard once per reader lifetime and checks immutable file identity again on access, including decoded-scene hits. Record checksums are checked on actual access. There is no full-shard checksum on every training step.

A single-writer file lock, append/fsync-before-index transaction, per-scene commit and orphan-tail diagnostics make interrupted construction resumable. A global bank is COMPLETE only after all declared scenes have enough real scored candidates and all declared producer parts are complete. Merging parts combines indices without duplicating shard payloads. A manifest-written-before-COMPLETE interruption is idempotently recoverable. An identical completed build is reused; changed identities or conflicting labels fail. Sampling state binds the exact immutable bank hash and preserves selection RNG and consumption counters. New manifests are opened only at an explicit trainer boundary; resume refuses a different bank.

## Data isolation and reuse

The original segment-disjoint TRAIN/DEV manifest has fifteen overlapping physical logs. The campaign therefore preserves fixed DEV and NAVTEST and excludes their entire physical logs from TRAIN. The new fixed TRAIN contains 90,890 scenes, rather than 101,592. The builder and native-policy index independently reject a TRAIN manifest that violates this embargo. Physical source identity is the timestamp/vehicle prefix, not the segment-directory suffix.

Own completed native Qwen-Flow RL behavior receipts contain requested FP32 8×3 trajectories, real official factors, exact cache/protocol identities and sampling provenance. `audit_candidate_labels index-native` can materialize a bounded, resumable SQLite candidate index without loading model weights. It verifies the native run is complete, filters to the new legal TRAIN, preserves source hashes and explicitly distinguishes parameter-state identity from checkpoint-file identity. Imported real labels are reused only if protocol and the actual metric-context checksum match; otherwise those exact coordinates are rescored.

Older `safe_flow` banks that use normalized 8×4 actions or a different fixed candidate-pool score definition are intentionally not treated as compatible. Native NAVTEST behavior and predictions are not training inputs.

## Lessons from earlier scorer failures

The three user-specified historical reports were recovered from the original repository's Git objects because they are absent from the new native branch:

- `reports/scorer_pdms93/FINAL_SCORER_PDMS93_REPORT.md`, blob at commit `41a6f0c6d70321924d9934933172560db1192f2a`.
- `reports/m0_independent_scorer_representation/NO_VQA_E35_SCORER_CAMPAIGN_RESULTS.md`, commit `f9b5ed2f6168acdc054c497669bf66ba8245233f`.
- `reports/m0_independent_scorer_representation/LOWRES_DINO_NAVTEST_REPORT.md`, commit `3ebbf8ff51338405755ea1ca15f9ab324a217874`.

The old external DrivoR result was not an independently trained M0 scorer. Frozen-feature residual and private scoring heads repeatedly failed to convert large Oracle opportunities into held-out gains. The low-resolution frozen DINO addition did not improve the matched Base. These reports motivate a different gradient contract here: exact candidate evaluation shapes the shared actor trunk and current student context during complete-framework training. They do not prove that the new auxiliary objective will improve planning. No old >93 promotion gate is imposed on complete-method training.

## Automated evidence

Command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/project/flow-safe-improvement-20261013/venv/bin/python -m pytest tests/world_postcot/test_post_bank.py tests/world_postcot/test_deployment_evaluation.py tests/world_postcot/test_campaign_deadline.py -q`.

Latest combined result (post/bank, deployment/evaluation, controller/deadline): **42 passed in 8.66 seconds** on CPU, with one pre-existing pynvml deprecation warning. Coverage includes:

- exact soft-label/masked BCE and pair sign/boundary rules;
- candidate independence and reorder consistency;
- real small CleanXActor post-only autograd over three updates, including memory/input/trunk gradients, zero candidate/label gradients and no action-output call or gradient;
- an actual two-process CPU/Gloo DDP run with an all-invalid-label rank and equivalence to the global single-process objective;
- full-update versus uneven-valid-count microbatch gradient equivalence within documented FP32 tolerance;
- exact coordinates, source/role/key rejection, sampler restore, immutable shard corruption detection, disjoint-part merge, finalization crash recovery and bounded proposal generation including stationary scenes.

These are meaningful tensor/storage/distributed tests. They do not substitute for real images, official labels, full-size GPU training, real Qwen gradients or deployment latency. Those belong to the integrated campaign's actual receipts.

## Real simulator pilot

**Real official pilot COMPLETE, 2026-10-25 15:07:29 UTC.** A fixed hash-seeded 32-scene sample from the new embargoed TRAIN produced exactly 512 actual candidates. Eight local CPU producers were pinned to CPUs 112–119, each BLAS/OpenMP=1 with no visible GPU. All eight exited 0. Build wall including process/data startup was 24.922 seconds; producer scoring sections were 3.605–3.939 seconds each for four scenes/64 candidates. Total wall including merge and audit was 33.156 seconds, with 197.611 aggregate child CPU seconds.

The independent label audit performed 132 additional real official simulations on 64 candidates in four scenes. Cached/recomputed, batch/single and reversed-order comparisons all had maximum absolute difference **0.0**, tolerance 1e-12. NC/DAC/TTC labels covered both 0 and 1, and EP varied continuously. Comfort/DDC failure values and NC=0.5 were not observed by this audit and remain explicitly outside its real coverage. The soft NC=0.5 handling has separate unit evidence.

Pilot artifacts: `/mnt/project/qwen-world-postcot-20261025/candidates/pilot32`, bank hash `6fabe6644129180f7fa4343b63220f3ef2c0af8f7edbed0bc14c7d2da48e4815`; actual audit `pilot32_LABEL_AUDIT.json`; PID/start-time/commands/costs `pilot32_control/RECEIPT.json`. Immutable pilot source hash `d377295d44edb13dc3a02dce0fa9683357e998f54c518e3a72066fb4997898eb`.

**Representative performance bank COMPLETE, 2026-10-25 15:10:28 UTC.** This uses the actual first 1,024 scenes in the new full-TRAIN seed-42 sampler, not a repeated pilot. It contains 16,372 truly scored candidates, at most 16/scene. One scene had fewer independent kinematically admissible proposals; it was retained with its real candidates rather than filled with fabricated duplicates. All scenes have at least two. Eight producers exited 0. Build wall was 52.753 seconds and total including merge/audit was 60.691 seconds, with 373.068 child CPU seconds. A separate 132-simulation audit again had maximum absolute difference 0.0. Bank `/mnt/project/qwen-world-postcot-20261025/candidates/performance1024`, hash `01bb6a0b3783d4517fd50d92d89c7a4c15b2b355a8f02dd525ffde9cfdcd8923`.

These banks qualify real labels and engineering performance inputs. They do not establish planning improvements. The full-bank result below comes from independent all-scene production receipts; it is not inferred from these subset manifests.

## Executable commands

Bounded pilot (one of eight externally scheduled producers; the 64-attempt per-producer limit gives 512 total):

```bash
python -m tools.qwen_world_postcot.build_candidate_bank \
  --data-root /mnt/project/qwen-world-postcot-20261025/data/TRAIN \
  --output /mnt/project/qwen-world-postcot-20261025/candidates/pilot32 \
  --max-scenes 32 --target-candidates 16 \
  --rank 0 --world 8 --max-simulation-budget 64
```

Finalize only after all eight parts are complete:

```bash
python -m tools.qwen_world_postcot.build_candidate_bank \
  --output /mnt/project/qwen-world-postcot-20261025/candidates/pilot32 \
  --world 8 --finalize
```

Official fixed-reference audit, maximum 132 additional simulations:

```bash
python -m tools.qwen_world_postcot.audit_candidate_labels audit \
  --bank /mnt/project/qwen-world-postcot-20261025/candidates/pilot32 \
  --output /mnt/project/qwen-world-postcot-20261025/candidates/pilot32_LABEL_AUDIT.json \
  --max-scenes 4 --max-candidates 16 --max-simulation-budget 132
```

Full TRAIN construction is the same immutable interface with `--all-scenes`, the new 90,890-scene manifest, explicit per-producer simulation budget and a separate output identity. It must be admitted by the main scheduler after the actual pilot and may incorporate a verified own-native policy index with `--extra-candidates-index`. It is now COMPLETE as documented below. `--resume` preserves exact existing proposal requests, scored records and identities; it does not silently enlarge the declared population or budget.

## Production launch and limitations

The full production controller was admitted on local `training-vla-zt`, PID 3864495, with sixteen single-threaded CPU producers at niceness 5 and no GPU. Receipt: `/mnt/project/qwen-world-postcot-20261025/candidates/full_production_control/RECEIPT.json`. Source `/mnt/project/qwen-world-postcot-20261025/snapshots/post_production_v3`, SHA `5f1e56b4b5aa05fc055d605d1936bf4732066de2f8a92f9a7def0b633c5c5230`. It first indexes both own completed 50k canonical/source RL lineages (200,000 declared behavior artifacts), merges verified disjoint shards, then builds the complete fixed 90,890-scene initial bank. Construction prioritizes the performance window but leaves the actual trainer sampler unchanged. The declared fresh-simulation cap is 1,454,240, reduced by exact prior-label reuse. Full-bank output is `/mnt/project/qwen-world-postcot-20261025/candidates/full_train_initial_v1`.

Before production, the original-trajectory prefilter was corrected: only newly constructed interventions are rejected by the generous motion filter. Logged GT and real own-policy trajectories remain real scoreable behaviors even when the diagnostic flags them, so difficult scenes cannot vanish from coverage. Rough logged paths can receive additional gentle observed-forward-velocity braking/curvature alternatives, all independently officially simulated; these are not presumed safe or high quality. Their explicit source parameters record the simplified lateral-velocity assumption.

`--reuse-bank` accepts only sealed banks of the same data/observation/coordinate/metric identity. A stored request is reused only after its actual current metric-context checksum, full observation hash and physical FP32 trajectory hash match. Different requests or contexts are resimulated. This prevents qualification/performance windows from creating duplicate official labelling work in the full build. Native behavior indices preserve real original labels under the same checks.

Full production is **COMPLETE**. All sixteen producer processes exited 0. The complete immutable bank contains exactly **90,890 scenes × 16 candidates = 1,454,240 records**, spread across 64 bounded shards with **3,320,198,119 payload bytes**. Bank hash: `356d24a0e8e9e6095a9dd1f8d2d96fe39394ca51408124f6d51b62cd99f95b7f`.

Actual source counts are 90,890 logged-pose candidates, 121,368 own-native RL candidates, and 1,241,982 coherent variants. Native indexing read 200,000 completed behavior artifacts, retained 1,424,664 unique legal requests from 15,171 scenes, and excluded 21,910 embargoed source artifacts. The builder performed 1,317,436 new official simulations and reused 136,804 exactly verified requests, including 121,368 own-native requests and 15,436 earlier pilot/performance requests. Native indexing took 508.835 seconds, its merge 362.402 seconds, complete-bank building 1,453.965 seconds and bank finalize 66.174 seconds. Aggregate child CPU consumption was 19,171.928 CPU seconds. No GPU was used.

The full bank has 39,634 scenes with a real NC/DAC hard comparison pair and 5,372 official NC=0.5 labels. A further 264 official simulation requests audited actual bank candidates; all exact request/label, batch/single and ordering checks had maximum absolute difference 0.0. The deep coverage audit verifies **all 1,454,240 index offsets, record identities/checksums, FP32 coordinate hashes, canonical factor aggregation, immutable shard checksums and full declared coverage**, with zero new simulations. Final evidence: `candidates/full_train_initial_v1_COVERAGE_AND_TIMING_PROVENANCE_v2.json`, SHA256 `644c76a09aede7b91d98bd8729a62284374be38d7f7a1e1f8a0e78e1e96ba199`. Its actual audit core used eight CPU workers for 14.747 seconds. The launcher did not retain that auditor process's wait status, so its receipt reports returncode null rather than inventing exit 0; the sealed output itself is PASS and its process was independently observed exited.

Raw TRAIN contains 625 logged futures with a timestamp gap. Their stored eight physical poses remain legal fixed-0.5-second canonical scoring requests, but they are **not certified correctly timed imitation GT**. The deep audit publishes a record-ID/hash-bound provenance correction overlay with actual raw time deltas and `logged_pose_candidate_time_mismatch`. Coordinates and official labels were not modified or rescored. New builder versions record this directly in source parameters. Main imitation uses the separate, actual-time valid-waypoint mask implemented by the trainer; candidate replay is not imitation.

## Full-model deployment and lifecycle evidence

The actual complete two-update qualification checkpoint `qualification/zero2_stage02/checkpoints/update000002` was strictly exported to `audits/full_export_parity_stage02/deployment`, model SHA256 `8a75fb22c704e8bf939eb190dbd7f80304152968ef122d3cac1f316ce6730e72`. Two separate real processes loaded the full training model and standalone export, respectively. With two actual current TRAIN observations, all three cameras and identical fixed FP32 noise, normalized and physical trajectories were **bitwise equal, maximum absolute error 0.0** at a tolerance fixed to 0 before execution. Export removes 36,852,497 train-only parameters while retaining predictive memory. Each process performed exactly two visual and two language calls for two scenes. Runtime guards observed no teacher, future/GT, metric/simulator or training-cache access. Receipt: `audits/full_export_parity_stage02_attempt2/PARITY.json`.

This test used A800 80GB GPU1 on tsinghua-zt5, sharing only with an admitted small owned teacher producer. Its 81.9/82.9-second process walls include model construction and loading and are **not** deployment latency measurements. Root runs latency independently on an exclusive device. These very early qualification weights prove engineering wiring, not planning quality. The first parity attempt's PYTHONPATH launch failure is retained rather than erased.

Strict RL export keeps real native LoRA wrappers, verifies the entire adapter key closure, exact FP32 tensor shapes/dtypes and frozen IL parent, and never silently merges BF16 adapters. New-process CPU actor tests verify the same K10 outputs. Formal DEV/NAVTEST summaries require every sealed fixed scene, original six factors and total, exact submitted trajectory, source log and official per-scene context checksum. Invalid or failed scenes never become valid zeros or disappear from the denominator.

## Periodic current-policy refresh and orchestration

The executable refresh pipeline generates two independent-noise K10 current-student trajectories per TRAIN scene, batched four current scenes per single Qwen execution. Fourteen unique historical candidates are retained with verified labels, prioritizing logged poses and own original policies without sorting by scores. At most sixteen deployment-independent auxiliary candidates remain per scene. The source checkpoint's **actually consumed** bank path/hash is used, not whichever registry is latest. Only full 90,890-scene immutable completion may publish a versioned bank. Formal sources are predeclared IL 10k,20k,…,90k, at most nine. A 5k trainer boundary pins a registry decision, saves the exact bank and sampler state, and preserves that decision through resume. A partial diagnostic or qualification bank is explicitly rejected by the formal registry.

The actual bounded 32-scene refresh qualification uses our own real 32-update full-model checkpoint and a maximum 64 new official requests. Its first run failed before candidate generation because a model constructor normalized the caller's sealed Qwen config in place. New constructors and export/load code defensively copy configuration and preserve the original source identity; old files are retained. A subsequent dispatch was rejected before child startup because the shared GPU admission lease had another consumer; no lock was bypassed. The admitted retry on permanent tsinghua-zt5 GPU3 completed with exit 0: **32 real scenes, 64 new own-policy trajectories, 64 real canonical simulations, 448 historical records preserved field-for-field, 512 total records**. Visual, language and VLM hooks each counted exactly eight calls for eight four-scene batches. Generation plus scoring took 56.349 seconds, excluding model/data loading. Every new coordinate/proposal/actual-feedback/source-checkpoint join was audited, with no extra simulation requests. Diagnostic bank hash `a34a127dde632efaa878ff52a9a36de8058830f20f987c9472d4d8228f5e0eaf`; actual receipt `audits/refresh32/REFRESH_QUALIFICATION_AUDIT.json`. An actual attempted formal-registry publication correctly failed and left the registry unchanged. Worker and child exit were independently verified; only the pre-existing teacher remained on that GPU. This is an **engineering PASS**, not scientific evidence from 32-update weights. No 10k formal refresh is claimed before its source checkpoint exists.

The campaign controller waits for actual engineering acceptance and complete correctly resolved teacher coverage, then schedules 100k IL with every-5k full DEV, freezes the highest-DEV IL checkpoint, verifies actual-parent real-PDM RL in a short qualification, and starts a distinct fresh formal 50k RL lineage. Qualification updates never initialize formal RL. The same selected models undergo final NAVTEST at seed2101 only after DEV locking; predeclared additional inference seeds2102/2103 are reported separately without changing checkpoints. There is no auxiliary-PDMS gain gate before complete-framework training.

Permanent idle eight-GPU resources are preferred. Temporary-host jobs receive cooperative checkpoint requests twenty minutes before their immutable authorization expires. Explicit owned-PID/starttime/process-group checks cover torchrun's separate rank sessions. Extreme hung jobs are terminated at deadline−90s and killed at deadline−30s only if ownership still matches. Such forced expiry is recorded `FORCED_EXIT_REVIEW_REQUIRED`, with the last sealed checkpoint; it is never called lossless/exact. Unknown startup/exit states never trigger duplicate training. A positively identified resource refusal before child startup has an explicit receipt and bounded cooldown; real training failures require review.

## Fixed effective native references

`campaign/BASELINE_BINDING.json` was pinned before new-method NAVTEST results. It chooses the **highest actual complete optimizer step**, never the highest observed score, within the original own native IL and canonical-RL lineages. Using the original native integrity verifier plus exact official feedback linkage, all 12,146 real scene rows and 44 physical logs were rechecked:

- Native IL77k: PDMS 88.1449451453.
- Native canonical RL50k: PDMS 90.5131641027, initialized from its own IL53k.

The new adapter permits explicitly named architecture, velocity/clean-x, precision, training-population and budget differences between methods. It still requires seed2101, single-candidate K10shift5, the same canonical evaluator and exact scene/context hashes; it is not a generic integrity bypass. Original native training used 101,592 scenes and included the fifteen DEV physical logs now embargoed from the new 90,890-scene TRAIN, which limits causal attribution. Subtracting IL77k from RL50k is not that old lineage's RL gain. New primary results will report paired physical-log-bootstrap comparisons to both references; extra sampling-seed statistics describe only the new method. No new baseline NAVTEST run was launched.

These completed storage, official-label, export and lifecycle checks do not establish that the complete new algorithm improves planning. Formal 100k IL, 50k RL, full DEV and final NAVTEST results remain pending actual campaign execution and are reported by the main campaign state.
