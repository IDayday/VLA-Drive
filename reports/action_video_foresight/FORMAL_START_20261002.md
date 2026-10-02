# Formal action-conditioned video training is running

Snapshot: 2026-10-02 17:04:53 UTC. Training source `1493deda247107efe076b9294863a6753391298a` is frozen. Subsequent reports and diagnostic source commits do not change that source.

| Arm | Target / auxiliary action / planner | Authorized host | Actual updates | Scene exposures | State |
|---|---|---|---:|---:|---|
| S0 | 8-frame DINO / absent / H_A | training-vlawm-zt2 | 0 | 0 | Waiting for complete local sequence targets |
| S1 | 8-frame DINO / GT ego / H_A | training-vla-zt2 | 0 | 0 | Waiting for complete local sequence targets |
| S2 | Native video / absent / H_A | training-vlawm-zt | 697 | 22304 | Real formal training |
| S3 | Native video / GT ego / H_A | training-rl-zt2 | 0 | 0 | Waiting for verified local full-cache staging |
| S4 | Native video / GT ego / H_A+W | training-vla-zt | 912 | 29184 | Real formal training |

All five host-local controllers are deployed. The first waiting-only plan was superseded before any optimizer update after a self-review found an ego-summary completeness issue in the evaluator. Its paused zero-update receipts remain intact. The new plan binds the corrected evaluator and all scientific identities. The source is not modified in any running checkout.

The common plan is 100000 optimizer updates, global batch32, microbatch4, eight GPUs/model, seed42, scheduler horizon100000 and LR warmup5000. The same 101592 train scenes/1176 logs and 1696 development scenes/16 disjoint logs are used. The existing frozen vehicle-MAE teacher and current C1 DINO targets are unchanged. Current/future/interaction weights are 1 / 1.1549543539882532 / 0.8328945981862067; future and interaction warm up for1000 updates, current does not.

The full native video training cache is COMPLETE: 6350 chunks,101592 scenes,269779689568 bytes. The independent eight-frame DINO cache is still extracting; no formal partial-cache bypass is used. Local replicas are staged only for the target type needed by the respective host. The unrelated ReCogDrive exports on training-vlawm-zt2 are not signalled; their remaining use of a GPU will block S0's device-ready check.

S2/S4 have identical actual initial tensors for every shared action tensor, state projection, driving token and W parameter, and the same generic source manifest. Their initialization summary states driving_weights_loaded=false and future_teacher_in_model=false. W initial hash: b68950f93392c509d31addc466cf27de07d3f8c15ed3f9c1603fdcdc88d6a8dd. Startup/profile/frozen100k weights do not initialize these formal students.

Actual observations at1/100/500 include all four finite raw and weighted losses, per-rank counts, FP32-master update checks and nonzero actual updates for W, Qwen, the original action decoder and future head. S4's isolated GT auxiliary action encoder also updates. These are warmup observations, not convergence evidence. The1000/2000 nodes are recorded automatically in the unchanged training source.

At this snapshot, job allocations total76.2429 GPU-hours; the union of declared host/device intervals is72.8131 GPU-hours because some authorized jobs share devices. Formal allocations account for7.4905 GPU-hours. Loading, failed attempts, extraction, probes and scoring are recorded separately. Idle pressure and unrelated tasks are excluded; no resource cap is invented. Detailed kind-level costs and scope limitations are in the machine-readable snapshot.

Old C1@100k's fixed128-development-scene W intervention now has canonical PDMS: base88.1752, after-layer9 permutation86.4875, after-layer18 permutation88.1415. Layer9 paired change is-1.6878 points,16-log95% interval[-4.4601,+0.5861]; layer18 change-0.0337,[-0.1002,+0.0295]. All128 scenes are retained,0 failures. The intervals include0 and the intervention can be out of distribution. This is a sensitivity diagnostic, not a new model gain or causal proof.

New main-method development PDMS is NOT_YET_EVALUATED. Fixed full-development milestones are5000/10000/25000/50000/75000/100000, using current-only deployment with auxiliary heads removed, FP32 master loading, TF32 off, original10 Euler steps and one candidate. No new milestone Navtest observer is launched. Final locked Navtest, key second seed and registered capacity/MAE controls remain pending.

Executed code self-checks include16 tests in tests/action_video_foresight on the current report/evaluator revision. Earlier core validation includes actual four-loss real data, direct final-W gradient, strict video weight loading, real NCCL normalization and continuous4 versus2+2 exact resume on the stated fixed environment; their original evidence/source remains preserved.

Do not launch a duplicate controller. To recover a safely paused S4 controller from the same immutable plan after checking state:

```bash
cd /mnt/project/VLA-Drive-action-video-source-1493ded
/root/miniconda3/envs/ddp/bin/python -u -m tools.action_video_foresight.run_formal_campaign --plan /mnt/project/action-video-foresight-artifacts/20261002/formal_plan_seed42_v2.json --arm S4 --resume-controller --acknowledge-stop
```

Evidence: FORMAL_PLAN_SEED42_V2.json, FORMAL_CONTROLLER_LAUNCHES_V2.json, FORMAL_PROGRESS_20261002_1710.json and C1_W_CANONICAL_PDMS_DIAGNOSTIC.json. Weights, scene images, targets and full private per-scene records stay outside git.
