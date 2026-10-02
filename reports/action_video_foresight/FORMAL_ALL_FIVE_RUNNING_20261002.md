# Five formal experiments are training

Snapshot: 2026-10-02T18:00:22.521152+00:00. Immutable training source `1493deda247107efe076b9294863a6753391298a`; later reporting commits do not alter the running source.

All five use the full 101592-scene / 1176-log training manifest, current C1 DINO, W144, frozen vehicle MAE, global batch32,8 GPUs/model and a common100000-update schedule. Complete independent eight-frame DINO and genuine four-tubelet V-JEPA2.1 targets have passed source/local-cache completeness gates. No profile, startup or old100k checkpoint initializes a formal student.

| Arm | Host | Updates | Scene exposures | Future auxiliary | GT action in auxiliary | Planner |
|---|---|---:|---:|---|---|---|
| S0 | training-vlawm-zt2 | 246 | 7872 | 8-frame DINO | no | H_A |
| S1 | training-vla-zt2 | 467 | 14944 | 8-frame DINO | yes | H_A |
| S2 | training-vlawm-zt | 2422 | 77504 | V-JEPA2.1 video | no | H_A |
| S3 | training-rl-zt2 | 1589 | 50848 | V-JEPA2.1 video | yes | H_A |
| S4 | training-vla-zt | 2677 | 85664 | V-JEPA2.1 video | yes | H_A + W |

The actual shared initialization tensor hashes match across all five, including the random driving action head, state projection, driving tokens and W. Generic Qwen is the only pretrained student source. Every completed journal record has finite raw losses and a verified FP32 master update. S2/S3/S4 have actual full-weight update1000 observations; S0/S1 have not reached that node at this snapshot. The exact observation records include gradients, changed elements, task counts and effective coefficients.

Current coefficient1; future1.1549543539882532; interaction0.8328945981862067. Future/interaction warmup1000; LR warmup5000. Raw current, future, interaction and ego losses are recorded separately. The future target family differs between DINO and video, so raw future MSE must not be used to rank those families.

The first complete development evaluation is at5000 updates, then10000/25000/50000/75000/100000. Existing controllers save, export pure-current/head-stripped FP32-master predictions, canonically score the full1696-scene/16-log development set and continue training. New-method PDMS is NOT_YET_EVALUATED. Final Navtest, key second training seed and capacity/MAE controls remain pending; no new Navtest observer is started.

All five use their own eight-GPU host. Additional frozen lightweight readouts share authorized free memory. No unrelated compute was terminated. The DINO-cache migration only signalled individually verified owned cache workers; completed chunks and source1937b7d are preserved. Both native train caches now contain6350chunks/101592scenes: DINO539557651936bytes, video269779689568bytes.

Metered campaign job allocations total 107.8118GPUh at the snapshot; per-device interval union 102.6066GPUh avoids double counting shared jobs. This includes loading, failures, extraction, diagnosis, probes and training. A standalone two-process NCCL check has no recorded elapsed interval and is explicitly NOT_MEASURED; these totals are not claimed complete to arbitrary precision.

Current code self-check:28 targeted CPU tests passed. Earlier real four-loss/GT-isolation/direct-W/NCCL/resume evidence remains bound to its original immutable source. No formal source change or repeated historical audit is required.

Evidence: `FORMAL_PROGRESS_20261002_1800.json`, `FORMAL_FULL_WEIGHT_1000_SNAPSHOT_1800.json`, `FORMAL_EARLY_LEARNING_CURVES_20261002_1800.{json,svg}`. Exact controllers/commands: `FORMAL_CONTROLLER_LAUNCHES_V2.json`. These are actual early learning records, not proof of convergence or improvement.

Recovery after confirming a paused state (do not start a duplicate live controller):

```bash
cd /mnt/project/VLA-Drive-action-video-source-1493ded
/root/miniconda3/envs/ddp/bin/python -u -m tools.action_video_foresight.run_formal_campaign --plan /mnt/project/action-video-foresight-artifacts/20261002/formal_plan_seed42_v2.json --arm S4 --resume-controller --acknowledge-stop
```
