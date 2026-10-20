# Action-effect repair: original S0 and reusable-asset audit

This is a read-only source and metadata audit on 2026-10-20. The new campaign has **not yet replayed the model, generated interventions, simulated, or trained** in this subtask. The historical real-image receipt is evidence to compare against, not a new campaign PASS. The Lead owns resource admission and the new replay.

The machine-readable inventory, exact small-file SHA256 pins and native file sizes are in [BASELINE_AUDIT.json](BASELINE_AUDIT.json); reusable paths and rejected lookalikes are in [BASELINE_ASSET_INDEX.json](BASELINE_ASSET_INDEX.json). No model weights, images, metric payloads or resource process inventories are committed.

## Bound model and workspace protection

- Audit start: `36493208ba4d8c25bedfcc43935a4ce4527131cc` in `IDayday/VLA-Drive`.
- Isolated branch: `audit/action-effect-baseline-20261020`; worktree `/mnt/project/VLA-Drive-action-effect-baseline-audit-20261020`.
- Original `/mnt/project/DriveVLA-M0` was already dirty at `6e96cf7321b134c42c2cf0fbbc315cd61c925b11`; its changes were not modified, stashed or reset.
- Applicable `/mnt/project/AGENTS.md` and repository `AGENTS.md` were read. This new action-effect work does not restart held ST feature, IQE expert/Scorer, MOPD or other old experiments. Only the Lead may dispatch experiments. Existing genuine jobs and native checkpoints remain protected.
- The sole Base is `/mnt/project/action-video-foresight-artifacts/20261002/students/formal_S0_seed42_full100k_v2/checkpoints/milestone_100000`, completed **100000**, run identity `8dd7d914f10559b0fbfc9593ee1f480a8bb2da80cccb53cba0d684a468e209cd`.
- Existing binding `/mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json` has content identity `3b910a6d4245580d77f868e4e0af5faf2fc7fccc25c561c935f219ab1bf7f099`; checkpoint inventory identity is `78f870647015194798af067eefc8d8fedb6e59e6db2f3ae719c63b783871446f`. The inventory identity is a digest of the exact checkpoint receipt, **not a single weight-file SHA**.
- Actual `identity.json` and native `COMPLETE.json` bytes were rehashed and match the binding: respectively `184ceb918be467eecf601dee96972622cf8f93867e9b453890b855df0a6e4a54` and `ee317d181734e0d750a7bb28be82d9ccf91f954c5b7910862a86c4a29f3cc2f2`.
- All 17 registered native `.pt` files currently exist; stat total **37,597,351,782 bytes**. Large native files were neither loaded nor rehashed here. Full payload byte integrity is inherited from `/mnt/project/st-world-policy-20261016/bindings/BASE_BYTE_AUDIT.json`, which recorded BYTES_VERIFIED; this audit does not pretend that inherited evidence is a fresh scan.

Historical source is `1493deda247107efe076b9294863a6753391298a`. The protected `DDPForesight.py`, `QwenOFT.py`, `GR00T_ActionHeader.py`, `foresight_dataset.py`, and actual `ddp_action_video_foresight.py` are byte-identical at 1493, this audit start, and reference `e0a6561e0708ec4f53a974e2b3ffd4c10406dfc9`. The e0 reference contributes decodable-feature research/reporting; it does not require replacing this Base or changing its forward function.

## Actual current-only inference contract

The strict checkpoint class is `starVLA.model.framework.ddp_action_video_foresight.DDPActionVideoForesight`, through `DDPFullForesight → DDPForesight → Qwenvl_OFT`.

Current observations contain `image`, `lang`, `state`, `token`. `DDPForesight.encode_current` rebuilds a whitelist of only `image/lang/state` before tokenizer or vision. It receives **CAM_F0, CAM_L0, CAM_R0**, one current image each, no history images. The data reader center-crops to 16:9 if necessary, resizes to **1024×576 with PIL LANCZOS**, and uses the original navigation prompt. Qwen processor then applies its actual fixed preprocessing; do not mistake the DINO teacher's 256×192 resize for the VLM input.

The VLM is local **Qwen3-VL-2B-Instruct**, language hidden width **2048**, 28 language layers; vision has 24 layers and three DeepStack injections at indices 5, 11, 17. Processor uses patch16, merge2 and temporal-patch2, left padding. Actual special tokens and prompt/processor pins are retained in the parent tokenizer directory and machine audit. One normalized relative ego history pose produces a `(1,4)` state, projected into the history slot. Native causal order is history → W → action.

`encode_current` already returns **W `[B,144,2048]`** and **action_queries `[B,8,2048]`** from the same Qwen execution. The **144 W** have view-major 3×6×8 geometry. Planner mode is **action_only**: raw final action hidden states are the complete planner condition, and the action head applies its existing `qwen_proj` once. W still affects causal action states through the unchanged original language computation. A new effect head may read W/A without a second VLM, new prompt token, future input or auxiliary target. A current-only compact cache should save the frozen W/A and legal state with their identities, not outputs of a trainable new effect head.

The existing DINO knowledge is the learned W representation and its current `dino_head` alignment. Current teacher targets are 3×6×8×1024, produced by DINOv3 ViT-L/16 at 256×192 with pool2. The native `TokenProjectionHead` maps W2048→DINO1024 view/position-wise. The DINO teacher network is **not** executed by Base deployment. Auxiliary current-DINO, future-clip and interaction heads may be stripped after strict full-state load; this does not remove W or its learned driving influence. Retaining and reading the trained current-DINO projection is a separately explicit effect-context option, not permission to run future heads or load a second visual encoder.

## Action and Flow Matching contract

This is the **original DiT/Flow model**, not the later single-Query IQE migration. Action DiT has 24 layers, hidden/cross-attention width1536, 24 heads×64, 8 action tokens, and output dimension4. Name `DiT-B` alone is insufficient: `QwenOFT` constructs the actual overriding `DiTConfig` from hidden1536.

Raw action is **8×4: normalized x, normalized y, sin(yaw), cos(yaw)**. Physical output is **8×3: x meters, y meters, yaw radians** in ego t0 rear-axle coordinates. Points are at dt0.5s, ending at4s. XY is absolute in that frame, not cumulative deltas. Exact original constants are:

```
x_raw = (x - 10.172484) / 8.805105
y_raw = (y - 0.360762) / 2.277741
yaw   = atan2(raw_sin, raw_cos)
```

The source provenance is original DDP1225 constants; no dev/test fitting occurs here. These are not claimed to be newly re-estimated statistics.

Training noise is standard normal with the original action dtype; the source time is `t=(0.999-Beta(1.5,1.0))/0.999`, with no clamp. Its small possible negative range must not be silently changed to uniform[0,1]. The exact interpolation is `x_t=(1-t)*noise+t*target`, target velocity `target-noise`, coordinate/time/batch mean MSE. Base training repeats eight independent noisy copies and preserves original dropout. An IEG paired target must use this actual direction and shared noise/time, rather than reversing a generic diffusion equation.

Deployment is original **Euler10**, t=0,.1,…,.9, `a ← a+0.1*v`. A fixed per-scene CPU FP32 standard normal `(1,8,4)` uses SHA256 `foresight-action-v1:{seed}:{token}` first8 bytes little endian mod(2^63−1), then moves to GPU. Registered sampling seed is42. New candidate sampling needs separate explicitly registered seeds, not mutation of the primary Base trajectory.

Inference reconstructs **learned FP32 optimizer master tensors** with DeepSpeed zero_to_fp32, performs `load_state_dict(..., strict=True)`, sets eval and `inference_fp32=True`, and disables TF32. Rounded BF16 model-save tensors are not a numerically interchangeable Base. The original Base training used BF16 execution with FP32 master/Adam moments; this does not mean canonical evaluation should use BF16.

## Data, simulator and usable prior results

Original training reader root is `/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1`: **101592 scenes /1176 logs**. Fixed dev root is `student_dev_v1`: **1696 scenes /16 logs**. Source index audit found **zero train/dev log overlap**. Base saw the full original training population; neither a new hash subset nor an expert holdout can be called Base-unseen.

Complete existing official training contexts are pinned by `/mnt/project/st-world-policy-20261016/audits/flow_policy/full_train_contexts_v2/TRAINING_MANIFEST.json`, SHA `b810cafbdaa42afd8554c08be443779e7d66edce56292c0c30940e728c0ed656`. It binds each legal current record/image, original ego target and real metric-cache hash. **It is a context/GT manifest, not a cache of S0 trajectories or W/A.** Do not load its GT/metric data in a deployable context encoder.

The active canonical score protocol is **NAVSIM v1.1, fixed cache PDM reference + one requested candidate**, internal0–1, submitted8×0.5s and simulation40×0.1s. Protocol identity `db383e17b887e08beb358b282fbeab223c80c558164bead81d52a45b1b798143`. NAVSIM v2/EPDMS is not bound by this protocol and cannot be mixed into labels. Detailed simulator/event extraction is owned by the feedback adapter audit.

Historical original-S0 full dev predictions exist at `/mnt/project/action-video-foresight-artifacts/20261002/evaluations/formal_S0_seed42_full100k_v2_dev100000_seed42/predictions`; their identity matches checkpoint78f870 and FP32 master/Euler10/seed42. Reconciled official dev summary reports **PDMS91.388363/100**, NC99.115566%, DAC98.349057%, TTC96.403302%, EP85.431072%, comfort100%, zero fraction2.476415%, 1696 scenes/16logs and zero failures. This is a historical descriptive baseline, **not a newly re-scored result**. Its old dependency pin differs from the current canonical protocol, so new paired comparisons must share the current scorer/context. No navtest trajectories/results were read for method or parameter selection.

This historical S0 is also distinct from the recent **C0 BC continuation PDMS91.1383**, and from any C1 posttraining endpoint; those cannot replace original S0 because their weights differ.

## Actual replay evidence and immediate next command

Existing historical input qualification uses fixed hash-selected TRAIN32, manifest `/mnt/project/st-world-policy-20261016/smoke/s0_input32_manifest.json`, SHA `c45e16a6c5bea8899e2a85b3ca16b8d75caf545d8d2b3c4e53e25c1162ab5bca`. Its `/smoke/s0_input32_v2/COMPLETE.json`, SHA `27768d1d46ee30c0368b63e851f05b56d2ff3e47582454c1afbfc7eaca0942e7`, records strict loading, 32 real imagesets, W/A dimensions above, exact poisoned-label independence, exact same-noise raw/physical repeats, one visual and one language call per scene, unchanged parameter/buffer hash, and no optimizer updates. It took245.94s load/peak12.95GB allocated on one A80080GB. This historic cost is not a new throughput claim.

After the Lead binds an actually free authorized GPU and a unique output path, the existing real-image replay entry is:

```bash
cd /mnt/project/VLA-Drive-action-effect-repair-20261020
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.qualify_s0 \
  --binding /mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json \
  --manifest /mnt/project/st-world-policy-20261016/smoke/s0_input32_manifest.json \
  --output /mnt/project/action-effect-repair-20261020/audit/s0_replay_train32_v1 \
  --device cuda:0
```

The output namespace must not already exist. This **has not been launched by this audit agent**. The original script runs deterministic TF32-off canonical eval and does not score or update. A new wrapper should compare every new per-scene raw/physical hash against the historical receipt; script self-repeat alone is not sufficient to establish historical identity.

For G0 cost pilot, prefer a new pre-score hash-selected TRAIN64 from the already fixed full context manifest, retaining IDs and log identities. Do not stratify it from dev outcomes or from a continuation's GRPO rollouts. Once new original-S0 scores exist, the Lead can freeze a larger TRAIN-only score-stratified manifest and budget.

## Asset boundaries and remaining work

No complete **original-S0 TRAIN101592 single-trajectory official bank**, nor complete original-S0 W/A condition cache, was found in the audited original-S0/ST/Flow asset roots. This is a bounded discovery result, not a claim that no unknown file could exist elsewhere. Generate only the compact contexts and predictions for the admitted G0 population; no multi-TB visual cache is needed. W144×2048+A8×2048 FP32 values take **1245184 bytes/scene**, about76MiB/64 scenes before metadata/serialization, not a measured stored-cache size.

Several attractive-looking existing assets are incompatible:

- `C1_100k_representations_v1` is old `DDPFullForesight`, checkpoint5e2ea076/source d1d408…, not this `DDPActionVideoForesight` S0. Its frozen W/A cannot be reused.
- Flow-safe `reference_samples/SFT_4seed` is SFT9a266… in original-BF16 root precision with fourseed ODE candidates; not this fixed FP32 S0.
- C0/C1 formal continuation predictions and on-policy rollout ledgers use changed policies and/or varying noise; not original-S0 candidate banks.
- Query/IQE caches belong to a different model architecture and checkpoint identity.

Pending new evidence: canonical model replay, new fixedTRAIN baseline trajectories/contexts, official fine-grained feedback, intervention execution, repair opportunity analysis, and all EGR/IEG training/heldout results. No scientific claim is marked PASS by this audit.
