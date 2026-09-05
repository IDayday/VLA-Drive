# PlanReg Task-Future Lite: method and evidence contract

Code base: `e85e1a1797f1a26303e9ee81d9f3d1231bc59978`. Worktree:
`/mnt/project/DriveVLA-M0-planreg-task-future-lite`, branch
`feature/planreg-task-future-lite`. This is not a merge of CandidateConsequence V2.
The original V2 worktree and its uncommitted changes were inspected and left alone.
No V2 weights, old candidate banks, plots or caches were deleted or overwritten.

## Exactly three physical tasks

| Field | Raw label | Network output / training scale |
|---|---|---|
| projected_gap | Minimum projected geometric polygon clearance over 10 Hz times, actors and lags 0/.3/.6/.9 s | Five-class logits: contact, (0,.5], (.5,2], (2,5], >5 m / fully observed no-actor |
| road_margin | Minimum signed corner distance to original drivable polygon union boundary | Scalar, clip(raw,-2,2)/2 |
| route_progress | Nonnegative centerline arc projection change of simulated body center | Scalar, raw metres / 40, no upper clipping |

Eight bins partition source indices 0..40 exactly: 0..5, 6..10, ..., 36..40.
The last projected-gap query needs frame 49 (4.9 s); missing coverage is UNKNOWN,
not a safe target, and is never replaced by the last available frame. Contact uses
`intersects`, not `distance == 0`. These labels are neither official TTC nor
responsibility/NC classifications. There is no additional contact head.

The sidecar reuses the exact scorer rollout for all 64 proposals. Only GT needs
one additional rollout if it is not already present. Official body conversion
uses the vehicle rear-axle-to-center offset and four corners. The original map API
is reloaded for a hole-preserving road union: the existing training cache builder
stores polygon exteriors, which is insufficient for this new target. The official
scorer keeps using its original cache unchanged. Map availability/coverage is
checked separately from road membership. Group, candidate coordinates, initial
ego state, metric-cache bytes, map geometry, vehicle, rollout/controller/bicycle
and reference-conversion source hashes form label provenance.

The runtime audit records the actually imported paths and SHA-256 values. Observed
training occupancy has 51 frames, 10 Hz and observation_sample_res=1. The local
official NAVSIM scorer file SHA matches the pinned v1.1 source
`autonomousvision/navsim@3e8291bfa89ff247231e0227778840cd0a036896`.
No custom DDC multiplier is added to Lite. Physical and official component names
remain separate. Arrays, actor lists, scores and scorer source files are read-only.

## One shared decoder, not a deployment search model

`PhysicalQueryDecoder`: 616,455 parameters, d=128, four heads, two blocks, FFN512.
Inputs are current pre-fusion planning registers [B,16,256], ego/command and
detached trajectories [B,K,8,3]. Each candidate has eight time queries; attention
does not mix candidates. Outputs are gap [B,K,8,5], road/progress [B,K,8]. K=1/8/64
and chunked execution are supported. Training samples GT plus seven distinct,
uniformly sampled proposals. No candidate/GT identity is supplied to the network.

The current branch cannot receive future pose, image or labels. The hindsight
branch calls the **same** decoder with stop-gradient EMA current/future visual
registers, adding true logged pose/time to memory keys only. Values remain visual.
The three future views supervise bins 0/2/5 respectively. Current plus three future
frames are encoded in one EMA visual call per scene batch, never once per candidate.
The future camera belongs to the logged ego, not the candidate ego.

The WM objective is current physics + 0.5 hindsight physics + 0.25 task distillation.
Gap uses CE/KL; road and progress use normalized SmoothL1. Each task has a global
DDP valid-count denominator; all-invalid tasks do not dilute valid tasks. Hindsight
answers used for distillation are detached, but supervised hindsight loss trains
the shared decoder. The teacher is not assumed to be perfectly observable or more
accurate. No future-register abs/delta matching runs in Lite; the old predictor is
not instantiated. Legacy V1.1 register-prediction mode remains available.

New loss gradients reach the decoder, registers/readout and vision Q/V LoRA, not
candidate coordinates, generator/scorer parameters, LLM or EMA. Planning loss
continues along its unchanged path. All 24 Q/V LoRA layers remain rank32;
3,145,728 LoRA parameters, 48 Q/V adapters, 96 A/B tensors. FP32 trainable storage,
Adam moments and EMA master are retained. Frozen VLM computation uses BF16.

Standard export strips EMA, legacy predictor and PhysicalQueryDecoder. Deployment
is current single-front image -> V1.1 vision/fusion -> original generator -> scorer.
It does not use physical answers, evaluator, future frames, search or coordinate
correction. Auxiliary diagnosis requires an explicitly loaded training checkpoint.

## Boundaries relative to the rejected large V2 proposal

Not implemented: ego surrogate, actor tracking slots, responsibility head, 41x4
neural outputs, full state/RGB reconstruction, PDM reference, comfort heads, separate
prior/posterior/teacher decoders, scorer residuals, ranking/listwise loss, CEM/TOAD,
coordinate refinement, latent matching, error-based sample rejection or replay.
Lite does support multi-candidate *training-time physical answers*, but does not
implement the previous multi-trajectory future-register/structured-consequence
model or any consequence-driven deployment policy.

## Scorer comparison controls

| Variable | Fixed DrivoR | V1.1 / Lite |
|---|---|---|
| Camera input | Four cameras | One front camera |
| Vision | Whole-frame DINOv2, bidirectional token interaction | InternViT dynamic tiles, read-only internal registers |
| Registers | 16 scene registers per camera | 16 internal; global/local 8+8 readout; final memory length 16 |
| Scorer function | Fixed four-layer decoder and six component heads | Exact same adapted function, but different input representations |
| Generator intermediate loss | prev_weight=0 in pinned config | Final-head-only default retained |

Function parity does not prove representation equivalence or that one selector is
stronger on a different candidate bank. No paired DrivoR-model candidate bank was
run for this task. Old epoch33 selected PDMS 0.9133282822 is an already reported
old-model result, not a newly trained V1.1/Lite result. No new Navtest result and
no causal WM PDMS benefit are claimed. Navtest was previously used for research
diagnosis; it is not used for Lite training/probe sampling or hyperparameter choice.
