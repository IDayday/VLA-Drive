# P2 world learnability — active

Implementation is real and both original vision/Qwen/DiT remain frozen. Reader, queries, reference encoding/projection and current/motion heads receive gradients and update. Head parameter capacity and initialisation are matched between W_PRE and W_POST; only the readout before/after frozen Qwen differs. Ground truth enters only matching/loss. W_POST backpropagates through frozen Qwen.

The initial recipe uses fixed64 references covering the unchanged ROI,5m XY residual decoding, reference-based current no-object eligibility, and GT future displacement loss with the current-centre gradient detached. It does not insert GT boxes/target counts. Per-scene numerator/count accumulation is verified against joint accumulation; live Qwen graphs are released after each scene. Only frozen visual features are memoised in-process, keyed by current pixel/grid content.

## Matched16-scene evidence at300 updates

| Readout | Geometry recall | Precision | Class-correct recall |
|---|---:|---:|---:|
| W_PRE |57.6779%|35.5658%|55.4307%|
| W_POST |61.4232%|41.1028%|58.8015%|

Each compared checkpoint saw2400 sample presentations =150 epochs over the same16 scenes. The W_PRE process was interrupted at304 updates for conservative family-budget reconciliation; the extra4 updates remain charged and logged but are not the comparison checkpoint. W_POST ends its16 phase at300. Earlier W_PRE200/W_POST200 comparisons and all failures remain. No LoRA experiment is justified by these results: W_POST is not failing while W_PRE succeeds.

##64-scene gate pending

Independent same-initialisation64 runs each receive696 updates. Combined initial isolation-family budget is W_PRE304+696=1000, W_POST300+696=996. The initial accidental1000-step64 reservations were cancelled before any update; their GPU wall time is charged. See ISOLATION_BUDGET.json. This bookkeeping adjustment is not a convergence-repair round or discarded bad result.

Do not claim80% recall/50% precision before final full-GT evaluation. Initial64 targets are exactly equal in every field to the independently generated capacity999 full cache; no training scene has overflow. Full64 initialisation,100-step checkpoints and fixed final696 evaluations are retained. No failed samples are filtered.

Task query-gradient norms, group gradients, parameter probes, losses, exposure and memory are logged. The legacy trainer's `grad_norm_after_clip` is the clipping bound computed from the measured preclip norm, not a separately recomputed norm. Assignment-churn logging and stronger per-task-gradient diagnosis remain required if a repair is launched. Do not present those unrecorded traces as already measured.

## Regression and artifacts

14 targeted CPU tests pass, including independent maximum-cardinality matching, coordinate/displacement gradients, reference eligibility, accumulation with empty/different-count scenes, append/mRoPE and zero-gate gradient behaviour. Real300-step world checkpoint strict reload/prediction parity, label contamination and newly added zero-gate action parity pass (one real scene), reusing the earlier64-scene native-prefix exact test. Actual16-phase optimizer/scheduler/RNG/sampler resume executed. Fast-kernel continuation is not claimed topology-independent bitwise.

The full-GT16-scene re-exports produced scene/object CSV, predicted NPZ and32 figures (16 scenes ×2 variants), with geometric evaluation matching. Actual paths/hashes in ARTIFACT_INDEX.json. Full64 and log-held-out exports remain pending final training; lack of matching motion instances must remain missing, not zero.
