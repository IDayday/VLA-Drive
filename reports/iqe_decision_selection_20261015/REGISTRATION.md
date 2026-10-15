# Fixed-pool cost-sensitive decision study

Registered before new-model optimization on 2026-10-15 UTC. Audit/implementation
base: `1375ad2f67ed841f44bc7e6ec42bcc0f6d301bf0`. Isolated branch:
`research/iqe-cost-sensitive-selection-1375ad2`.

The subsequent `d97781a5` commit archives geometric results/visibility analysis;
`94542caa` adds an unrelated event-time diagnostic; `a16c6ef8` adds a summary.
No later shared training compatibility fix is needed. This study branches from
the requested audit commit, reads the completed geometric evidence as a
historical addition, and does not import or restart the stopped experiments.

Base70k SHA256: `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`.
Pool: `baccbfbfa27b387c30a58acf59cd6bd53a564cadd7dc6bdb29fab80de8b80029`.
Stable IDs are resolved from the actual registry: Base, E11 road EMA3000,
interaction raw3000, safe-progress raw1000. No generator updates, new candidates,
new scoring calls, external scoring weights, Navtest or ACTIVE mutation.

Authoritative frozen outputs live in
`/mnt/project/iqe-cost-sensitive-selection-20261015/main_v1`.
`SELECTION_STUDY_BINDING.json` binds all source artifacts, hashes, selected R00
weights/predictions and actual official reference context. Bank joins validate
all scene/ID/context/score/component records before model fitting.

Internal split: seed314159, union source-log/group/parent links, then stable
hash split. D_S has16581 scenes/211 logs; inner fit13139/169, inner val3442/42.
R00 saw every D_S scene: its inner-val results are in-sample background only.
Base also saw D_S; experts did not fit on D_S. New models never use external
stage_val, calibration or development labels for duration selection.

Two scratch models only: semantic scene-query Router and physical-trajectory
set-conditioned selector. Common context projection initializes identically.
Router has no trajectory argument. Set model has a Base role bit, no specialist
ID/index embedding. Both use hidden256, 4heads, two readout layers, dropout0;
the set model adds one noncausal candidate-set layer. No visual spatial reader.

Natural scene sampling; cost CE weights `r_i-min(r)` retained across scenes.
`Z_train=mean_fit(sum_i weights)` is fixed before fitting; no batch-normalized
cost or second weighted sampler. No absolute-value, component or ranking loss.

Each inner run:3000 updates, global128, micro16, AdamW1e-4/wd.01,
warmup100 then constant, clip1. Validate at500/1000/1500/2000/2500/3000.
Choose maximum inner selected PDMS, fewer newNC+DAC, then earlier step.
All scheduled checkpoints must execute; no early-stop or result-driven extension.
Refit once from the original random initialization on fullD_S for
`ceil(T_inner*16581/13139)` updates, capped by predeclared5000.
Maximum after actual split is3786. Last refit checkpoint is fixed; external
stage_val cannot choose another endpoint. Existing dev is exploratory, not blind.

Model seed20261015; train RNG20261035; sampler20261025; bootstrap11235/2000.
Optional margin grid is0/.25/.5/1/2 **logit units**, only selector_cal; newNC/DAC
budget0. Always-Base is a reference, never evidence of useful selection.
Report raw argmax regardless of calibrated fallback.

Conditional P_current: only if both inner models lack positive gain, or both
have high-to-zero errors after successful64-scene fits. At most2048fit/512val,
2000updates; current-map/current-or-past-agent inputs only, never future states.
It is a privileged diagnostic and cannot be exported. No automatic VLM unfreeze.

Real-context resume probes are disposable four-step continuations, eight
executed updates per model including the uninterrupted reference. Production
models remain scratch-initialized. Optional overfit checks are capped300 each.
The old experiments are held by user; results and all old weights remain intact.
