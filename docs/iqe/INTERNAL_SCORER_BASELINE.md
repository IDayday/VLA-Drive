# Own expert and internal scorer baseline

This is an isolated offline experiment using our trained Query migration of the
original S0 framework. It does not change production release qualifications,
the running Base, or any other experiment's locked snapshot.

`iqe.internal_baseline` reuses the protected adapter, Query branch, IL definition,
normalizer, official single-output scoring implementation and exact-resume IQE
trainer. Its new scorer and all projections are randomly initialized. There is
no external scoring checkpoint, teacher or second VLM.

`bind-base` resolves the declared run's latest complete receipt once (or an
explicit checkpoint in that run), checks its SHA256/schema/full-training origin,
and writes immutable `BASE_BINDING.json`. This experiment uses 72,000 optimizer
updates, not a diagnostic initialization. The historical
`learned_driving_weights_loaded=false` field describes **initialization of that
Base training run**; it does not mean that its trained 72k tensors are random.
The offline adapter contract is explicitly not a deployment qualification.

The fixed inherited roles are sampled by seeded hash before reading rewards:
20,000 fit, 1,024 stage_val, 1,024 selector_cal, and all 1,696 dev_report scenes.
All 20,000 fit IDs appear in this Base's saved training list. The other selected
roles do not. This is an exposure statement about this Base run, not a claim
about unknown generic VLM pretraining. There is no final_test/navtest input.

One unchanged `encode_current` call returns W `[B,144,2048]` and A `[B,8,2048]`.
The W layout is three views, each a 6×8 spatial grid, in view/row/column order.
These are current-observation hidden states trained with auxiliary objectives,
not ground-truth future observations. The auxiliary prediction heads are stripped.
Original planning uses the original 16×256 Q-Former memory and ego encoder.
The score context stores raw frozen W/A, their true-valid masks, and the current
four-number ego input. No trainable projection is cached. Training targets occupy
separate, explicitly named supervision fields, excluded by the scorer whitelist.

The expert is an independent deep copy of Base Query/Decoder/all regression
heads. The bottom 20% Base fit scores define difficult candidates; seeded hashes
break ties. GT must score at least .8, beat Base by more than 1e-6 and receive
full NC/DAC. Ordinary examples come from outside the bottom 20%, at most 2,048.
The cumulative sampling plan is exactly 90% hard and 10% ordinary over ten
128-scene updates. The 1,000-update IL recipe and unchanged dropout/DropPath use
only the private expert optimizer. Stage_val Oracle selects its checkpoint.

The score head uses separate LayerNorm/Linear W and A projections, an ego MLP,
type and deterministic token positions, and physical trajectory XY normalized
by the locked Base constants, sin/cos heading and seconds. Two 256-wide,
four-head Transformer decoder layers attend within a trajectory and to the
projected context. Context projections run once per scene. Expanding candidates
as batch items creates no cross-candidate attention or expert identity features.

Pointwise loss is SmoothL1(beta=.1) + .2 BCEWithLogits over valid named
components. The first 500 updates use 128 naturally sampled fit scenes. The next
1,500 use 64 natural scenes and 64 comparison samples with q proportional to
the absolute true reward gap (ties ≤.01 excluded). The latter contribute .2
times their ordinary pair-loss mean; they do not enter pointwise regression and
are not gap-weighted a second time. This is the dataset-level gap-weighted
objective. No per-scene normalization cancels the gap for K=2.

Two consumed cursors and RNG streams, phase, bank hash and q hash are saved by
the existing optimizer-boundary checkpoint implementation. Prefetch/peek never
advances consumed state. Validation chooses highest true stage_val selected
PDMS, then lower regret and fewer new safety violations. Training maintains one
continuous 2,000-step schedule across its two phases.

The main result is predicted total-score argmax with Base ties. Optional margin
calibration uses only natural selector_cal and the fixed grid 0,.01,.02,.05.
GT is audited supervision, never an Oracle/deployment candidate. The diagnostic
bundle contains both own heads and the own Base dependency, and cannot update
formal ACTIVE. Scientific success is evaluated separately from engineering.

The 64-scene overfit head is freshly initialized and never warms the main
scorer. Its uniform diagnostic batches use the mathematically exact global
gap-weighted objective rather than pretending to be q-sampled. The selected
main scorer is separately tested with scene mismatch, reversed trajectory time
and zeroed W, reporting actual regret rather than just score variation.
