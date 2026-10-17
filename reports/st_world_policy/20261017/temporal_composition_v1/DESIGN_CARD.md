# T3/T4: identifiable temporal mechanism before new training

Implementation start: `951711fbd2eae1730262a79575b08024bcd2001d`. Author:
`/root/st_owned_descendants_review_v2`. This is implementation evidence, not an
independent approval or a real temporal training result.

**Question.** Does causal history reduce ambiguity of future representation,
and does raw time-difference supervision add useful change information beyond
absolute future supervision? These are hypotheses, not established benefits.

| Hypothesis | Identifying comparison | Evidence that would contradict it |
|---|---|---|
| H1: true past observations resolve motion/occlusion ambiguity that a current image alone cannot resolve | T3−T2, same spatial parent, original selected scenes, adjacent future pairs, patches, target/PCA, precision and update exposure; T1−T0 separately measures history without future labels | No held-out future improvement over copy-current / history extrapolation references; correct history no better than repeated-current or time-reversed diagnostic; planning or safety worsens |
| H2: difference loss teaches relative visual change missed by absolute reconstruction | T4−T3, identical history encoder and absolute losses; shared predeclared scientific coefficients after the existing scale measurement | Absolute/static content deteriorates, change direction/amplitude remains no better, or lower auxiliary error does not transfer to PDMS |
| H3: apparent benefit is merely extra input/compute or exposure | Report T1−T0 and T3−T1, input/encoder cost and original-scene valid denominators | A gain vanishes under matched exposure or is explained entirely by history input; then do not attribute it to future/delta learning |

Source authority is the complete archived v1.0 contract, sections 3, 6.1–6.4,
12, 14 and 16, plus its complete multi-agent execution contract. This patch
introduces no paper-derived mathematical method. It composes existing reviewed
code: `HistoryEncoder`/`build_history_encoder` in
`starVLA/model/modules/st_world/history_adapter.py`; `TEMPORAL_CELLS` in
`st_world/config.py`; `requests.sample_requests`; existing masked temporal losses;
the original `train_world.py`; and the fixed8 indexed teacher provider.
Their exact inspected revision is the start SHA above. Existing DINO-Foresight
and history evidence cards remain their source evidence; this patch makes no new
claim that those papers established our T3/T4 benefit.

The original history path already has real T1 evidence (historical two-GPU,
4-update versus explicit 2→4 full-state comparison). Its official-runtime
resource-monitor failure is retained. The completed full-population T2 world2
64-update run establishes that provider/runtime combination only. Neither is an
actual T3/T4 result, nor formal global32 efficiency approval. No old test rerun
or old failure is relabeled PASS here.

**Minimal discovered gaps.** The model already composes history with future and
delta; no encoder or trainer rewrite is needed. `validate_profile_recipe`
rightly rejected history under the original T2 admission. A new explicit
`temporal_composition` binds an unchanged T2 parent recipe, history COMPLETE,
history manifest identity/hash, and exactly the T3/T4 cell differences. The
original T2 guard remains. Dataset attachment checks loaded history source,
partition, view order, scene/log and current-record pins against the actual
selected current/future population. This adds no second history-shard/image scan.

**Actual asset counterexample.** The real exact32 provider scene list and the
old history64 manifest have **0 overlapping scene IDs**: all 32 are missing.
Therefore combining those paths would be wrong. `PREPARATION_INPUTS.json`
retains exact small-metadata pins and this negative result. Build matching causal
history for the immutable exact32 selection with the existing bounded builder;
do not replace the future population, repeat current images, or call 64 scenes
full history. Full history is still unavailable.

**First real observation to obtain.** One separately admitted T3 or T4
4-update/32-scene PROFILE, original 20k schedule/global2/micro1, with actual
history and future provider identity checks, finite loss and nonzero gradients
on intended history/W/readout branches. Compare raw input/label boundaries and
loss denominators with existing tests. This is an integration observation;
4 updates cannot adjudicate H1/H2, converge a model, select W capacity, or select
scientific loss coefficients. Unit delta and existing future weight are explicitly
diagnostic only; this patch does not adopt a calibration measurement.

After independent code/resource checks, scientific T0–T4 use the contract's
matched 5k exploration/common20k schedule, selected spatial parent and one-time
training-only coefficient decision. This patch does not authorize those runs.
