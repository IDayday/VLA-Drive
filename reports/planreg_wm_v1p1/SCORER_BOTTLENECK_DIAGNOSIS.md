# Scorer bottleneck diagnosis

## Old reported facts — not rerun in this task

| Checkpoint | selected PDMS | offline Oracle@64 |
|---|---:|---:|
| epoch27 | 0.9137876582 | 0.9872393164 |
| epoch33 | 0.9133282822 | 0.9881961163 |

Epoch33 reported regret=0.0748678341; catastrophic misselection (oracle>0.9,
selected<0.5)≈3.41%. Semantic slots were nearly identical and the gated crop
residual was about 2.96% of thumbnail RMS. No matched no-WM formal result exists;
WM has not been shown causally to improve PDMS. Base/VQA superiority is unknown.

## New offline analysis of the unchanged epoch33 bank

`epoch33_head_substitution.json` is a new calculation on the existing 12,146-scene,
136-log scored bank—not new inference or a deployable oracle-assisted model.
True component replacements are never used in training or deployment.

| True component substituted | selected PDMS in offline counterfactual |
|---|---:|
| NC | 0.92527 |
| DAC | 0.93206 |
| TTC | 0.95134 |
| EP | 0.95546 |
| Comfort | 0.91335 |
| EP+TTC | 0.97534 |

The JSON includes benefited/harmed scenes and log-cluster bootstrap confidence
intervals. DDC weight zero never forms `0*log(0)`; NC uses the exact binary mapping;
invalid TTC retains the model prediction. Effects are **not additive independent
component contributions**, and counterfactual upper bounds are not forecast PDMS.

94 PDMS is a target, not a guaranteed acceptance score. On the fixed epoch33 bank,
it requires regret dropping from 7.487 to 4.820 points, about a 35.6% reduction.

## New train-only learnability probe

`SCORER_LEARNABILITY_PROBE.json`: epoch33 checkpoint, 16 train-log scenes, complete
64-candidate groups, frozen vision/fusion/Q-Former/generator, 150 extra scorer-only
AdamW updates at LR=1e-4. No Navtest labels or high-regret Navtest token list were
read. Cached scene features are allowed **only while their upstream is frozen**;
unfreezing or updating an upstream module/query/gate invalidates the cache.

- Exact scorer loss: 0.35691 → 0.18656.
- Selected PDMS on these same training groups: 0.98648 → 0.99277.
- Their offline Oracle@64: 0.99527.

This demonstrates additional fit on a very small, mostly easy training sample,
not generalization. The old model had already seen full trainval; these scenes
are not an unseen validation set. Probe updates are additional compute and not
part of an equal-compute 27-epoch result. No automatic formal replay was enabled.

The probe and replay builder bind token, complete candidate tuple, source
checkpoint, evaluator and metric-cache hashes. Singleton/group checks were tiny
on the sampled groups (float32-rounding scale); this does **not** establish global
candidate-group independence. Full 64-group cache keys are always retained.
Never attach old labels to newly generated coordinates.

The first probe attempt used FP32 target views and exposed an in-place label-view
version error. The diagnostic now preserves the online evaluator's FP64-label
boundary before calling the unchanged exact loss. The scorer implementation,
its decoder, six-head BCE, TTC mask, NC/DDC mapping and aggregation were not edited.

## Next evidence gates

Uniform full-data remains the formal default. Replay sampling is opt-in research;
only a train-only, log-split pilot can justify changing it. Do not use Navtest for
loss weights, sampler, temperature, stopping, epoch or layout/convergence decisions.
Its previous use for research diagnosis is disclosed; it is no longer wholly
unseen for all new design choices.

Final paired evaluation must report selected/Oracle/regret/catastrophes, top2/4/8
recall, EP/TTC errors and ordering, mean/P10/P25 candidate quality and log-cluster
bootstrap. Future-register predictor outputs never enter the scorer. No rank,
listwise, repulsion, consequence or new scorer loss was added.
