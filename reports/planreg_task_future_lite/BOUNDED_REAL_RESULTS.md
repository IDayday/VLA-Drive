# Bounded real-model results, not a PDMS claim

Artifacts are rooted at
`/mnt/project/DriveVLA-M0-formal-runs/task_future_lite_20260905`.
Use the v2 shared artifact / v2 smoke / v2 probe. The first Lite artifact and
results remain available but are superseded: a final frame-key MLP bias cancelled
algebraically in `embedding(pose)-embedding(0)`. It was removed (128 parameters),
and all paired-init/update/export checks were rerun on the corrected topology.
No existing V1.1 artifact was changed.

## Numerical and wiring evidence

- Full real InternVL3-2B Base and VQA, not a tiny substitute: 468 effective
  trainable tensors were bitwise identical before learning, including the new
  decoder. Effective trainable parameters: 21,249,830, all FP32.
- Eight B=2 optimizer steps over eight distinct training logs (turn, stop,
  crowded and near-road-boundary examples, repeated twice). All 468 tensors
  changed; all 24 Q/V LoRA blocks had nonzero gradients. Adam first/second
  moments contain 42,499,660 FP32 values. Nonzero gradient was not used as a
  substitute for checking actual tensor changes.
- Peak allocated memory in this **B=2 single-GPU smoke**: 18.6224 GiB; this is
  not the formal B=4/16-GPU layout measurement.
- First-step total loss 27.24623, unweighted WM 1.19515, WM weight .01; all eight
  step losses and gradients finite. These eight steps do not establish convergence.
- Same-batch unclipped gradient audit: vision plan norm3.21246, unweighted WM
  norm.47257, weighted WM/plan .001492 at lambda .010145, cosine -.51414.
  Register ratio .001131/cosine .11457; readout ratio .001016/cosine .15679.
  This one early batch is not a justification for increasing lambda, particularly
  given the negative vision cosine. Optimizer.grad remains unpolluted.
- Student-only export and strict reload: no auxiliary/EMA constructed; no future
  inputs; current policy trajectory max_abs_diff=0. No old agent checkpoint was
  used to initialize the fresh planning stack.
- Real train-cache sidecar test: original four scorer arrays exactly unchanged,
  serialized cache state unchanged; identical GT reuses the generated rollout.
  Fixed upstream six logits, aggregate and selection parity differences remain 0.

The real smoke's label fingerprints predate the additional controller/bicycle
source fingerprint fields, which change provenance keys, not physical label values.
The expanded source-key code is covered by the subsequent real sidecar test and
16-GPU benchmark. Earlier V1.1 numerical/EMA tests are also rerun in the full suite.

## Train-only, log-disjoint 150-step learnability probe

Sixteen distinct logs, eight train and eight development, with four scene
categories. This is a small **frozen-upstream** diagnostic. The visual student
received eight smoke updates on only the probe-training logs first. VLM
pretraining exposure to these logs is unknown. It is not an unseen-pretraining
claim, a formal no-WM experiment or evidence of downstream generalization.
Only in this frozen probe are visual features/trajectories reused in RAM; formal
training recomputes dynamic student and EMA representations online.

The current/hindsight shared decoder and a same-capacity action/ego-only decoder
are fitted for 150 steps. Wrong-current and wrong-future inputs are interventions,
not training-data selection. Correct and wrong future comparisons keep the same
logged pose, so differences are not caused by removing pose metadata.

Development numbers from `learnability_probe_v2.json` (lower is better):

| Input | Gap Brier | Road clipped MAE (m) | Progress MAE (m) |
|---|---:|---:|---:|
| Current visual + action | .72849 | 1.01521 | 5.36745 |
| Action/ego only | .80329 | 1.07535 | 4.35318 |
| Wrong current image | .66409 | 1.22506 | 3.68978 |
| Hindsight, observed bins only | .76362 | 1.08998 | 5.64943 |
| Wrong future, same pose/bins | .75531 | 1.09860 | 5.14594 |

Current metrics use all valid bins; hindsight rows use bins 0/2/5, so their
aggregate values are not a matched-bin teacher advantage estimate. The JSON also
reports calibration bins, near-contact recall, road sign/near-boundary errors and
entropy by horizon. Near-contact recall for current/action-only is .80995/.79638.

Interpretation: train fitting is feasible, but development evidence is mixed.
Road error is worse with mismatched current images; gap Brier does not consistently
prefer the correct image. Progress is better with action-only here. Hindsight is
not uniformly better. No visual causal benefit, complete counterfactual uncertainty
distribution, physical safety guarantee or PDMS improvement is established.
Retain the specified .01 -> .10 WM schedule and equal tasks; report these limits
for researcher review rather than silently changing the tasks or deleting hard data.

## Verification status

The complete existing+new suite ran: **231 passed, 23 warnings**, no skips,
37.19 s. This includes the prior 201 tests and 30 new tests; a subsequent focused
  eight-test run passed after report/config-audit changes. Compileall and diff-check
  passed. Tests include two-process Gloo global masked-loss normalization, temporal
coverage/boundaries, holes/center conversion, K/chunk interfaces, gradient routing,
  old-mode/export compatibility, cached-data contracts and scorer immutability.

Label extraction in the real smoke averaged .36139 s/scene (maximum .47960 s,
16 scene calls), excluding the original scorer and cold-map loading. A separate
synthetic-production-shape **cost-only** A800 benchmark (B4/K8, 20 warmup + 100
iterations, FP32 storage/BF16 autocast) measured .002771 s for the current decoder
and .008976 s for the three hindsight calls. This microbenchmark excludes vision,
labels and backward and is explicitly not a substitute for the full real-model
smoke or distributed end-to-end measurement.

NOT_RUN: new Navtest evaluation; paired DrivoR-model candidate-bank comparison;
multi-seed formal results; full 27-epoch convergence/comparison. No bank was
available/required for a matched model comparison, and the formal runs must finish
before final deployment evaluation. The benchmark and actual launch states are
reported separately in EXECUTION_STATUS; nothing here promotes a queued run to PASS.
