# GT-future completion diagnostic at 512 updates

This diagnostic responds to the question whether role masking actually uses real other-vehicle futures, and whether that helps predict the hidden ego. All results below are on the SAME fixed64 TRAINING scenes; they are neither held-out scores nor PDMS. Checkpoints have512 verified updates each. One training seed and one sampling seed42 are used. Each arm retains64/64 queries and has0 failures. No optimizer updates were made by this diagnostic. Total allocated inference cost, including loading:0.241582 GPU-hours on previously idle vla-zt2 GPUs1/2. Formal training source4f27cbb5b83806331325aa473a01da4fd1738d60 remains unchanged and running.

## What the current training code actually does

- Main B/C FM and camera-only evaluation hide ALL future trajectories. Ego and other vehicles are jointly generated from noise; executed ego is joint slot0. They do not first obtain a separate accurate neighbor forecast and feed it to another ego planner.
- C role auxiliary hides the entire ego or one eligible neighbor future. Other valid futures are CLEAN GT, inserted into joint DiT state slots, with an explicit known-coordinate embedding. They are not predicted substitutes or Qwen prompt features. Target states are FM-corrupted during training and initialized from noise when sampling.
- B's matched auxiliary is all-hidden. C auxiliary occurs every4updates with weight0.1; final64 small-fit updates disable the auxiliary for both. C used528 ego-hidden and546 neighbor-hidden scene presentations, with718 all-hidden fallbacks out1792 extra presentations, alongside8192 main-task scene presentations. Thus the main ego predictor does not see GT other futures on every training example.
- Whole-actor trajectory masking is a trajectory analogue of masking image patches: the visible parts are real labels. Learning useful completion and transferring that learning to all-hidden deployment are distinct empirical questions. The earlier camera-only table could not answer the first question; the missing conditional diagnostic was a validation gap.

Code: `starVLA/model/modules/vehicle_joint/masks.py` (`known_mask`), `action_head.py` (`loss` and `sample`), and `starVLA/model/framework/DDPVehicle.py` (`forward` and `predict_action`). Targeted CPU action/mask tests:9 passed, including an added assertion that visible neighbor states equal clean GT while the hidden ego state equals the prescribed noise/target interpolation. This is evidence for the data path, not proof of algorithmic effectiveness.

## Fixed comparison

All64 ego identities were fixed before model loading. Cameras alone build the current graph before labels load. The same current Hungarian loss assignment, without a2m/class gate, attaches same-track future GT only to already selected predicted neighbor slots. Future labels never select nodes. Within each model: target, current graph and initial noise stay identical; compare all-hidden with ego-hidden/other-valid-futures-known. Ego labels are opened only after generation for scoring.

All-hidden predictions exactly reproduce the prior saved camera-only outputs (maximum absolute difference0). Known GT coordinates remain EXACT at every Euler step; unmodeled vehicle yaw stays zero; hidden input values are deliberately NaN-poisoned and removed before encoding. All19 no-condition scenes yield exactly identical paired outputs. This rules out the suspected replacement of visible GT by model predictions in these checked paths; it does not establish that the model makes beneficial use of GT.

## Actual results

Meters; lower is better. Positive conditional-minus-all-hidden difference is worse.

| Model / population | Scenes | All-hidden ego ADE | With GT ego ADE | Difference | Paired log95% interval |
|---|---:|---:|---:|---:|---|
| B Joint / all |64|3.448309|3.496724|+0.048416|[-0.013747,+0.118727]|
| C Joint-Mask / all |64|3.800499|3.893978|+0.093479|[+0.033578,+0.157720]|
| B / other GT available |45|2.833750|2.902608|+0.068858|[-0.019405,+0.166699]|
| C / other GT available |45|3.114241|3.247189|+0.132948|[+0.049204,+0.218314]|
| B / no other GT |19|4.903842|4.903842|0|[0,0]|
| C / no other GT |19|5.425847|5.425847|0|[0,0]|

On45 C queries with other GT, ego FDE worsens2.474504→2.579594m; its paired interval includes zero. Ego yaw MAE changes0.049410→0.049662rad, also with interval crossing zero. The stronger negative evidence is ADE under this fixed protocol. The intervals are log-cluster paired bootstrap over this training diagnostic (43logs for the45 condition queries), not cross-training-seed uncertainty.

C also worsens when ALL active neighbor points are known:32scenes, ADE2.822894→2.942947m, difference+0.120053m [ +0.017166,+0.234159 ]. Incomplete future availability alone therefore does not explain the result. Strata were registered before reading their errors; see strata.json.

## Interpretation and limits

The checked implementation does expose clean other-actor GT as intended for C's role task. However, this512-step model has NOT demonstrated useful ego completion; supplying GT slightly worsens its measured ADE. The previous all-hidden results also did not demonstrate MASK transfer. Neither observation proves that the general masking idea is invalid, nor permits a claim of planning benefit.

A concrete representation problem remains: the scene-mean assigned current center error is approximately12.44m in both arms, despite using the correct same-track GT future. A real future is thus attached to a still inaccurate predicted vehicle slot/current condition. Current detection coverage is only14/333 for B and11/333 for C under the separate2m evaluation criterion. These poor representations, limited role exposure and competing objectives are plausible contributors; this experiment does not isolate their causal contribution. Do not reintroduce a correctness gate that deprives initially inaccurate queries of supervision.

B never trained the GT-conditional mode; its result is an intervention diagnostic, not a fair trained-completion baseline. B/C have the same45 condition-available ego scenes but their predicted graphs and supplied neighbor populations differ:181/186 known assignments,1432/1471 known xy points. Only11conditioned scenes have identical track-and-point-count sets across B/C. Therefore the primary result is each model's OWN paired conditional effect; a direct B/C conditional gap is not a pure mask-effect estimate. Private assignment/track files stay outside git.

The small-fit recipe ends with64all-hidden updates; this endpoint diagnostic alone does not determine whether conditional ability was acquired earlier then lost. No conclusion about full-training convergence, held-out generalization or Navtest is available here. Existing formal runs continue with their registered recipe; no training/model/graph thresholds were changed in response to these results.

## Evidence and reproduction

Diagnostic source:ea0398943f1cb2904f1357814bb754a44f64465b. Small-fit training source:4843e4ddf8340ecc8b44a47fc5fc9b688a58e3a8. Checkpoint/current-data/label/query hashes are in B_identity.json and C_identity.json. Full derived query rows: B_queries.csv and C_queries.csv. All metrics/failures/denominators remain in the summaries. No raw trajectories, annotations, track IDs, scene pictures or weights are published.

Exact executed commands are recorded in plan.json. For a new rerun, use an AVAILABLE authorized GPU, a new `--output` and a new `--run-id`; keep all other identities unchanged. Do not restart an existing COMPLETE allocation. Example targeted test command:

```bash
cd /mnt/project/VLA-Drive-ddpolicy-completion-20260928
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/ddp/bin/python -m pytest -q tests/vehicle_joint/test_action_head.py tests/vehicle_joint/test_graphs_masks.py
```
