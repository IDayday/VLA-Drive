# Bounded acceptance evidence

Evidence root: `/mnt/project/simwam-recipe-20261015/qualification/`.
Qualification weights are not formal initializations.

| Check | Actual evidence/result |
|---|---|
| Locked source oracle vs isolated code | `math_extended.json`: 64 checks PASS; actual original class methods loaded directly, FP32/BF16 mock action experts |
| Time/sign/std and source tails | shift5 grid, sigma1 guard, last stochastic step, explicit FP64 formula, innovation and relative-clamp behavior; unchanged source bodies |
| Loss/gradient reductions | Source total/PPO/anchor and adapter gradients; worlds1/2/8 with partial/all tied ranks; tied ranks retain anchor |
| DDP four-channel mapping | Explicit heading-pair-average calculation; original normalization and relative-to-t0 semantics retained |
| Source engine clipping | `source_engine.json`: real DeepSpeed optimizer step; effective gradient_clipping=0.0 |
| Real single-GPU learning | `single_real_v1/`: eight updates, two fresh G8 groups, all 192 A/B changed, frozen SHA unchanged |
| Real eight-GPU learning | `multi_real_v1/`: one fresh 64-candidate rollout, four actual updates; first-inner ratio error exactly0 on all ranks |
| One vs eight GPU update | `multirank_parity.json`: first gradient relative L2 7.4973e-8, cosine1; max saved-chain ratio discrepancy 3.5763e-7 |
| LoRA coverage and merge | `lora.json`: all96 projections actually called; zero-LoRA ODE max error0; trained merged velocity max1.1921e-6 and ODE max2.3842e-7 |
| Native interruption/resume | `recovery.json`: interrupt after inner1, finish through update8; model/Adam/scheduler/RNG/next fresh chain/advantages/losses exactly equal |
| Original raw deployment export | `raw_observation_export.json`: actual VLAAgent, two real TRAIN inputs; original upstream unchanged, no noise head/teacher/cache/scorer dependencies |
| Paired ODE scoring | `dual_grid_evaluation_v2/rank0.json`: two real DEV scenes, four independently scored trajectories; direct original native output exact |

Eight-GPU interruption/recovery initially failed exact equivalence in
`eight_gpu_recovery_v2/`: final adapter max difference 2.65e-8, despite exact
first chain/advantages/old density and the same scheduler/cursors. The default
DDP reducer rebuilds buckets after its first backward; restarting changes the
iteration at which this FP32 reduction layout is used. The fix explicitly
uses `find_unused_parameters=True`, `static_graph=False`, bucket25MB; the
installed Reducer does not rebuild in this mode. All192 adapter gradients
remain mandatory, so unused detection does not relax the parameter contract.
The new `eight_gpu_recovery_v3_fixed_buckets/` test is running; no exact
eight-GPU recovery success is claimed before its result exists.
Both formal variants and complete evaluations are not yet reported as done.

## Numerical limits and preserved failures

The serial explicit-8-scene vs distributed-eight-rank test does not show
bitwise Adam parameter equivalence. Maximum parameter differences across
steps1–4 were 2.40e-10, 5.79e-9, 1.51e-6 and 2.72e-6. Relative LoRA-update
L2 reached 0.003287. Initial overly strict parameter-level checks failed;
their logs and intermediate JSON remain in the private evidence directory.
Subsequent measurement found loss-mean agreement within 1.8e-8, first-step
gradient relative L2 7.5e-8, and fixed-chain density-ratio discrepancies at
most 3.6e-7, versus PPO clip0.005. This is an output/gradient-qualified FP32
platform adaptation, not a claim that the failed parameter tolerance passed.

The first native-cache smoke omitted original CUDA FP32 autocast and failed
with a BF16-input/FP32-linear mismatch. Original autocast was restored, not
the original model changed. Compatible old original-observation samples then
matched exactly. A diagnostic relative-clamp reconstruction initially used
a rounded FP32 std in its peak; it was corrected to the source Python-double
order. The source probability function itself remained unchanged.

The first paired DEV evaluation could not locate DEV v1 caches because only
TRAIN cache roots were passed. It aborted, retaining that failure. The
corrected evaluation passes the existing DEV v1 root explicitly; no missing
scene became reward0 and no v2 fallback was used. The corrected two-scene
smoke completed. Full DEV runs have distinct identities from smokes.

Previously verified independent-candidate batched scoring and corrected
condition-cache evidence are reused from
`reports/ddp_constrained_flow_noise_20261014/README.md` and its linked
numerical/efficiency reports. The scoring implementation hash remains locked.
No unrelated full-repository re-audit was performed.

## Real commands

From this worktree, with the existing ddp environment python, the following
entries have been executed. GPU commands require an authorized permanent
Tsinghua host; dispatchers enforce policy. Replace `PY` with
`/mnt/project/flow-safe-improvement-20261013/venv/bin/python`.

```bash
PY -m scripts.analysis.simwam_recipe.lock_upstream --source /mnt/project/simwam-recipe-20261015/upstream/SimWAM --flow-grpo /mnt/project/simwam-recipe-20261015/upstream/flow_grpo
PY -m scripts.analysis.simwam_recipe.describe_recipe
PY -m scripts.analysis.simwam_recipe.qualify_math --output /mnt/project/simwam-recipe-20261015/qualification/math_extended.json
PY -m scripts.analysis.simwam_recipe.dispatch_baseline --lanes 2
PY -m scripts.analysis.simwam_recipe.dispatch_evaluation --host training-vlawm-tsinghua-zt2 --export /mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-checkpoints-v1/frozen_visual --export-sha256 9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4 --label SFT_update000000
PY -m scripts.analysis.simwam_recipe.qualify_recovery_run --name eight_gpu_recovery_v3_fixed_buckets
```

Native training entry: `scripts.analysis.simwam_recipe.train`, invoked by
`python -m torch.distributed.run --standalone --nproc_per_node=8 --module`.
The real qualification passed `--qualification --stop-after-update 8` and
the fixed qualification TRAIN list. Formal invocation omits those flags and
requires the sealed complete hard manifest. `--resume /absolute/checkpoint`
continues the same output directory and identity. SIGUSR1 requests a native
inner-boundary save; it does not discard the saved old chain. The real
single-GPU interruption test used `--stop-after-update 1` then resumed
`update000001_inner1` through update8.

Export entry: `scripts.analysis.simwam_recipe.export_policy`; coverage entry:
`qualify_lora`; distributed comparison entry: `qualify_multirank`; exact
resume comparison entry: `qualify_recovery`. `--help` gives their actual
arguments. The final formal run commands and frozen source SHA will be
recorded when dispatched, rather than listing nonexistent runs.
