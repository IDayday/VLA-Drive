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
| Formal initial pairing | `qualification/formal_initial_pair.json`: all eight ranks, 64 candidates, exact initial chain/old logprob/all candidate submetrics across R_SOURCE and R_CANONICAL |

Eight-GPU interruption/recovery initially failed exact equivalence in
`eight_gpu_recovery_v2/`: final adapter max difference 2.65e-8, despite exact
first chain/advantages/old density and the same scheduler/cursors. The default
DDP reducer rebuilds buckets after its first backward; restarting changes the
iteration at which this FP32 reduction layout is used. The fix explicitly
uses `find_unused_parameters=True`, `static_graph=False`, bucket25MB; the
installed Reducer does not rebuild in this mode. All192 adapter gradients
remain mandatory, so unused detection does not relax the parameter contract.
The new `eight_gpu_recovery_v3_fixed_buckets/` test **passed**: all eight ranks'
RNG, saved/next-fresh chains, advantages, old density, losses, grad norms and
learning rates are exact, as are LoRA/Adam/scheduler states. Its receipt is
`qualification/eight_gpu_recovery_v3.json`. The reducer reports one 18,874,368
byte bucket and `has_rebuilt_buckets=0`. The failed v2 evidence is retained.
Both formal variants and complete evaluations are not yet reported as done.

The fixed-bucket layout was also compared against an explicit serial eight-scene
reference through eight actual updates and two fresh global rollouts:
`qualification/multirank_fixed_bucket_parity.json`. First gradient relative L2
is 7.4973e-8 with cosine1. At update8 the parameter max difference is 2.9030e-6,
relative LoRA-update L2 is 0.0013631, and fixed-chain ratio discrepancy is at
most 5.9605e-7 versus PPO clip0.005. This measures the final formal layout;
it still does not claim bitwise serial-vs-parallel Adam equivalence.

`qualification/diagnostic_interfaces.json` exercises the separately named
`native_ddp_grid` and `unclipped_math` interfaces. Native mean-only sampling
matches the original Euler sampler exactly on a real DDP condition; the
unclipped density matches Normal exactly. No score or optimizer update was
added, and neither interface replaces the formal source sampler. The actual
shift5 stochastic steps have DDP buckets318/444/642, deltas
0.126263/0.198413/0.357143 and stds0.052016/0.049801/0.044544.

`qualification/gradient_saved_batch.json` uses only saved real qualification
collection2 at update8: zero new samples or scores. It separates PG and beta50
anchor gradients for steps7/8/9, their cosines, logical channel contributions,
raw/clamped PPO ratios and effective LoRA weight norms. The PG norms were
0.02735/0.04758/0.14913, while weighted anchor norms were
0.0000411/0.0001207/0.0008426. This early acceptance batch cannot establish
the later balance or driving efficacy; the same entry can inspect formal
saved batches without changing their behavior identities.

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

The two actual formal dispatches use the immutable source directory
`/mnt/project/simwam-recipe-20261015/source/12b8e0d75691d7bed04ab02072020e130f711a3b`:

```bash
PY -m torch.distributed.run --standalone --nproc_per_node=8 --module scripts.analysis.simwam_recipe.train --config configs/flow_rl/simwam_source_ddp.yaml --manifest /mnt/project/simwam-recipe-20261015/hard_manifest.json --output /mnt/project/simwam-recipe-20261015/runs/R_SOURCE_seed42_v1
PY -m torch.distributed.run --standalone --nproc_per_node=8 --module scripts.analysis.simwam_recipe.train --config configs/flow_rl/simwam_canonical_ddp.yaml --manifest /mnt/project/simwam-recipe-20261015/hard_manifest.json --output /mnt/project/simwam-recipe-20261015/runs/R_CANONICAL_seed42_v1
```

The first runs on tsinghua-zt4 and the second on tsinghua-zt3; they are separate
single-host jobs, not a 16-GPU job. Do not relaunch an active command. For native
recovery after confirmed worker exit, append `--resume` with that run's actual
sealed checkpoint from `latest.json`, keeping the same source/config/output.
Both were dispatched fresh from SFT, without acceptance weights.

The real finite milestone controller runs:

```bash
PY -m scripts.analysis.simwam_recipe.watch_evaluations --snapshot /mnt/project/simwam-recipe-20261015/source/12b8e0d75691d7bed04ab02072020e130f711a3b
```

It does not restart training or open Navtest. `controller.lock` prevents a
second owner. Real read-only eight-rank aggregation has also run:

```bash
PY -m scripts.analysis.simwam_recipe.summarize_training --run /mnt/project/simwam-recipe-20261015/runs/R_SOURCE_seed42_v1 --output /mnt/project/simwam-recipe-20261015/analysis/R_SOURCE_interim.json
PY -m scripts.analysis.simwam_recipe.summarize_training --run /mnt/project/simwam-recipe-20261015/runs/R_CANONICAL_seed42_v1 --output /mnt/project/simwam-recipe-20261015/analysis/R_CANONICAL_interim.json
```

Final Navtest preparation used the existing complete 12,146-scene / 136-log
input and v1 cache population. Membership and all metadata/cache paths were
checked without generating trajectories or reading scores for selection:

```bash
PY -m scripts.analysis.simwam_recipe.final_navtest prepare
PY -m scripts.analysis.simwam_recipe.final_navtest serve --once --snapshot /mnt/project/simwam-recipe-20261015/source/12b8e0d75691d7bed04ab02072020e130f711a3b
```

The actual one-pass gate returned
`WAITING_FOR_BOTH_ENDPOINTS_AND_DEV_STAGES`, with `Navtest_opened=false`.
The finite final-stage controller will only lock and evaluate actual update20k
exports after both training budgets and registered DEV stages finish. It uses
the same frozen evaluation entry, one seed and both separate ODE grids. Full
Navtest is still pending; asset preparation is not a Navtest result.

Export entry: `scripts.analysis.simwam_recipe.export_policy`; coverage entry:
`qualify_lora`; distributed comparison entry: `qualify_multirank`; exact
resume comparison entry: `qualify_recovery`. `--help` gives their actual
arguments. Exact formal commands, host admission and source archive hash are
in `formal_dispatch.json`. Analysis-only changes after that commit do not
mutate the running source snapshot.
