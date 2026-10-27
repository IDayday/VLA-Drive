# Latest Qwen-Flow training and Navtest request — 2026-10-27

The three newly registered full Navtest evaluations are pending. No new scores
are claimed. Existing completed scores are IL80k88.157880, IL77-parent RL24.6k
90.689893, and fresh three-frame/current-DINO IL14k82.899980.

Current checkpoint/progress receipts, read at this request:

| Training | Saved updates | Fixed budget | State |
|---|---:|---:|---|
| Original current-frame IL | 98,000 | 100,000 | RUNNING |
| IL77-parent CANONICAL RL | 50,000 | 50,000 | COMPLETE |
| Fresh three-frame/current-DINO IL | 24,000 | 100,000 | RUNNING |
| Separate Qwen-World/PostCoT IL | 35,000 | 100,000 | RUNNING; its independent workflow is preserved |

The completed RL receipt records12500 fresh rollouts,100000 raw scene exposures,
800000 new candidates and50000 optimizer updates, including108 anchor-only
updates. Four inner updates are not counted as four fresh rollouts.

This request pins IL98k, RL77-parent50k and DINO24k before rolling retirement.
Hardlinks preserve native COMPLETE/manifests/optimizer/RNG assets without
modifying original training. RL has native LoRA state rather than a standalone
model.pt; its native files manifest binds the state, and merged export/native
ODE parity is required before full evaluation. Temporal rank0 must pass native
model.predict parity before its other shards start.

Protocol: canonical NAVSIM v1.1; full12146 scenes/136 logs; single candidate;
source shift5 ODE K10; seed2101; no teacher/scorer selection at inference.
Unchanged qualified model/evaluator sources are434cba8(original IL),5a6bb6e
(IL77-parent RL),22abe52(fresh temporal). No training/seed/method change.
Eight native shards per model were selected from16 currently eligible physical
GPUs on fixed tsinghua-zt3/zt4, with qualified two-process sharing on initially
idle GPUs,32GiB admission guard and CPU-loading reservations. Actual dispatch
is refreshed against ownership/headroom. Other real/unknown jobs are preserved;
withdrawn temporary servers, including vla-zt3, are excluded. No filler is killed.

The finite budget is36438 new logical single-candidate inferences, plus separate
export/parity qualification costs. Completed values are reported only after
exact membership, scene seals, official metrics and worker exits pass. Paired
analyses compare RL50k with its exact IL77 parent and RL24.6k; IL98k with IL80k;
fresh DINO24k with fresh DINO14k. Historical IL53-parent RL is not a matched
control. Test observations are disclosed and do not change training or parents.

Six targeted CPU tests and the real sealed registration check passed. The new
report-publication guard preserves porcelain leading whitespace, with an explicit
regression test for the prior conservative publication refusal. Native inference,
reward, action encoding, sampling and optimizer files are unchanged.

Executed entry forms:

```bash
python3 -m unittest tests.rl.test_latest_navtest_request tests.rl.test_finite_navtest_scan -v
/usr/bin/python3 -u -m scripts.analysis.qwen_sourceflow.latest_navtest_request \
  --registration /mnt/project/qwen-sourceflow-native-20261018/latest_NAVTEST_20261027T0100Z/registration.json \
  --report-dir /mnt/project/VLA-Drive-qwenflow-latest-navtest-20261027/reports/qwenflow_latest_navtest_20261027
```

Do not duplicate an existing owner. Queue scheduling advances compute as needed;
the analyzer runs directly once the compute owner finishes, with no separate
result-polling process. Historical measured worker costs, eight shards/model,
cohosting/loading and tail buffers determine the next reading window recorded
under ETA_BASIS.json/NEXT_READ.json in the asset root. No repeated assistant
log/status checks before that window unless the user asks or a failure/completion
event occurs. Scoped report commit/push verifies the remote SHA after completion.
