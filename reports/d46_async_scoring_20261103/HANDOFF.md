# Collection-local asynchronous scoring

CPU implementation ready; no GPU qualification, live deployment or speed claim.
Base scientific source: `82d5f784d02ad4bc182231a267edab8a2a69833e`.

`starVLA.rl.d44_native_rl.async_collection.collect_scenes` accepts the existing
actor, an ordered/lazy iterable of `{descriptor, token, log, metadata?}`, the
existing `RewardService`, CPU generator, recipe, epoch and behavior version.
Optional `before_scene(generator, index)` / `after_scene(generator, index)`
callbacks preserve world16 virtual-lane RNG hooks. The result is an ordinary
scene list compatible with the existing `material_digest`, credit and native
checkpoint validator. There is no new Trainer, scorer, recipe or recovery path.

Only D0, D5, D3_FPO and D3_AWM are accepted. D1/D2/D6 are deliberately rejected.
No runtime currently selects this helper: the synchronous `collect_scene`
remains the default, with `score_submit=None`. The parent integration should
select this path using an audited execution-only opt-in, for example
`execution_optimizations.async_scoring={protocol: d46_fixed_draw_async_scoring_collection_v1,
max_pending: 4}`; do not add it to scientific recipe/parameter identity.

## Minimal integration

Keep the original COLLECTING transaction, behavior version and conditions order:

```python
# Existing before_collection hook runs first. Requests must be lazy if condition
# materialization is currently interleaved with per-scene actor calls.
def requests():
    for j in range(global_scenes // world):
        token, padding = stream.at(state['scene_cursor'], rank, j, world)
        yield dict(descriptor=conditions.get(token), token=token, log=stream.logs[token],
                   metadata=dict(sampling_source='uniform', padding=padding,
                                 beta=dual.beta, expect_fresh=True))

completed = collect_scenes(actor, requests(), service, generator, cfg, epoch, version,
    max_pending=4, cost=attempt.add_cost,
    before_scene=extension.before_scene, after_scene=extension.after_scene)
# First perform the existing all-rank check_errors consensus. On ANY exception,
# leave batch.scenes empty/uncommitted and never compute credit/step/save it.
scenes.extend(completed)
# Existing after_collection, advantages, global std, commit and optimizer follow.
```

The parent Trainer wrapper must retain its normal exception-to-global-consensus
handling around this snippet, including ranks whose local collection succeeded.
Do not directly return a local rank on scorer failure while peers enter DDP.

`AsyncScoringCollector` also exposes `collect`, `finish` and `assert_drained` for
an explicit wrapper. `collect` returns no material. Only `finish` publishes the
whole local collection after every response validates. Its context manager and
failure path cancel queued futures, join running futures and discard all staged
scenes. Do not write Futures to native state. A save hook may call
`assert_drained`; the existing Boundary must still reject COLLECTING even when
the scorer queue is empty. A scorer stall therefore remains a stall to be
handled by existing scorer/worker supervision, without fabricating a checkpoint.

## Ordering and boundaries

- Sample chain and recompute current/reference transitions exactly as before.
- Submit the original request via existing `RewardService.submit`.
- Finish that scene's original FPO/AWM sigma/epsilon MC draws and old energy
  before `after_scene` or the next scene. FPO retains MC4; this helper does not
  change the MC count or the energy implementation. The separate optional fused
  energy helper can replace the unchanged energy block independently.
- Resolve at most four pending scene jobs, in original scene order. Factor
  construction/finalization uses no model or RNG calls. No score filtering,
  reweighting, scene reorder, cross-collection pipeline or stale-policy update.
- Every fixed-arm PG candidate request, physical candidate and scorer-attempt
  counter is unchanged on success. On failure, any extra already-generated
  uncommitted work remains physical attempt cost; no retry is implemented here.
- Generation/scoring wait times can change. `seconds_scoring` and
  `scoring_wait_worker_seconds` measure actual blocking wait at resolution, not
  CPU service latency or hidden scorer work; never compare their sum to GPU time.

## CPU evidence and next qualification

Four targeted tests use the real collector and material digest, a deterministic
CPU actor and a delayed SyntheticService. They cover all four allowed arms,
FPO/AWM MC4, exact request arrays/records/factors/advantages/material digest,
generator/global RNG states, counters and order; genuine overlap plus pending
bound/checkpoint barrier; failed/invalid futures with no partial batch or Adam;
and conditional-arm rejection plus per-scene virtual RNG hooks.

```sh
PYTHONPATH=. /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/d44_native_rl/test_async_collection.py
```

Result: **4 passed in 2.89s**, one existing pynvml deprecation warning. No GPU
allocation, official PDM request, model weight load or live source edit occurred.
Root owns real fixed-draw world16 comparison, source migration and deployment.
Compare the complete original/cached material and all optimizer/RNG/cursor
states before accepting an execution migration. Measure full collection/cycle
wall time and scorer concurrency/memory under the same workload; scoring is
currently about one second per cycle, so achievable speedup may be small.
