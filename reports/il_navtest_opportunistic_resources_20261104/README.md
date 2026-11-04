# Revocable idle resources for IL Navtest, 2026-11-04

The user clarified that all idle project resources outside the five fixed
containers may be used, with no known shutdown time. This overrides earlier
resource withdrawals for those additional containers. It does not restart any
terminated experiment or change any training budget or algorithm.

The five fixed containers remain the control/recovery anchors. D IL stays on
the protected eight GPUs of `training-vlawm-tsinghua-zt3`. Temporary resources
are used first for independent, recoverable evaluation shards; no cross-node
training collective was introduced.

## Operational changes

- Updated the shared dispatch policy under its existing file lock and saved
  the prior policy. Explicit project SSH candidates are registered as revocable,
  idle-only resources. Offline or mismatched containers are not usable routes.
- Added an idle-only check to both the scheduler and the worker. Any existing
  GPU owner, even a recognized project process, blocks an additional temporary
  assignment. Workers repeat admission after acquiring their GPU lease.
- Verified SSH container identity, UUID deduplication, measured headroom and
  the existing qualified runtime. Real or unknown processes are never evicted.
- Kept CPU preparation on fixed containers and replaced only the idle CPU
  evaluation scheduler. No learner or active evaluator received a signal.
  New sole scheduler: PID `2611930`, start ticks `1520950281`.
- Retained the 2,000-update checkpoint interval, B1, single candidate, K10,
  shift5 ODE, seed2101, full 12,146-scene population and canonical v1.1 scorer.

## Shutdown and recovery boundary

Native evaluation already writes validated per-scene records atomically to the
shared storage and holds an exclusive rank writer lock. Those completed records
survive a temporary container disappearing; replay verifies identity and hashes
and recomputes only missing work.

The new wrapper marks revocable workers explicitly. On an orderly stop request,
it lets the evaluator finish its current scene. The existing controller accepts
the stopped shard only with matching source/export/rank/identity, a real zero
exit code, and a native scene-boundary receipt. It retains the old attempt and
routes unfinished work to fixed containers; that fallback survives controller
restart.

An abrupt power loss is different: an unreachable SSH endpoint is not proof of
process exit. Missing RESULT/COMPLETE receipts are never fabricated or treated
as success. Recovery must establish exclusive writer ownership before using the
native resume path. Committed scene results remain reusable; work not committed
before the loss may need recomputation. No zero-loss guarantee is made for
uncommitted computation or a server that shuts down without warning.

## Executed evidence

`/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m pytest tests/il_report -q`
passed **77 tests**. Added checks cover an owner appearing after lease acquisition,
unexplained GPU activity, revocable-host return without a fixed deadline,
fixed-container fallback, and refusal to fabricate a lost host's exit evidence.

Real B1 qualification ran one scene on each of:

| Container | Qualification result | Normalized / physical trajectory max error | Submetric max error |
| --- | --- | ---: | ---: |
| training-vlawm-tsinghua-zt4 | PASS, exit 0 | 0 / 0 | 0 |
| training-vlawm-zt2 | PASS, exit 0 | 0 / 0 | 0 |

The reference was the same D68k export's already completed scene, with the same
source, inputs and scene-bound seed. These two extra physical qualification
scorings are costed separately, not counted as a new complete Navtest result.
Cold model loading took about 195 seconds; the one-scene model forward took
1.27–1.28 seconds and canonical scoring took 4.89–4.95 seconds. This does not
establish full-bank throughput or a measured training speedup.

A later repeated full inventory found **23 eligible idle GPU routes**, excluding
the two then-running qualification workers. Routes existed on `training-vla-zt`,
`training-vlawm-zt4`, `training-vlawm-zt2` and `training-vlawm-tsinghua-zt4`.
This is a capacity snapshot, not 23 launched inference workers. Availability is
checked again before dispatch. Other containers were occupied, had unexplained
activity, or were unreachable and were skipped.

Private shared-storage receipts are under
`opportunistic-resource-preparation-20261104/`: authorization and prior policy,
immutable resource grants, admission inventories, exact qualification commands,
per-invocation costs, qualification parity and scheduler handoff. No weights,
images, scene payloads, credentials or large caches are included in git.

At handoff the registered evaluation queue had no active or pending banks.
The next D milestones are 70k, 72k and subsequent 2k boundaries; additional
capacity is available to those registered jobs, without duplicating completed
evaluations or restarting stopped training.
