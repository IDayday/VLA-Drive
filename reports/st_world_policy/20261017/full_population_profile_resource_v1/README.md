# Independent resource review: full-population world2 PROFILE

**PASS_CODE_RESOURCE_PROFILE64_ONLY.** Reviewed implementation `a6cd73920816816cbaa13544cf898002445b9981`, integrated at Lead `4ca82434`; runtime remains `a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a`. The actual controller and INPUTS hashes are pinned in REVIEW.json. This reviewer is independent of the controller implementer.

Ten independently written CPU tests passed: 0.084 s process wall, sampled RSS 25,571,328 bytes, last observed logical reads 2,225,128 bytes. Test process affinity was CPU8/9. No GPU, model/checkpoint contents, teacher tensors, large database, actual signals, ledger changes or dispatch were used. The author's 14 tests are not counted as our evidence. TEST.log preserves each case and TEST_RESOURCE.json records this test process's scope and limits; `/proc` samples are not a hard process RSS maximum.

The positive exit-race case removes a stale leader snapshot after `leader_done`; the negative case retains a real owned orphan and returns failure. Further cases cover permanent RSS/storage failure despite rc0, unknown cleanup membership, 180 s cleanup clipped by the absolute end, no double addition of rank I/O telemetry, output-only config migration, partial-slot accounting, cumulative 64/128 resume boundaries and rejection of the unsigned template. These fixtures exercise the new composition, not real torchrun ranks. Mature PPID/starttime/pidfd ownership and signaling proofs are inherited through exact source pins rather than reimplemented or retested broadly.

The existing metadata factory completed separately; its independent terminal review is `../full_population_provider_preparation_terminal_resource_v1/REVIEW.json`, SHA `da2bfb4a547cb93d9543b81e9f36db5c0d0fd39bd0572a0c6fe28d68205a126b`. No new database/PCA validation was performed here.

## Admitted initial invocation

`COMMAND.json` contains the exact Lead-path prepare/run argv and pinned ADMISSION.json. It grants one initial preparation and one initial invocation, on GPU0/1 and CPU0–3, OMP/MKL/OPENBLAS=2. Initial admission does not authorize resume. The reviewer checked the actual small admission parser accepts prepare/run and rejects resume, without invoking common(), prepare(), execute() or GPU discovery.

The fixed bound remains full original TRAIN population, world2/global2, cumulative64 updates/128 sample exposures, native1500 s, outer1620 s and cleanup180 s. Original20k scheduler/warmup1000 is preserved. A separate 80 GiB reservation covers at most two full or partial native checkpoint slots across the initial and any separately admitted resumed portion. It preserves old2303 GiB commitments and a separate1 TiB free floor. Preparation derives only `runtime.output_root`, sending checkpoints into the independent profile namespace instead of the32 MiB metadata factory namespace.

The installed ZeRO2 writer saves one rank0 full model and partitioned Adam/master state per rank. Prior world1 small receipt totals37,365,927,466 bytes for tensor files; two such slots leave11,167,490,988 bytes below80 GiB before small world2/rank overhead. This is source-supported planning evidence, **not measured world2 checkpoint size**. No checkpoint tensor file was read. The space reservation is advisory, not a filesystem quota.

Fresh statvfs showed6,625,796,268,032 available bytes against3,658,238,394,368 required before this new output. Existing immutable CP receipt expires20:22:33 UTC. The new admission ends earlier; see its exact timestamps. Lead must freshly recheck GPU idleness/UUIDs, source pins, dispatch policy, output absence, locks, one-loader sequencing and free space at prepare/run. This report does not authorize stopping another task.

## Remaining boundaries

- Aggregate process RSS is sampled every1 s and own output/free-space every10 s, including cleanup discovery. These trigger stopping; they are not hard real-time quotas and cleanup may exceed a trigger.
- Terminal parent `/proc` counters include waited descendants once. Rank telemetry is separate. There is no new exact all-rank real-time I/O cap; startup/preparation and monitor counters must retain their stated scopes.
- Actual world2 startup, teacher service rate, VRAM/RSS/checkpoint peak, full-population throughput and continuation remain unmeasured. This is not formal global32 qualification or scientific evidence.
- Prior T1 live descendant-monitor failure and the prior W extraction supervisor failure remain failures. No old receipt was rewritten.
- The immutable lease is not extended by resume. Completed64 cannot be resumed to128, and a third checkpoint directory is refused even when an earlier slot is partial.

Only the Lead dispatches. The source/review commit does not itself run the admitted job or prove it successful.
