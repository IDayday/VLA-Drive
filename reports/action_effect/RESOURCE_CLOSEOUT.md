# Resource closeout — action-effect campaign

Source start: `36629e67`. Live administrative checks used the configured,
host-key-verified SSH aliases `training-rl-zt2`,
`training-vlawm-tsinghua-zt2` and `training-vlawm-tsinghua-zt5`. Current server
dispatch policy and applicable AGENTS were read first. No withdrawn host was
used. Worker-name direct aliases initially failed host-key/DNS checks; no trust
checks were bypassed, and the configured aliases succeeded.

Final live snapshot: **2026-10-20 04:17:11 UTC**.

| Host | Campaign processes still live | Original monitor | Pressure workers | Restoration state |
|---|---:|---|---:|---|
| rl-zt2 | 0 | Running, unpaused | 7 | Existing monitor retained |
| tsinghua-zt2 | 0 | Running, unpaused | 0 | Original monitor resumed; fresh1800s observation |
| tsinghua-zt5 | 0 | Running, unpaused | 7 | Existing monitor retained |

This is a live `/proc` scan for this campaign's immutable artifact paths and
`tools.action_effect` commands, including training, inference, native scoring,
qualification and queue/controller processes. Separately, recorded launch PID
and start-tick identities were checked:61 rl-zt2 launch receipts and24 receipts
on each Tsinghua host had no matching live process. PID presence alone was not
used to identify or signal a process. Unknown or unrelated real tasks were
untouched; the pressure-worker counts do not imply every GPU is available.

The finite G1 queue, both route fleets, native-evaluation queues, temporal-eval
waiter, online qualification queue and final-comparison controllers have no live
campaign process. Their original source files remain for provenance and do not
represent a running queue. Completion metadata confirms:

- 23 actual training RESULTs are COMPLETE at1000 optimizer updates:6 main G1,
  16 route arms/seeds and1 no-temporal G1 control. Small RESULT pins were checked;
  no weights were reopened or training resumed.
- The current native queue records31/31 COMPLETE jobs. No bank/native feedback
  was re-simulated or re-audited as part of resource closeout.
- The original strict final DEV comparison exited0 after595.908296s for18
  requested methods plus S0. Its v2 controller found the already completed v1
  output and did not create another comparison or signal its child.

## Original pressure restoration

Initial04:13:49 inspection found rl-zt2/ts5 original monitors alive. ts2 had a
genuine restoration gap: its original agent had exited and the persistent
`/tmp/zt-idle-pressure-v1/PAUSED` flag remained. The original control API is
`resume`, not an ad hoc replacement `start` command. Lead was informed of the
gap, inspected exact command/source hashes and authorized restoring the same
original monitor only.

At04:15:12 UTC the only state-changing closeout action was:

```text
/tmp/zt-idle-pressure-v1/pressurectl resume
```

Before that command, the existing controller lease was fresh (9.8s old), all
eight ts2 GPUs had0MiB/0% and no compute-app owner, and no agent process was
present. `manual-start.json` contained expires0 and an empty UUID list, so no
one-shot immediate-admission request could bypass the1800s window. Source
hashes were checked again immediately before invoking the original API:

```text
pressurectl e1605b1be3ecddbbd9f4032a092be90329764923e9896bbfedc867e0d7d3e24d
agent.py    fef1275969dcc4406c59e8eef4e7009a300352a391b0a42a9d2adfe5baacdcc1
control.py  a56e5318b4eadcf2142e9ab9ed53429dc4fdfd6aa21ddfc28824394177b9912a
```

The original captured pressure-release receipts bind the same `pressurectl`
hash. The agent source hash also matches both still-running original monitors
on rl-zt2 and ts5. The API only removed PAUSED; the existing external controller
then launched the original `agent.py` without `--worker`. No replacement code,
new filler implementation, direct worker command or GPU reset was introduced.
No signal was sent to any process during this closeout.

At04:15:50 the restored ts2 monitor reported WINDOW1800, threshold15%, unpaused,
valid controller lease,8 observation windows at0s and0 workers. At04:17:11 its
windows were83s, still0 workers. The original code starts a fresh LoadWindow,
requires a complete1800s low-load window, resets on gaps/manual worker exit and
protects unknown GPU owners; no immediate filler was started. A future owner or
load change can defer admission. Completion of the full idle window is therefore
**IN_PROGRESS**, not falsely reported as eight pressure workers restored.

rl-zt2/ts5 retained their original monitors and seven existing workers each;
they were not restarted or duplicated. Both reported WINDOW1800, threshold15%,
unpaused state and a fresh valid lease. Their pending GPU windows remain under
that existing policy. There is no remaining missing-monitor restoration gap.

## Evidence retained outside Git

```text
/mnt/project/action-effect-repair-20261020-budget-v2/resources/closeout_20261020T0413Z/
  CLOSEOUT.json
  SUMMARY.json
  training-rl-zt2.stdout.json
  training-vlawm-tsinghua-zt2.stdout.json
  training-vlawm-tsinghua-zt5.stdout.json
  training-*.final.json
  ts2_original_monitor_resume.json
  ts2_original_monitor_after.json
```

Final `CLOSEOUT.json` SHA256:
`5f6c653419d7cd98c1692e995d4f8afbc25c2788fdea0d15f59f7fa642d8a5ce`.
It binds the live host summaries,23 small training result pins, native-queue
completion, comparison exit and exact original-monitor restoration receipts.
Raw process inventories stay only in that external directory and are not
committed. No credential/config/key contents are included in this report.

The actual online and supplemental CPU-cost evidence also remains intact:

```text
/mnt/project/action-effect-repair-20261020-budget-v2/qualification/online_S0_A1_B1effect_final1000_seed42_v2/online/{COMPLETE.json,WORKER_RESULT.json,DEPENDENCY_AUDIT.json}
/mnt/project/action-effect-repair-20261020-budget-v2/qualification/cpu_operator_cost_cache_v1/{COMPLETE.json,AUDIT.json}
```

The original online receipt's cached comparison remains NOT_REQUESTED;
the later CPU supplement independently records exact same-scene S0 raw/physical
and A1/B1-effect physical cache agreement. Its profiler numbers remain supported
operator FLOPs lower bounds, not whole-VLA FLOPs or scientific planning gains.

No new training, inference, simulator or GPU qualification was dispatched.
No checkpoint/cache/report was removed. The active/archived Base, its weights,
the locked22k IQE identity and formal ACTIVE were not changed. Scientific code
and the immutable `64b115d2` online deployment worktree remain untouched.
