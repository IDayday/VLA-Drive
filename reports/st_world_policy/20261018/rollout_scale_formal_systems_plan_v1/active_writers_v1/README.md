# Active writers and remaining storage — 2026-10-18

This is a read-only resource/accounting review, **not formal launch approval**.
The sole proposed C1 rollout-standard-deviation run still needs complete
PROFILE240 evidence and exact final launcher/monitor/registration approval.

At 18:08:19 UTC, tsinghua-zt had eight protected native Qwen/sourceflow IL ranks,
using approximately 75,797–75,953 MiB each. The owner is
`/mnt/project/qwen-sourceflow-native-20261018/runs/IL_seed42_v2`, source
`d4d0012769540af8378e013c072d6d8053d9f275`. The latest complete checkpoint seen was
1000 updates; the bounded log tail showed full-model training at 1120 updates.
We did not stop, signal, reconfigure, or read model contents from that run.

Its registration specifies 100,000 updates, checkpoint every 1000, five permanent
milestones and two rolling recovery states. The actual save/retention call chain
publishes atomically before retiring eligible old payloads. The registered full
checkpoint estimate is 38 GiB. File metadata for its completed full-model
qualification checkpoint totals 40,177,620,333 bytes (37.418 GiB), below that
estimate. The live 1000-step checkpoint is only 13.960 GiB because it precedes the
full-parameter Adam state: using that smaller size for future storage is wrong.

A conservative future simultaneous occupancy allowance is:

```
ceil((8 * 38 GiB - 14,989,269,433 already occupied bytes + 1 GiB small artifacts) / GiB)
= 292 GiB
```

Eight covers five permanent, two rolling, and one atomic publication in flight.
This is peak retained/in-flight capacity, not cumulative write traffic. Existing
files are already charged by filesystem free space. We do not double-count them,
assume their immediate removal, or change the other project's retention policy.
The 38 GiB number is a measured-supported registration estimate, not an enforced
limit on that external process. Fresh admission and owned-run monitoring must
retain a safety margin if it grows beyond this projection.

The resulting proposed free-space admission is **1529 GiB**:
1024 GiB shared floor + 209 GiB new formal run + 4 GiB unresolved old small claims
+ 292 GiB active IL remaining. At the local 18:11:30 snapshot, free space was
3,544,741,150,720 bytes, leaving 1772.30 GiB above this combined reservation.
All sampled nodes see the same project device; host free values are not summed.

The running approved PROFILE on tsinghua-zt5 GPUs 0–3 is protected. Its
SMOKE+PROFILE allowance remains active until it finishes and all owned processes
exit. Formal must wait for those exits; afterwards PROFILE files are occupied
space, with no duplicate future 96 GiB allowance. zt2/3/4/6 and rl-zt2 were empty
in this snapshot. The failed `training-vla-zt` SSH alias was resolved by read-only
local inspection of `training-vla-zt-worker-0`; its eight GPUs were empty.

The local historical G2 controller is alive but `IDLE_AWAITING_REQUESTS`, with no
active keys; every registered cadence endpoint through 100k is COMPLETE. It and
the historical lock holder remain untouched. No currently active large writer
was attributed to them. A new request or a different writer invalidates this
point-in-time inventory and requires refreshed accounting.

`SOURCE_PINS.json` points to sanitized raw inventories kept outside Git and to
small config/receipt/source evidence. `ACTIVE_WRITERS.json` is a condensed report,
not a raw process inventory. `CHECKS.json` contains eight metadata/arithmetic
assertions, all passing. No model, optimizer, scene, cache, DEV or Navtest score
payload was opened, no experiment was launched, and no scientific score was read.
