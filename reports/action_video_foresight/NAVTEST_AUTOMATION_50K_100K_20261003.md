# S0–S4 training-time Navtest automation deployed

The user explicitly requested each experiment at 50k and every 10k thereafter,
without waiting for training to finish. The independent observer is **actually
running**, PID `3679175`, with **30 registered tasks**: S0–S4 ×
50k/60k/70k/80k/90k/100k. At deployment all tasks are WAITING for exact checkpoints;
S4 is approximately47.8k. This is deployment evidence, not completed Navtest scores.

Observer source: `dd1df3b3d5b6d571226258e0cb2b33cb77d0e107`.
Unchanged training and model/metric evaluation source:
`1493deda247107efe076b9294863a6753391298a`.
Branch: `feature/action-conditioned-video-foresight-20261002`.
The observer source was pushed and its remote SHA verified before this report.

Registration identity:
`e8ff6893731b857a65cd18340002e99107ce039f58c2e31d08e404f5254693d6`.
Artifacts: `/mnt/project/action-video-foresight-artifacts/20261002/navtest_50k_100k_20261003`.
Launch receipt, command, real host checks, registration asset hashes and initial
state are in [the evidence JSON](NAVTEST_AUTOMATION_50K_100K_20261003.json).

## Behavior and protocol

The five-second observer preserves exact COMPLETE checkpoint shards with hard
links before rolling retention removes60k/70k/80k/90k. There is no change to the
running trainer, its source, its save rules, or its optimization. Each available
checkpoint queues immediately while training remains RUNNING. The model's host
can share its eight GPUs with capped exporters. The existing DEV evaluation lock
serializes export waves; inadequate headroom postpones only evaluation. No
unrelated or pressure processes were stopped.

Scoring runs locally on the canonical host, at most2jobs×16CPU workers. This
avoids the earlier remote-host SSH-alias failure. The full12146scenes/136logs,
official environment, full-precision map/reference cache, FP32-master restoration,
FP32 compute/TF32off, seed42, singlecandidate and original10FMsteps are retained.
Deployment removes all auxiliary heads and GT inputs. The inherited checkpoint
field C1 is a visual configuration; experimental S0–S4 identity is validated via
the formal registrations and checkpoint run hashes.

Results include all scene rows, failures, PDMS/submetrics, zero scores and ego
ADE/FDE/yaw. No invalid scene is dropped. A completed task is not rerun; resource
waits and incomplete time-bounded exports resume from their saved artifacts.
An invalid or interrupted task retains its failure and does not stop other tasks.
The earlier final-only Navtest timing is superseded by this user request; the
common100k training endpoint and development-based selection remain fixed.

## Actual verification

* **40 targeted CPU tests passed**, including exact checkpoint preservation,
  no substitution, duplicate exclusion, live-training launch, stop behavior,
  S-arm identity, unchanged exporter lock compatibility, and complete scoring
  validation against previously completed real12146-scene results.
* **5/5 host preflights passed**: legal connection, frozen source/asset hashes,
  GPU headroom and nine real current images per host. No model inference or
  optimizer update was performed by these preflights.
* The observer was launched detached, its command/PID and fresh heartbeat were
  checked. A second actual observer invocation failed at the expected flock;
  the original continued running.
* Initial test collection hit duplicate test basenames and was fixed by renaming
  the new test file. One registration attempt ran before worktree checkout had
  finished; it wrote no registration and was retried after checkout completion.
  These were setup errors, not training/model failures.

Shared storage had about2TiB free at setup. Each checkpoint is about35GiB; holding
the20 otherwise rolling snapshots can retain about700GiB. The50k/100k hard links
share storage with already permanent milestones. No weights, cache contents,
raw data, private scene images or keys are included in the commit.

## Resume after an actual observer exit

```bash
cd /mnt/project/VLA-Drive-action-video-navtest-source-20261003
/root/miniconda3/envs/ddp/bin/python -u -m tools.action_video_foresight.navtest_milestones watch \
  --registration /mnt/project/action-video-foresight-artifacts/20261002/navtest_50k_100k_20261003/registration.json
```

Do not start this while the observer remains live. Status and aggregate scores
are `status.json`, `TASKS.json` and `SUMMARY.csv` in the artifact directory;
each `jobs/S*_*/result.json` denotes a validated complete evaluation.
See [the operating instructions](../../docs/ACTION_VIDEO_NAVTEST_AUTOMATION.md).
