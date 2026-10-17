# C0/C1 first formal pair: exact execution and bounded storage

These commands refer to actual prepared artifacts, not an approval bypass.
Metadata preparation alone does not permit launch. The launcher verifies three
independent reviews, the source/config/data/target/runtime identity, fresh GPU
availability, flock ownership, the lease and separate disk budgets. This file
does not assert that a launch has occurred; use the immutable invocation receipt.

Runtime is `/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017`, commit
`6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`. Do not edit that tree. Both arms
start from the same bound original S0, with a new matched optimizer. No PROFILE
or SMOKE weight is an initializer.

The first pair runs on `training-vlawm-tsinghua-zt4-worker-0`: C0 on GPUs0–3 and
C1 on GPUs4–7. All five planned C arms use world4, micro4 and accumulation2,
global batch32. Do not assign world8 to individual arms to fill another host:
rank-derived RNG, microbatch grouping and reduction order would add an execution
factor to this single-seed control. Subsequent C2–C4 leases are registered when
the matching slots become available; no automatic extra arm is launched here.

The fixed complete train manifest has101592 scenes/1176 logs. Its SHA256 is
`b810cafbdaa42afd8554c08be443779e7d66edce56292c0c30940e728c0ed656` at
`/mnt/project/st-world-policy-20261016/audits/flow_policy/full_train_contexts_v2/TRAINING_MANIFEST.json`.
Both existing metric-cache roots are required and reused in place:

- `/mnt/project/constrained-flow-noise-20261014/metric_cache_v1`
- `/mnt/project/flow-safe-improvement-20261013/metric_cache_v1`

New teacher, condition and metric tensor cache allocation is **zero**. Configs
preserve four actual Euler10 actions/scene for policy arms, MC50, original BC
repeat8, all scenes and valid denominators. C0 executes no scorer service. The
shared official runtime adaptation is BF16 execution with FP32 master/Adam state,
ZeRO2, FA2 and TF32 off; no precision search occurs in this launch.

## Boundaries and cost

Each arm has `max_updates=stop_after=4096`, warmup100 and the original4096 cosine
clock. Save every512;512/1024 are diagnostics, not scientific elimination points.
The24-scene tail is retained. Exact endpoint exposure is131064 scenes per arm,
1048512 BC-repeat presentations. A policy arm needs524256 primary scoring calls.
Its hard ledger cap532448 permits8192 extra admissions if the full primary set
completes. There is no separate8192-retry counter: earlier failures can consume
later primary budget, causing failure before backward. `retry_attempts=1` does
not authorize unlimited retries or a budget increase.

The immutable lease is48h; an invocation's28800s training allowance plus780s
save/cleanup grace must fit before lease expiry. At most six invocations are
admitted, but **six complete8h segments do not fit into48h once overhead is
included**. Do not edit a lease/config/approval in place. A new expired-lease
continuation needs its affected identity and compatibility review.

Cache storage retains the768GiB cap and1TiB filesystem free floor. Checkpoints
are separate. The conservative C reservation is187GiB for C0 and209GiB for
each policy arm, including pause/partial files and logs. Existing W reservations
remain960GiB. The launch admission additionally reserves123GiB of remaining
cache claims; it never deletes prior outputs.

On resume the shared resolver verifies the latest compatible complete native
checkpoint, including payload checksums. The launcher counts **all** existing
checkpoint directories, including partials, and reserves every remaining512
boundary/final save plus a possible nonperiodic pause and partial write. More
than15 total directories requires storage review; it does not trigger cleanup.
The selected resolution and this arithmetic are saved per invocation. The native
trainer independently validates the actual checkpoint when loading it.

The separate observer is stdlib `/proc` only, every10s, with a strict256MiB file
cap including its footer. Time is a cooperative monotonic limit between samples;
one proc sample can finish after the deadline. It sends no signals and has no
GPU context. Report10s sampling-boundary error, RSS sum as an upper bound, and
live-tree cumulative CPU without summing cumulative samples. It is not the
PROFILE-only one-second observer.

## Initial launches

Run only after all actual reviews and complete profiles are present. The helper
itself refuses a missing, stale, nonindependent or non-PASS review. It also refuses
an existing output directory rather than implicitly restarting it.

```bash
ssh -o BatchMode=yes training-vlawm-tsinghua-zt4 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/st-world-policy-20261016/resources/launch_reviewed_flow_v1.py \
  --registration /mnt/project/st-world-policy-20261016/formal/consequence/registrations/C0_fulltrain_seed42_official_4096_v1/REGISTRATION.json

ssh -o BatchMode=yes training-vlawm-tsinghua-zt4 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/st-world-policy-20261016/resources/launch_reviewed_flow_v1.py \
  --registration /mnt/project/st-world-policy-20261016/formal/consequence/registrations/C1_fulltrain_seed42_official_4096_v1/REGISTRATION.json
```

Exact outputs are under
`/mnt/project/st-world-policy-20261016/formal/consequence/runs/` with the two
run IDs above. Each registration keeps the original `LAUNCH.json`; subsequent
invocations get their own immutable `invocations/<actual_timestamp>/LAUNCH.json`
and log. Timestamp names are generated by the launcher, not copied from this
document. A root `DISPATCH.lock`, exact GPU flocks and the trainer's run lock
protect against duplicate work. No auto-followup is installed.

## Resume the same run

After verifying that the prior invocation has exited and the latest native
receipt is complete, use the same configuration/approval and original lease:

```bash
ssh -o BatchMode=yes training-vlawm-tsinghua-zt4 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/st-world-policy-20261016/resources/launch_reviewed_flow_v1.py \
  --registration /mnt/project/st-world-policy-20261016/formal/consequence/registrations/C0_fulltrain_seed42_official_4096_v1/REGISTRATION.json \
  --resume

ssh -o BatchMode=yes training-vlawm-tsinghua-zt4 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/st-world-policy-20261016/resources/launch_reviewed_flow_v1.py \
  --registration /mnt/project/st-world-policy-20261016/formal/consequence/registrations/C1_fulltrain_seed42_official_4096_v1/REGISTRATION.json \
  --resume
```

Only an explicitly acknowledged existing operator stop additionally uses
`--acknowledge-stop`. A wall-budget pause needs no operator-stop acknowledgement.
Do not erase STOP evidence, reset the sampler, restart warmup, load model-only
weights, clear failed feedback ledgers, or resume PROFILE directories. MAX_STEPS
is terminal and cannot consume another optimizer step. Arbitrary crash replay is
not promised: a failed unsaved policy version can leave charged admissions, and
its request-identity guard may reject a numerically changed replay. Preserve the
failure and use the documented native-boundary recovery process.

## Evaluation and scientific scope

Full dev remains1696 scenes/16 logs; the fixed log-bootstrap protocol and invalid
prediction rules are inherited. The planned512/1024/4096 diagnostics have their
own immutable checkpoint exports and evaluation budgets; the training feedback
cap does not silently authorize them. No navtest or ACTIVE update is part of
these commands. The S0 parent has already seen the train population; this wave
is mechanism exploration, not newly unseen-data foundation-model training.

Flow code, full-data metadata, correctness, efficiency, formal progress and
scientific effects remain separate statuses. The user-facing cache decision
does not claim a learned planning gain or completed temporal on-demand provider.
