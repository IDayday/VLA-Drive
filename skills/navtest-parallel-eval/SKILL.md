---
name: navtest-parallel-eval
description: Run complete NAVSIM v1 Navtest with available GPU processes across servers and concurrent canonical CPU PDMS scoring; use for checkpoint evaluation, resumable sharding, or speeding up Navtest without changing model inference or evaluator semantics.
---

# Navtest parallel evaluation

Deliver complete, attributable Navtest scores. Reuse the project's native prediction
path and pinned canonical scorer. Scheduling and sharding may change; the selected
trajectory, sampling, precision, population and score definitions must not.

## Establish the contract

Locate the actual training source, complete native checkpoint, current-only input
manifest, full metric-cache index and canonical environment. Read the current
dispatch policy and latest user instructions before selecting hosts. Keep old
controllers and registrations as historical evidence; explicitly supersede their
evaluation routing when requested. Do not launch a competing legacy observer.

For this project's locked v1 population require exactly 12146 unique scenes and
identical token/log sets for inputs, predictions and metric index. Export q0 and
q_final when the model supplies both; the latter is the main executed trajectory.
Maps, future labels and LiDAR are scoring assets, never additional model inputs.

Pin the model source and checkpoint hashes, input identity, sampling seed,
coordinate/heading convention, horizon, evaluator source and numerical runtime.
Retain the native FP32-master restore and strict state loading, FP32 inference,
TF32 off, original ten-step FM, one candidate and scene-bound proposal noise for
the structured-world runs. A shard index must not enter the scene seed.
Read [the runtime contract](references/runtime-contract.md) for this installation.

## Use available hardware efficiently

GPU count and shard count are derived from currently usable resources for each
dispatch, never fixed by the skill. A past deployment of four hosts or 32 GPUs
is evidence for that run only. Do not require a whole idle eight-GPU server for
inference: suitable individual GPUs can participate. Refresh availability before
each queued checkpoint; choose CPU concurrency from actual spare CPU capacity
and scoring backlog as well. If resources change during a run, finish or recover
the owned work units without changing scene identity or redoing successful work.
Use [the scheduling details](references/scheduling.md) when preparing routes,
recovering an interrupted attempt or coordinating pressure restoration.

Inventory authorized hosts in parallel. Compare physical GPU UUIDs across
containers, not container names. Sample utilization and free memory more than
once; a momentary zero is not proof a device is idle. Check the exact worker
Python imports, model paths, disk, CPU capacity and current-image availability.
Keep runtime overlays isolated from active training environments.

Use all suitable unique GPUs. Start one model process per selected GPU and
profile its real peak memory and scene rate. Share memory with existing work only
within the user's authorization and measured capacity. Do not kill genuine
training; confirmed occupancy or pressure scripts may be terminated only when
already authorized. Record why a host or device is unavailable instead of
counting it as deployed. Recheck before dispatch and when load changes.

Measure interference with existing useful work as well as inference speed.
Available memory alone does not establish spare compute capacity. If cohosting
materially slows an active training job, route subsequent work to idle devices
and record the actual slowdown. Drain owned inference at a safe work boundary
when supported; preserve valid outputs and never terminate genuine training.

When the user authorizes pressure-script preemption and restoration, capture
the exact script, argv, working directory, user, process start time, descendants
and GPU binding before stopping the confirmed filler. Keep any required private
environment locally; do not publish credentials. Use PID/start-time checks and
targeted signals, never broad Python or GPU-use matches. If the script spans
multiple GPUs, release its complete parent and account for every affected card.
After the actual inference workers exit, restore the original filler on the
released GPUs with its original resource limits. Verify the replacement process
and GPU binding. On failure or cancellation also restore released fillers once
the worker processes have exited. If another genuine task acquired the card,
defer restoration until it is available and report that state. Do not create a
new filler when none was stopped, or occupy cards still serving queued useful
work. The existing project pressure guard recognizes the exact
`/mnt/project/gpu_stress.py` script; extend recognition only after inspecting an
explicit alternative. Pressure-script policy does not classify low-utilization
real training as disposable.

Use independent scene shards or a leased shared queue, without a cross-host
NCCL barrier. Reuse the native encode/proposal/refine functions rather than
reimplementing the model. Keep the full semantic identity common to all shards;
store host/GPU/shard placement separately. Allocate work by measured throughput
or small reclaimable chunks so a slow host does not determine the finish time.
Avoid simultaneous model-master reconstruction storms and repeated checkpoint
hashing. A validated, immutable FP32 export may be shared by workers if it is
strictly bound to the native checkpoint and matches native restoration.

Stage shard images and calibration locally or use bounded prefetch when shared
storage limits throughput. Preserve pixel hashes, ROI, calibration and legal
ego-state encoding. Batch size, AMP, TF32, candidate count or sampler changes are
not scheduling optimizations: retain the established path unless separately
validated and authorized. Measure prediction parity on fixed real scenes before
accepting a new worker wrapper; compare q0, q_final and out-of-range flags.

## Stream CPU scoring while GPUs infer

Write each prediction atomically, followed by a record binding its token, model
identity and file SHA256. CPU workers consume only complete verified pairs. Run
the existing canonical initialize/score functions in a bounded process pool;
score both trajectories as their scenes become ready. The CPU stage must not
wait for the entire GPU bank. Separate Qwen and canonical Python environments.

Reuse log-local caches and process initialization. Begin with available CPU
capacity after allowing for active training and image preprocessing; increase
workers only while ready-scene backlog and measured throughput justify it.
Distribute CPU shards across hosts if CPU scoring is the bottleneck, using the
same verified runtime. Use atomic claims or exclusive shard ownership so multiple
GPU/CPU processes cannot publish conflicting results. No changes to PDM simulator,
scorer weights, interpolation or responsibility rules are permitted.

Resume successful predictions and score records after verifying identity and
hashes. Recover only expired work leases. Infrastructure retries reuse the same
scene seed and parameters; scientific errors stop the affected dispatch and are
reported. Keep failed scenes in the denominator, mark incomplete/failed results
invalid, and never issue a full-population COMPLETE marker for a shard or a
partial bank.

## Finish and report

Before merging verify exact scene membership, no duplicates, all payload hashes,
and both trajectory scores. Aggregate per scene; do not average shard means.
Save full metric components, zero scores, failures, costs and executed commands.
Compare only matching populations/protocols; record actual checkpoint updates.
Use log-cluster bootstrap with pooled scene weighting for paired differences.
Inference seeds and bootstrap draws are not training seeds. Intermediate Navtest
use requested by the user must be disclosed in subsequent research interpretation.

Report actual hosts, unique GPUs/processes, completed/total scenes, concurrent CPU
progress, elapsed time and GPU-hours. Distinguish measured speedup from estimates
and theoretical limits. Continue the pipeline to full completion or accurately
report the remaining blocker. Publish the aggregate results and reproducible
commands without private images, caches, weights or keys.
