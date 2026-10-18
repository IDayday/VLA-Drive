# One-candidate consequence SMOKE / PROFILE resource draft

**NOT_APPROVED — no experiment was dispatched and no training approval is issued.**
This independent Systems plan covers only `scene_center_rollout_std_v1`.
The sealed mathematical proposal, actual C1 configuration and historical cost
evidence are pinned in `PLAN.json`. The implementation SHA, per-update evidence,
implementation tests and independent correctness decision are still required.
Feature research remains paused. Existing Base and other projects are untouched.

## Fixed experiment and population

Retain original bound S0; C1 canonical PDMS; four candidate rollouts per scene;
ten Euler steps; MC50/chunk5; original repeat8 BC/dropout; BC/policy weights1;
world4/micro4/caller accumulation2/global32; seed42;4096 learning-rate clock and
warmup100; BF16 execution with FP32 master/Adam/ZeRO2 and TF32 off. Only the
registered advantage normalization changes. SMOKE and PROFILE each start from
the original S0, never from a diagnostic endpoint. Their weights remain
`weights_usable_for_formal_init=false`; retaining evidence does not authorize
formal initialization.

| Stage | Population / exposure | Updates and scoring admissions | Runtime bound |
|---|---|---|---|
| SMOKE | Fixed32 TRAIN scenes chosen before scores;128 scene presentations |4 updates;512 primary official scores;640 total admissions |900s trainer;1200s outer TERM +180s owned-group cleanup;1800s lease |
| PROFILE | Original101,592-scene/1,176-log population and original seed42 sampler, first240 updates;7,680 presentations |240 updates;30,720 primary scores;31,744 total admissions |7200s trainer;7500s outer TERM +180s owned-group cleanup;8400s lease |

SMOKE repeats its small population for correctness only; it is not throughput
evidence. PROFILE does not loop64 scenes. Its first240 global batches precede
the first epoch tail; actual IDs/exposure still must be recorded. Primary totals
are31,232 scores; combined admission ceiling32,384. `retry_attempts=1` remains;
640/31,744 are durable total-admission limits, including retries and uncertain
interrupted admissions. Their headroom is not a separate per-scene retry cap.
Exhaustion fails/pauses before an optimizer step; it never drops labels.

PROFILE performs1,536,000 MC points **for each** of old/current CFM and61,440 BC
repeat presentations. SMOKE performs25,600 MC points for each and1,024 BC
presentations. MC points are not extra official score calls. No evaluation or
DEV/navtest calls are admitted. New on-policy predictions must be scored; the
completed old C1 feedback ledger is analysis evidence, not training replay.

The pinned existing schema already permits `max_updates=4096`, `stop_after=240`,
`sample_limit=101592`, PROFILE timing and warmup100. The old pinned schema rejects
the new method; Lead has begun a concurrent WIP addition, which is not an approved
implementation. Authoritative budget checks use this reviewer's clean `e4e9603`
baseline rather than that WIP. `RegisteredFlowDataset` requires manifest scope
to equal execution scope. Lead must create a metadata-only PROFILE derivative of
the original FORMAL manifest, retaining every entry and order, changing scope
and recomputing identity, with an immutable sidecar binding the parent SHA and
entry/order equality. Do not point PROFILE directly at the FORMAL manifest,
re-run `prepare` over all image/context files, or weaken its scope check. SMOKE
gets a separate32-entry parent-pinned derivative. This reviewer did not open or
rewrite the large manifest.

## Existing resource evidence and what it does not prove

Original C1-profile topology was measured on this same tsinghua-zt4 A800 host.
Historical concurrent4-rank runs A/B each completed240 updates on a repeated64
scene population. Combined199 normal updates averaged16.478371/16.138092s.
A's steady windows were mean16.631493/16.323702s, p50 16.392169/16.362653s and
p95 18.959885/17.069361s. Allocation was4173.40/4098.14s. These are historical
resource references, not measurements of the new normalizer or full-population
IO. Source changes, CPU affinity and co-tenancy are recorded, so differences
cannot automatically be called a causal normalization speedup.

Historical peak GPU allocated14,119,265,792B, reserved about17.0GB. Sum-of-process
RSS peaked153,920,135,168B during startup and about75.7GB in steady windows; RSS
sums double-count shared pages and are not PSS. Roughly9,080 tree CPU-s per100
steady updates included ranks and live nonrank descendants;4 global CPU scorer
workers were used. There is no need to repeat the completed8-versus4 topology
search. The new work must measure its extra failure vote, global int64 count /
FP64 sum / centered-moment reductions and receipt cost.

Original C1's final native receipt plus stat-only inspection reports
11,473,309,360B across10 files; the independent profile measured
11,473,309,292B mature and4,917,271,926B initial native states. Original C1 mature
checkpoint writes were15.37–17.32s. No checkpoint payload was opened here.
Historical240-update feedback/rollout output was874,417,280B/227,720,330B;
that is a planning reference, not an exact new-output prediction.

## Single-host allocation, fresh admission and bounded output

Prefer `training-vlawm-tsinghua-zt4-worker-0`, GPUs0,1,2,3 (UUID pins in JSON),
leaving4,5,6,7 available. At17:04:56UTC all8 were empty. This snapshot does not
reserve them or certify current availability. Immediately before each stage:
re-read policy/authorization; verify exact host, UUID/topology/environment and
clean integrated source; take dispatch/run/per-GPU locks; verify no real/unknown
process owns the four slots and each is <64MiB / <5% utilization; check lease,
mount/quota, outstanding reservations, CPU affinity and available memory.
If occupied, defer; never terminate another job to satisfy this draft.

Request16 logical CPUs for the owned process tree, with the exact available
NUMA-aware cpuset selected and recorded at admission; OMP/MKL/OpenBLAS/NumExpr1,
TOKENIZERS_PARALLELISM=false, existing2 loader threads/rank and4 global scorer
processes. This affinity differs from historical128-CPU eligibility and is an
explicit resource boundary, not a claimed identical CPU benchmark. No second
concurrent trial is requested. Maximum reserved GPU time is
4*(1800+8400)=40,800 GPU-s=11.333 GPU-hours including lease/cleanup headroom;
normal PROFILE is estimated around1.2h from history but not guaranteed.

Reserve **96GiB new shared output**:64GiB native states,8GiB feedback/rollouts,
2GiB controls/logs/profiler,22GiB atomic/partial-write margin. Allow at most six
native directories across both stages (four intended initial/end states plus
two interruption/partial-save allowances); no unbounded retry or cleanup.
Keep all failures and receipts. No teacher/visual cache is built. The17:04
snapshot showed3,644,383,870,976B free on the single shared project mount; do not
sum it across hosts. Admission requires this reservation plus other active
claims and the existing1TiB free floor. Check inode/quota separately; df's
rounded100% used is not a zero-free-byte claim.

Resource stop thresholds for the existing owned supervisor: sampled aggregate
RSS256GiB; fresh available RAM at least512GiB after other claims; GPU reserved
48GiB/rank (leave substantial80GiB headroom); total new-output96GiB and native
directory ceiling above. These are conservative stopping envelopes, not hard
kernel memory guarantees. Record unavailable cgroup/PSI explicitly. Fresh checks
and monitoring must be wired in the actual launch, not inferred from this file.
Use existing process sampling with bounded output (256MiB/stage) and timestamped
identity; no new general resource platform. Limit IO concurrency to this one
run and the existing4 scorers/8 loader threads. Measure logical and physical IO
including startup hashes/model loads/checkpoint checks; do not assume cache hits
or drop integrity checks. Sustained interference with protected jobs pauses this
run at its safe boundary rather than reducing targets, scores or data.

## Measurement and completion criteria

PROFILE is one uninterrupted240-update run: warmup1–20, steady21–120;
warmup121–140, steady141–240. Use existing complete-rank profiler and synchronised
update boundaries. `save_every=240` measures one mature native write in the last
window: report100/99 normal updates plus its complete-window cost, not a silently
filtered100/100. Also report checkpoint amortization at original512 frequency,
initial/final saving and entire launch-to-clean-exit allocation separately.
SMOKE uses `save_every=4`, timing mode off. Both preserve4096 clock / warmup100.

Record every rank and slowest-rank mean/p50/p95, scenes/s, candidates/s, feedback
admissions/latency, loader/H2D and IO waits, old rollout/CFM/current BC/FPO work,
normalizer CPU/CUDA/collective costs, GPU peaks, whole-tree live scorer CPU/RSS,
actual output bytes and checkpoint cost. Module work that overlaps must not be
summed into a fictitious critical path. Throughput windows use timing mode;
existing Flow schema rejects trace mode. Prefer lightweight normalizer ranges /
events resolved at existing boundaries, with overhead stated. No per-update
extra full-model gradient diagnostic, new profiling search or score recompute.

New receipts must bind method/population/old-policy identity; N, zero/nonzero
groups, std and scale; finite state; loss, clipping and available gradient norms;
score validity and exact consumed sample cursor. CPU correctness covers allzero,
tail24 and empty/failing-rank cases rather than choosing a special real batch or
extending240 updates to encounter them. Four-rank SMOKE must establish the actual
new collective path and synchronized failure handling before PROFILE.

Trainer wall/stop limits act at optimizer boundaries. Owned outer timeout may
terminate only its own process group after grace; partial files remain invalid.
On any math/hash/data/score/lease/resource failure, stop and preserve evidence.
No automatic extra invocation, new run ID, call-budget extension or reset clock.
SMOKE may explicitly resume only its exact registration/lineage within total4
updates and remaining shared budgets after review. PROFILE timing rejects
`--resume` by design: interrupted windows are INCOMPLETE, cannot be stitched or
restarted automatically, and cannot claim efficiency qualification. Preserve
native state for diagnosis; a separately approved rerun would spend a separately
declared budget. Completion of these diagnostics does not approve formal4096.

## Remaining minimal wiring, owned by Lead/implementer

1. Seal implementation/tests and full-TRAIN per-update evidence; independent
   correctness review of normalizer, collectives and receipt/resume identity.
2. Produce the scope-only full PROFILE manifest plus parent/equality receipt,
   fixed32 SMOKE subset, final configs/target contracts and new registrations.
3. Reuse the existing single-job portions of
   `resources/profile_flow_tsinghua_2x4_v1.py`; its old source/path, replica search,
   repeated64 population and already-used output names must not be executed.
   Bind only this candidate/new namespace, fresh leases/limits and monitoring.
   `launch_reviewed_flow_v1.py` is FORMAL-only and is not the diagnostic launcher.
4. Obtain exact code/config/lease review and fresh admission, then Lead dispatches
   sequential SMOKE→PROFILE. This report alone supplies no PASS launch receipt.

Actual CLI and handoff commands are in `RUNBOOK.md`; arithmetic/schema checks are
recorded in `PLAN_CHECKS.json`. No GPU, model loading, simulation, DEV or remote
experiment was performed by this reviewer.
