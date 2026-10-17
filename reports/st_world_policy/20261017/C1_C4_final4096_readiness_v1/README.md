# C1–C4 final4096 shared-metadata readiness

Observed **2026-10-17 15:56:33 UTC**. This is a read-only snapshot, not a dispatch, remote process inspection, or scientific result. No SSH, GPU query/loader, model/optimizer/RNG tensor read, metric-cache read, training interruption, scoring, or evaluation launch occurred. Lead's local GPU0/1 teacher64 PROFILE remains untouched.

| Arm | Fresh shared status | Completed / 4096 | Latest complete native step | Current training host |
| --- | --- | ---: | ---: | --- |
| C1 | RUNNING | 2717 | 2560 | training-vlawm-tsinghua-zt4-worker-0 |
| C2 | RUNNING | 2704 | 2560 | training-rl-zt2-worker-0 |
| C3 | RUNNING | 2394 | 2048 | training-rl-zt2-worker-0 |
| C4 | RUNNING | 2235 | 2048 | training-vlawm-tsinghua-zt4-worker-0 |

`SNAPSHOT.json` records exact observation times, runtime identities, source pins, latest native receipt hashes, original registration/approval, and the latest actual resume launch. Checkpoint inventory existence and file sizes were checked; large payload checksums were deliberately not reread. The frozen exporter must still perform its full native checksum and strict-load checks before use.

All four root `RESULT.json` files describe **earlier** `BUDGET_PAUSED` allocations at 1700 / 1723 / 1709 / 1740 updates. Their source evidence is retained. Fresh status and subsequent complete checkpoints exceed those steps; the old results must not be used to conclude that current training has stopped. Remote PID liveness was not verified in this no-SSH task.

Each registered allocation retains `max_seconds=28800`; this is not a guarantee that that allocation reaches 4096. If it pauses at its native boundary, preserve the existing lineage and let Lead handle its already-defined continuation. This review neither requests nor performs a restart. No reliable final wall-time ETA is asserted.

## Static dependencies verified

- All four final endpoint registrations are predefined at **exactly 4096**, seed42, matched C0 control, single original Euler10 trajectory, full **1696 unique dev scenes / 16 logs**. They do not substitute the 512/1024 diagnostics for the common endpoint.
- All four use immutable evaluation runtime `/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017`, source `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`. Registration, approval, original launch, current identity, config, preparation pins and prescribed source-file hashes validated. Latest resume launches retain the same registration/approval/source.
- C1 reuses `full_dev_execution_v1`; C2 `full_dev_C2_C3_execution_v1`; C3 its resolved `full_dev_C3_execution_v1`; C4 independently reviewed `full_dev_C4_execution_v1`. The old unresolved C3 preparation is not used.
- Deployed helper v5 is byte-identical to the independently reviewed version: export SHA256 `ad0ee24987bdba9ee3fbc46a5efbdb1445b0a6a69d4d588fc32e960738b876bf`; score SHA256 `2e62c84a2eecde3e9af585d0a6fb77f9ff33d27a2ec76cd0fa2b594b3a8f15c3`.
- Fixed score protocol is `db383e17b887e08beb358b282fbeab223c80c558164bead81d52a45b1b798143`. The existing C0_4096 official-score COMPLETE receipt is present and pinned. No metric index/cache or per-scene score table was opened here.

No missing static source/approval pin was found. No source or helper change is required for these four final jobs.

## Concrete remaining dependencies

For **every** arm, exact `step_004096_*/COMPLETE.json`, final evaluation registration/ROUTE, final export completion and final score completion are absent at this snapshot. Status: **WAITING_EXACT_NATIVE4096_AND_FRESH_LEAD_ROUTE**.

Lead can use the existing final job as soon as its exact native 4096 receipt passes identity, step, inventory and payload validation. `RESULT.json` is not the canonical native completeness receipt: a stale result must not block a valid final native checkpoint, nor qualify an incomplete one.

After the native receipt exists, Lead must seal the existing route schema with actual tag/receipt hash, runtime identity, fresh hostname/GPU UUID/environment, and validity window. Then use an available existing export slot (maximum4), followed by an existing score slot (maximum2) after complete export and verified child exit. Do not reuse an expired diagnostic route. No slot or free-GPU check was performed by this review.

The helper's evaluation allowlist is **training-rl-zt2, training-vla-zt, training-vlawm-tsinghua-zt**. Training placement on Tsinghua4 does not authorize evaluation there under this helper. Preserve active C1–C4 and Lead's local GPU0/1 PROFILE. Fresh host/source/environment/mount/GPU/resource checks remain Lead's launch-time responsibility.

Official scoring remains two CPU workers, at most1728 admissions including bounded retries, retry-attempts1, at most two export/score invocations, and original exact completion/exit checks. No broader resource or scientific matrix is added.

## Existing commands, not executed

`EXISTING_COMMANDS.json` copies the four exact original fixed job/command records and pins; it is not a new runner. Job IDs are:

- `C1_step004096_dev1696_seed42_v1`
- `C2_step004096_dev1696_seed42_v1`
- `C3_step004096_dev1696_seed42_v1`
- `C4_step004096_dev1696_seed42_v1`

On the host named in the newly sealed route, set `JOB`, `TAG`, `GPU` and `ROUTE_SHA` to those **actual** values. Exact4096 tags and route hashes do not exist yet, so this review does not invent them. The already-reviewed entry points are:

```bash
/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/launch_formal_dev_export_v5.py \
  --job-id "$JOB" --checkpoint-tag "$TAG" --gpu "$GPU" --route-sha256 "$ROUTE_SHA"

/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/launch_completed_dev_score_v5.py \
  --job-id "$JOB" --route-sha256 "$ROUTE_SHA"
```

`--resume` is only for a verified prior incomplete invocation of that exact job and identity, within the existing admission budget. Do not add it to a fresh run or launch a duplicate after COMPLETE. A route is an execution dependency, not permission to change inference precision, sampler, solver, population, or score protocol.

## Reproduction and limits

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python reports/st_world_policy/20261017/C1_C4_final4096_readiness_v1/collect_metadata.py
```

Collector limits:2 CPU affinity,1GiB address space,120s alarm,2MiB per metadata file and64MiB total reads. Observed collector reads: **1,739,683 bytes**, execution below1s. It extracts only definitions from the pinned helper; argparse/launch code is never executed, and JSON rereads are also bounded. Only small aggregate metadata, commands and evidence pins are committed, not private per-scene data. Snapshot collection is not a recurring monitor.

All findings are readiness/provenance findings. No early ranking, winner, final statistical comparison, or new training/evaluation authorization follows from this snapshot.
