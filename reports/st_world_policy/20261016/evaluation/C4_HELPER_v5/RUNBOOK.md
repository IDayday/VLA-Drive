# C4 additive helper v5 candidate

These files are prepared, tested candidates, not deployed helpers or launch
approvals. Only Lead may deploy the reviewed exact bytes to `resources/` and
bind a fresh route / explicit completed endpoint before invoking them.

For each fixed job (`C4_step000512_dev1696_seed42_v1`,
`C4_step001024_dev1696_seed42_v1`, `C4_step004096_dev1696_seed42_v1`), Lead must
first read its actual exact COMPLETE receipt, verify canonical receipt identity,
training identity, requested step, world4 inventory and sizes using the existing
admission path, and seal the endpoint plus its exact host/GPU/environment route.
The frozen exporter subsequently verifies full payload hashes before loading.
Do not use `latest`, a prefix population, an in-progress native save, or the
historical status tag as authorization. No route is created by this candidate.

After independent review and exact-byte deployment, the existing CLI is:

```text
/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/launch_formal_dev_export_v5.py --job-id C4_step000512_dev1696_seed42_v1 --checkpoint-tag <EXACT_COMPLETED_NATIVE_TAG> --gpu <PINNED_FREE_GPU_INDEX> --route-sha256 <EXACT_ROUTE_SHA256>
```

Use the corresponding fixed 1024/4096 job ID only at its actual endpoint. One
explicit same-identity `--resume` is allowed only under the existing two-attempt
budget and remaining valid route time; no automatic retries are added.
The resource checks and campaign slot lock must still pass at invocation time.
Training cards are neither shared nor interrupted.

Only after all 1696 predictions and rank0 COMPLETE plus the attempt RESULT
exist, the exporter has exited and the exact references still match:

```text
/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/launch_completed_dev_score_v5.py --job-id C4_step000512_dev1696_seed42_v1 --route-sha256 <EXACT_ROUTE_SHA256>
```

The score launcher retains empty CUDA, two workers, two shared score slots,
rank0 ledger lock, 1728 total calls across at most two invocations, and complete
manifest/native/export identity checks. It runs on the job's same pinned host.
It invokes the unchanged frozen 6714 scorer with campaign run ID
`st-world-policy-20261016`; separate arm/step outputs and ledgers prevent overlap.

Paired analysis commands are already in the new COMMANDS.json. They are metadata
only; Lead must authorize analysis after matched C0/C4 complete exports/scores.
512/1024 remain descriptive diagnostics and never checkpoint selection. There is
no navtest, new control evaluation, additional trajectory, policy change or
scientific result in this handoff.

Reproduce the bounded owner metadata checks (no launcher main is executed):

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/C4_HELPER_v5/VALIDATE_CANDIDATES.py --output <NEW_PRIVATE_TEST_RESULT_PATH>
```

The validator sets its own affinity to two CPUs. It accepts C4 actual metadata,
rejects wrong arm/step/control/source/runtime/launch/approval/pins and guessed
tags, preserves old C3-null rejection, and compares all sixteen old job paths to
v4. The owner result is 153 checks passed; both candidate `--help` calls exit 0.
Neither result replaces independent review.
