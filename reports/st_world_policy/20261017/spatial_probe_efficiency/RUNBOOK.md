# Bounded endpoint qualification commands

These are the actual inspected CLI interfaces. **No command in this file was dispatched against real data.** The actual 5000 endpoint and its cache signature do not exist yet in this review. The variables intentionally require Lead-supplied real bindings; there is no invented checkpoint path or default `latest`. `CLI_HELP.json` records successful help calls.

Lead supplies a clean immutable runtime and the following reviewed files:

- `ST_PROBE_RUNTIME`: extraction/fit worktree whose HEAD equals the binding/registration source SHA. Reviewed code source is `7db26ae4e89d7c13a63130e3954b31dd4514cdd6`; report-only integration can inherit matching executable hashes with an explicit record.
- `ST_W_BINDING`: sealed `st_spatial_w_binding_v1`, pointing to the actual original formal registration, exact native5000 tag/identity, the fixed train64 ordev16 manifest, and correct observation identity. `max_seconds=900`, canonical FP32 runtime and one leased GPU.
- `ST_W_ROOT`: the exact fresh W root in its approved **typed tensor** claim. The existing metadata-only admission does not support this yet.
- `ST_STORAGE_POLICY`: the reviewed policy using the existing shared ledger,768 GiB cap, separate ≤1 GiB W package, and unchanged4 GiB metadata package/free-space commitments.
- `ST_FIT_REGISTRATION`: sealed `st_spatial_probe_fit_v1`, same actual arm/5000 train+dev caches, pinned protocol file SHA `7e2dcd4a938dba575f4000f641f59e2cd48a80ad1ece190b0a31000596fc956e`, CPU device,1200-second cooperative limit and64 MiB output cap. No changes to300 cycles or recipe.
- `ST_FIT_ROOT`, `ST_PROBE_LOG_ROOT`: separately budgeted probe checkpoint/output and bounded resource-log roots.
- `ST_CPUSET`, `ST_GPU`: fresh allowed-host CPU/GPU lease; one GPU index must map to the recorded UUID. The in-process device in the binding is `cuda:0` after masking.

Before any real command: check current dispatch policy, quiescent exact endpoint, actual runtime/source/config/hash/environment, storage registration and global invocation counter. Complete resource approval and the Lead's process-tree/RSS/GPU-memory supervisor are required. `timeout` alone is not an RSS cap, byte-rate limiter or GPU lease. Budget ceilings and their unimplemented enforcement boundaries are explicit in `EXECUTION_PLAN.json`; no existing protected training process is signalled.

The protocol and small controls can be inspected without loading images/weights. Do not execute `plan-cache` repeatedly as a status check: it fully hashes native payloads and selected images.

```bash
cd "${ST_PROBE_RUNTIME:?clean reviewed runtime required}"
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  /usr/bin/time -v -o "${ST_PROBE_LOG_ROOT:?}/plan.time.txt" \
  timeout --signal=TERM --kill-after=30s 600s \
  taskset -c "${ST_CPUSET:?reviewed CPU lease required}" \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.spatial_probe plan-cache \
  --binding "${ST_W_BINDING:?actual 5000 binding required}" \
  --output "${ST_W_ROOT:?exact registered tensor claim root required}" \
  > "${ST_PROBE_LOG_ROOT}/plan.json" 2> "${ST_PROBE_LOG_ROOT}/plan.stderr"
```

This does not activate CUDA, so static package checks can run with an empty CUDA mask while the immutable binding still names its eventual extraction GPU. The native/image I/O is still real and counted. Lead checks the plan identity and byte fields against the exact tensor claim before admission. This preparation alone does not provide a new ledger mutation command.

After independent typed-storage approval and current GPU lease, first extract the fixeddev16 of the first available5000 arm, then its fixedtrain64. Each split has its own binding/root/log directory; do not overwrite control evidence. Initial concurrency is one loader/extractor globally.

```bash
cd "${ST_PROBE_RUNTIME:?}"
env CUDA_VISIBLE_DEVICES="${ST_GPU:?single reviewed GPU index required}" \
  PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  /usr/bin/time -v -o "${ST_PROBE_LOG_ROOT:?}/extract.time.txt" \
  timeout --signal=TERM --kill-after=30s 960s \
  taskset -c "${ST_CPUSET:?}" \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.spatial_probe extract \
  --binding "${ST_W_BINDING:?}" --output "${ST_W_ROOT:?}" \
  --storage-policy "${ST_STORAGE_POLICY:?typed W admission required}" \
  > "${ST_PROBE_LOG_ROOT}/extract.json" 2> "${ST_PROBE_LOG_ROOT}/extract.stderr"
```

`extract` has no `--resume`; replaying these exact arguments validates and reuses correctly receipted records. PARTIAL exits2. A continuation consumes an explicitly counted extra role lifecycle, with a new attempt log namespace, unchanged binding/root and retained partial evidence. Do not enlarge limits or remove partials automatically. A single model load can cross the cooperative deadline, which is why the outer supervisor is necessary. Preserve the actual command exit code; do not pipe it through a success-only `tee` without pipefail.

After both64/16 COMPLETE receipts and full hashes pass, ensure the extraction process has exited and released its GPU. Fit on CPU:

```bash
cd "${ST_PROBE_RUNTIME:?}"
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  /usr/bin/time -v -o "${ST_PROBE_LOG_ROOT:?}/fit.time.txt" \
  timeout --signal=TERM --kill-after=30s 1260s \
  taskset -c "${ST_CPUSET:?}" \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.spatial_probe fit \
  --registration "${ST_FIT_REGISTRATION:?sealed CPU fit required}" \
  --output "${ST_FIT_ROOT:?separate bounded output required}" \
  > "${ST_PROBE_LOG_ROOT}/fit.json" 2> "${ST_PROBE_LOG_ROOT}/fit.stderr"
```

For an actual interrupted same-identity fit, use the same command, append `--resume`, and write a new attempt log directory. At most one extra CPU fit invocation per arm in this preparation. `--resume` must not be passed on the first invocation. Fixed300 terminal checkpoints do not add updates. The two slots and STATE pointer preserve committed cursor/RNG; a hard stop may discard uncommitted work since the last25-cycle save, not silently skip samples.

Capture actual full wall/RSS, call counts, logical file read workload, filesystem byte counters and final W sizes. If per-cycle or per-scene p50/p95 is requested, add a reviewed thin timing observer around actual calls; current CLI output and `/usr/bin/time` do not contain those distributions. All such timings are UNKNOWN now. Do not use historic image-to-trajectory timing as image-to-W performance or synthetic tests as real fit speed.

The required resource summary after first qualification is: source/runtime and checkpoint pins; requested/completed coverage; actual GPU and peak memory; native/hash/load/image/write/evaluation costs; CPU fit wall/RSS and actual optimizer updates; all retries/retained partial bytes; no train/dev leakage; and the remaining 768 GiB reservation/free-space/checkpoint budget. Lead then obtains independent real efficiency/correctness conclusions before dispatching the remaining arms under the same finite plan.
