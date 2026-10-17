# Bounded cache decision and actual readiness

This is an engineering update for the ST world/consequence campaign, not an IQE
cache migration or a new scientific result. It implements the current instruction
to avoid several-TB cache allocation. Runtime training remains the reviewed
DriveDreamer-Policy execution adaptation; this note changes no model, target,
precision, sampler, optimizer or active run.

The original plan is `docs/research/WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md`
v1.0, especially sections 5.3 and 6. The previous readiness snapshot and failures
are preserved. The lead source at this update is
`00263262` (test-only integration after `a6666b39`); active W and C execution
trees remain `fdf25f99b702754787904215bf1c2a2b7515ea75` and
`6714b1e9fb2e0c8059b90b8b4d48cd40796fe345` respectively.

## Enforced storage boundary

- Existing complete current-frame training target tensors: **629,345,297,104 B
  = 586.123482 GiB**, summed from the three immutable COMPLETE receipts.
- Campaign cache cap: **824,633,720,832 B = 768 GiB**, including cache outputs,
  metadata, staging and ledger overhead. Existing reservations are
  **645.318253 GiB**, leaving **122.681747 GiB**. Reservations are conservative
  allowances, not measured occupied bytes.
- Retain the separate **1,024 GiB filesystem free-space floor** and independent
  checkpoint reservations. The current four W plans reserve 960 GiB in total;
  this is not cache space. Do not delete current caches or checkpoints to meet a
  new request, and do not treat filesystem free bytes as an increased quota.
- New writers must use `tools/st_world_policy/cache_storage_budget.py` and the
  actual policy/claim. Legacy completed writers were bounded before this guard;
  they are not retrospectively described as instrumented by it.

Actual policy:
`/mnt/project/st-world-policy-20261016/resources/CACHE_STORAGE_BUDGET_POLICY_v4.json`.
The active jobs reuse the complete current caches without generating another
copy. This update launches no cache writer and changes no storage policy.

## Future labels: reject the multi-TB prebuild

Native current labels on disk are 18×24×1024 FP32 plus a bool mask, **1,769,904
bytes per image**. The reader applies the registered fixed PCA384 transform to
FP32 targets. Four future times over 101,592 scenes and three views give
1,219,104 image occurrences: an un-deduplicated **2,009.511968 GiB** payload
estimate, before staging, metadata or dev. This estimate is not a measured
unique-image population. No such full future cache is authorized or started.

Read-only exact SHA joins find current-cache matches for **441/765** train-smoke
future images and **123/192** dev-smoke future images. These counts are not a
full-population hit-rate estimate. Timestamp matches in the full train metadata
do not prove image equality. Reuse requires matching source image and the full
teacher, grid, preprocessing, dtype and transform contract; roles remain
separate even when bytes are shared.

The present `TeacherCache` requires COMPLETE population coverage and reads all
four future times before the existing pair sampler. It has **no on-demand
teacher service or disk rotation**. A missing record fails; it must not become
an invalid-source mask, a skipped scene or a replacement sample.

The preferred next engineering boundary, conditional on the spatial result, is
exact existing-feature reuse plus generation of genuinely missing requested
targets with the unchanged frozen teacher and bounded RAM. Waiting is attached
to the original update: preserve sample order, random streams, all legal future
times and target values. A request-aware provider must preserve the existing
adjacent-pair RNG and patch requests. It needs target/mask, numeric and consumed-
cursor resume qualification before training. **This provider is not implemented
or approved by this audit.**

An optional 64 GiB disk payload + 8 GiB staging + 4 GiB metadata tier is only a
76 GiB feasibility example. Rotation would additionally need immutable slots,
reader leases, atomic publication and safe retirement of only new ephemeral
files. It is not an active reservation, deletion authorization or hidden change
to the current reader. No quantization, reduced temporal supervision or altered
dataset is used to fit the cap.

Read-only feasibility evidence:
`/mnt/project/st-world-policy-20261016/audits/temporal_bounded_storage_feasibility_v1/`.

| Artifact | SHA256 |
|---|---|
| `SUMMARY.txt` | `412e0fa7d147786b2ef7db141d7c1fe9987e62ad740ebfce8b9affe6996c5fd7` |
| `FEASIBILITY.json` | `b359cc8e97b7f6233597c2805cd19ea25d484f1549a861a1185c2171e10b745f` |
| `METADATA_REUSE_COUNTS.json` | `10631dd2410988e184abfd8f3fe11f537ace9f24bb5d5ad87235929d5d041f11` |

## Small learned-W probe: code review complete, real execution pending

The fixed 64-train/16-dev learned-W probe does not require a full-dataset cache.
All eight arm/role roots reserve **546,104,992 B (~520.8 MiB)** together,
including metadata and staging; actual new probe cache bytes remain zero.

Independent Correctness closed SPROBE-01: metadata-only `plan-cache` invokes no
CUDA/cuDNN device discovery, while actual extraction/fit still verifies full
runtime, CUDA device, TF32 and deterministic settings. The original failed
review and the later outdated-test-fixture failure remain available. The fixture
was corrected in independent commit
`c646c16ac30edb32b69f6ac09159470f05a70390`, integrated as `00263262`.
Only the independent test file changed in that integration; reviewed production
code is identical to `a6666b3931f31d6c51975daf76ac48b469ea2129`.

Actual independent command (production ddp environment):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
timeout 120 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_spatial_probe.py \
  tests/st_world/test_review_spatial_probe.py
```

Result: **20 passed, 8.20 seconds**, no skips, no GPU/model/scoring job. The
independent report is
`audits/correctness/LEARNED_W_SPATIAL_PROBE_REVIEW_v2.json`, SHA256
`9e4ca8f5b4935852e436aa7cccc536b8312ab7a39155f484bb31fe774aa4db34`, under the
private campaign root. Its status is `PASS_LIMITED_CODE_AND_CPU_REVIEW`, not a
formal GPU launch approval. Actual 5k checkpoint extraction, real-image W
parity, probe fitting and scientific results remain pending. The shared head
has 495,362 parameters; 300 exposure cycles need not equal 300 optimizer steps
when an exposure has no valid labels.

## Independent Flow path

The complete 101,592-scene/1,176-log C manifest and its two existing metric-cache
roots introduce **zero new teacher/condition/metric tensor cache bytes**. Three
real dataset rows have been loaded through the actual current-image/state/GT
boundary; this is not a full-population numeric test.

The rl-zt2 eight-GPU reference PROFILE completed 240 updates and 30,720 official
requests with a complete native checkpoint. Its `PAUSED` state is the intended
bounded PROFILE boundary on the unchanged 4,096-update clock, not a formal
training result or initializer. The concurrent four-GPU profiles are still
pending at this snapshot. Formal C0/C1 metadata has been prepared for independent
review; no formal C launch is claimed here. Cache engineering does not block
this independently qualified zero-new-cache path.
