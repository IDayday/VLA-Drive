# Exact prepared metadata resource package

**PASS_PREPARED_PACKAGE_PENDING_LIVE_ADMISSION**, reviewed 2026-10-17 05:47 UTC.
No live migration or build was performed. This is a finite metadata qualification
review, not approval of teacher extraction, full future preparation or training.

The exact preparation record is
`/mnt/project/st-world-policy-20261016/audits/metadata_migration_preparation_v1/PREPARED_V5.json`.
Its runtime is the clean, isolated
`/mnt/project/VLA-Drive-st-metadata-qualified-runtime-v1-20261017`
at `1162b955b3bf9abb6728823a84a50c38e1a19bb8`.

## Bound package verified

| Identity | SHA256 |
| --- | --- |
| v5 policy | `652bc1f8b5792ee558b9b30064bd117df64a4675b8ea0f84429bea75555a2739` |
| Unchanged v4 | `5fcc4421789cda3ed23b366dd78cf9fa7a867c3e7734aa3f1ed6bdaf1836618a` |
| Original ledger | `da4a5b89427a7aacf41265b0c3babe528b6a1999557b3e34dc709c13ccab743c` |
| Current index storage registration | `4501172512fa2c82c0c8dd86a8b437829c812dd5ded3c3be222a6555f43537d2` |
| Future32 storage registration | `2ef10a89fa5ed5df408cdf82edb363d2c4ce09955cc78f669854b7faee6dcfa4` |
| Future32 preparation registration | `7479fb602c9b550fe3b8d9b451cbb1c9596c6439e41492fb43a3a7cd5e8db5e8` |
| Future32 PLAN | `32bcf1d6edd7204a4b21d275fc521c66e817809cad127860f70bfddcdc781794` |
| Seven-pilot inventory | `e05bcc80f5c27c08725e93713968e9f670c8e354791de881eadfc97c7261f151` |

The inspection script verifies 28 checks: exact prior claims/reservations and
campaign envelope, one unchanged ledger, original package baseline, the seven
retained pilot stat inventories, MiB-rounded charges, exact entry/dependency
source hashes and callback declarations, preexisting PLAN/lock pins, and the
32-scene plan/registration/metadata-only limits. The new PLAN preserves every
selected row and its ordering from the original32-scene plan.

| Item | Charged reservation |
| --- | ---: |
| Seven retained pilots, including prior future32 PLAN | 16 MiB |
| Current full metadata index, output + staging + control | 1,024 MiB |
| Future32 metadata build, PLAN + staging + control | 32 MiB |
| Total added | **1,072 MiB = 1,124,073,472 bytes** |

Total campaign reservations become **694,029,271,910 bytes**. They remain below
768 GiB by **130,604,448,922 bytes**. The one metadata package cap remains4 GiB;
the unused portion is not automatically reserved again. v4, original legacy
reservations/claims and 1 TiB free floor are unchanged. New roots are within the
existing allowed `/caches` namespace. The only allowed-root extensions are exact
retained pilot roots, with no live writer permitted there.

Current index: `caches/full_current_metadata_index_v1`, 1 GiB total,
1008 MiB builder limit leaving16 MiB control, max1200s /4 GiB process-tree RSS.
Future32: `caches/future_metadata_qualification32_v2`, total32 MiB including
PLAN, control and partials, max450s /4 GiB RSS, one worker and64 MiB/s source
read throttle. No new teacher tensor is authorized. Two-CPU affinity is to be
enforced by the outer launcher, not silently inferred from these registrations.

## Required before actual invocation

- Lead-owned fresh quiescence and disabled writer/retry controllers covering
  every original cache/PCA claim and all adopted pilot roots.
- A fresh pinned checkpoint commitment receipt; the accompanying independent
  recommendation retains1983 GiB rather than deducting current allocations.
- Exact outer monitor: only the owned child process tree; CPU2 affinity;
  cumulative finite wall deadline; process-tree RSS; bounded output pipes;
  failure propagation and no automatic retry. The inside callbacks continue
  to check directory allowance/free floor/RSS/deadline before writes and during
  bounded loops. The outer monitor complements those callbacks.
- Comparator/reference qualification outputs, temporary files and monitor logs
  must remain within the same32 MiB future control package. Its separate450s
  reference window must be explicitly registered; the build registration is
  not blanket permission for additional processes or persistent outputs.

Until these actual bindings are supplied, **actual admission is pending**.
Existing active model training readers do not need a new policy/runtime. There
is no authorization here to mutate their pinned v4 files or stop those jobs.

The review used bounded explicit control/source files and stat calls over seven
specified pilot roots. No image/teacher/weight payloads were read, no full
current manifest parsed or rehashed, no global project scan, no GPU operation,
no deletion, no ledger write and no job dispatch occurred.

Replay the same read-only check (no build):

```bash
cd /mnt/project/VLA-Drive-st-metadata-actual-admission-20261017
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 timeout 45s \
  taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/metadata_actual_admission_v1/review_prepared_package.py
```
