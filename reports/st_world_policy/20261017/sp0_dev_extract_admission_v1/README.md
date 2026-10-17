# SP0 dev extraction: one-shot resource admission

Result: **PASS_BOUNDED_DEV_EXTRACT_ADMISSION**. This approves only the prepared SP0 dev extraction under the pinned controller. It is not evidence of completed extraction, tensor correctness, training throughput, or formal temporal-training qualification.

The reviewer read 196,164 bytes of small JSON, helper source, and filesystem metadata. No model, native checkpoint, image, teacher tensor, SQLite payload, or GPU computation was read or run. The existing independent controller review and 57 tests are inherited; this review does not repeat them. The existing controller's `validate` accepted this exact admission with `(960 seconds, 64 GiB RSS)`.

## Exact execution identity

- Runtime: `/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017`, clean commit `581399eb309677c20fe7464791b4f18685e947a6`.
- Original preparation SHA256: `34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431`.
- Actual dev signature SHA256: `3de51e64aec7f555f8d5c053695656346e1549796dad08c2c5bdafa88a26cff4`.
- Post-migration ledger SHA256: `433852d03c46345c5418880693f0575a2111a30e1c1292c01b1161e366c5daca`.
- Admission SHA256: `4efd23dd4be6d3f42d826c569449a7dbb927d89a3a8ee2e5dd2a8b620ccc8010`.
- The seven previous ledger claim rows exactly match the preserved migration snapshot. The policy declares one new dev W member; the extractor will reserve its actual row. This review did not modify the ledger or perform the migration.

The migration's completed receipt, actual result, new policy and ledger agree. The Lead's pinned quiescence evidence records no active/unknown cache writers and disabled queued writers. This review did not repeat the eight-host census.

## Limits and protected space

| Item | Admission |
|---|---:|
| Dev scenes | 16 |
| Typed W output + staging + metadata | 26,085,776 bytes |
| W package maximum | 1 GiB; only this exact member declared |
| Cache cap | Existing 768 GiB |
| Effective metadata reservation | 1,264 MiB of 4 GiB |
| Checkpoint commitments | Full 2,103 GiB; no subtraction of existing files |
| Additional free-space floor | 1 TiB |
| Observed free space at review | 7,415,042,912,256 bytes |
| CPU affinity | 0, 1 |
| GPU | Local 5, `GPU-cba744b9-96c8-c3e2-f836-2f0126c038a0` |
| Observed process RSS / CUDA allocator ceiling | 64 GiB / 24 GiB |
| Cooperative / outer runtime / abnormal cleanup | 900 / 960 / 40 seconds |
| Plan + extract logical read allowance | 256 GiB total |
| Already consumed by completed plan | 37,533,999,575 bytes |
| Remaining extract read allowance | 237,343,907,369 bytes |

The 1,264 MiB metadata amount includes 1,136 MiB declared in the shared ledger and 128 MiB administrative allowances (64 MiB qualification reports, 32 MiB SP0 controls, 32 MiB completed factory). Administrative allowances are not described as already consumed bytes or as new ledger claims. All later admissions must carry them forward.

The GPU observation was a separate read-only hardware query: GPU 5 had 0 MiB used and 0% utilization with the specified UUID. The evidence script records that observation; it does not independently measure GPU state. The launcher must recheck availability and acquire the correct GPU lock immediately before starting. Other jobs and unknown owners remain protected. Resource monitoring is the previously reviewed polling controller, not a hard cgroup isolation guarantee.

## Validity and command

The admission expires at Unix `1792224928.5775177`. Start no later than Unix `1792223928.5775177` so the complete 960-second run plus 40-second cleanup fits. The pinned checkpoint commitment remains valid until Unix `1792226649.4307477` (2026-10-17 08:44:09 UTC). The entire permitted window precedes that expiry.

Only Lead may execute this command once:

```bash
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py \
  --admission /mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/sp0_dev_extract_admission_v1/DEV_EXTRACT_ADMISSION.json \
  --admission-sha256 4efd23dd4be6d3f42d826c569449a7dbb927d89a3a8ee2e5dd2a8b620ccc8010
```

No automatic retry, train extraction, fit, further W members, or formal training is authorized here. The review script is an audit record of this issuance, not permission to regenerate an expired approval. Any incomplete execution retains its one-shot evidence for a separate decision.
