# One explicit SP0 dev retry: final independent resource admission

**PASS_ONE_EXPLICIT_DEV_EXTRACT_RETRY_RESOURCE_ADMISSION.** This supersedes the pending status of the earlier resource precheck only for the exact single attempt below. It does not change or renew the original failed attempt's admission.

Lead supplied independent correctness PASS for source `4a9ef2be`: 13 owner and 12 independent CPU cases passed. The reviewed helper SHA is `747873b72ce6bcb2cfd3163bd89ce1560e4538d1069952d8deed176c349a67ef`. This resource review inherits that separate conclusion and does not claim to have rerun those tests. The actual new supervisor `validate`, `stage_inputs`, and fresh hardware check were independently called on small metadata; all passed without executing a stage.

Admission:
`ADMISSION.json`, SHA256 **`adc1b2fea2418f138a8668c780c3ed6ef0f2a9c3226fe1c4695a37d579245898`**.

Only this new fixed directory may be created by Lead:
`/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/stages/dev_extract_manual_retry_v1`.

The original dev_plan, original failed dev_extract, and original PREPARATION remain pinned. The original 256 GiB role budget is not restarted: **68,166,209,851 bytes** were charged to plan plus failed extraction, leaving **206,711,697,093 bytes** for this attempt. No new plan is run. No additional retry, train_plan, train extraction, fit, data expansion, or cache claim is approved.

| Limit | Value |
|---|---|
| Model runtime | Original clean `581399eb309677c20fe7464791b4f18685e947a6`; rechecked by launcher |
| New controlling helper | Lead's `spatial_probe_actual_preparation_v1/supervise_once.py`, SHA `747873b7…` |
| Child observer / bindings | Original `c1af891b…` / `31d2d302…`, colocated and exact-pinned |
| Input | Original dev16 and actual dev signature |
| GPU | 5, UUID `GPU-cba744b9-96c8-c3e2-f836-2f0126c038a0`; fresh check passed |
| CPU | 0, 1 |
| Observed RSS / GPU allocated limit | 64 GiB / 24 GiB |
| Cooperative / outer / cleanup | 900 / 960 / 40 seconds |
| Existing typed W claim | 26,085,776 bytes; no migration or extra member |
| Retry controls | Up to 5 MiB inside the existing SP0 32 MiB allowance |
| Effective total metadata reservation | Unchanged 1,264 MiB of 4 GiB |
| Checkpoint protection | New 2,183 GiB commitment plus 1 TiB free floor |

The fresh extraction check observed **7,337,491,095,552 available bytes**. The shared cache ledger is still SHA `433852d03c46345c5418880693f0575a2111a30e1c1292c01b1161e366c5daca`. New checkpoint receipt is `CHECKPOINT_COMMITMENTS_2183G_V1.json`, SHA `d5bce1fec28d6ce3dab496287c9be41852ca154495ecf4cf686c7e780ebf4d52`, including the active GPU0 PROFILE's 80 GiB. Quota remains unverified. The old 2,103 GiB receipt is not used.

The admission expires at Unix **1792229597.0036902**. Start no later than **1792228597.0036902** to preserve the full 1,000 seconds for execution and cleanup. Both are earlier than the checkpoint receipt's expiry. Observed resource limits retain polling/I/O/kernel scheduling qualifications and the counter lock's bounded two-second wait; no limit was relaxed.

Lead's exact one-shot command:

```bash
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py \
  --admission /mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/sp0_dev_extract_manual_retry_resource_v1/ADMISSION.json \
  --admission-sha256 adc1b2fea2418f138a8668c780c3ed6ef0f2a9c3226fe1c4695a37d579245898
```

The new helper rechecks global-loader and GPU locks/ownership, runtime, memory, full validity and storage before consuming the fixed attempt. Its failure remains spent and preserved. Success requires the actual complete result; this approval is not a claim of successful W extraction or scientific qualification. The reviewer did not dispatch, create the new attempt, touch the ledger, alter old evidence, or read any model/image/tensor payload.
