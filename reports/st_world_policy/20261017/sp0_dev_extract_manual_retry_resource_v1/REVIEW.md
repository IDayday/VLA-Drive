# SP0 dev extraction manual retry: resource precheck

**PENDING_INDEPENDENT_CORRECTNESS_NO_ADMISSION.** This report prepares the exact resource facts; it does not permit launching the retry. The original expired admission, failed extraction, plan, runtime and migrated storage evidence remain unchanged.

The new Lead controller at commit `61942f3fdecddb8cc374720bda130b73c876431b` adds only the fixed `stages/dev_extract_manual_retry_v1` attempt and explicit lineage/accounting. Exact source pins observed:

| Helper | SHA256 |
|---|---|
| supervise_once.py | `747873b72ce6bcb2cfd3163bd89ce1560e4538d1069952d8deed176c349a67ef` |
| observed_cli.py, unchanged | `c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643` |
| prepare_bindings.py, unchanged | `31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84` |

The helpers are colocated in Lead's `reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/`; the child observer still imports the clean model runtime `581399eb309677c20fe7464791b4f18685e947a6` specified by unchanged PREPARATION SHA `34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431`. This precheck does not substitute new helper code into that immutable model runtime.

The actual new `stage_inputs` was called only on small control JSON. It reproduced these exact charges:

| Item | Logical bytes |
|---|---:|
| Completed dev plan | 37,533,999,575 |
| Failed extraction, including admitted in-flight hash | 30,632,210,276 |
| Previously charged total | 68,166,209,851 |
| Original role ceiling | 274,877,906,944 |
| Retry remaining | **206,711,697,093** |

These are conservative recorded logical-read charges, not exact physical I/O. The old monitor snapshot's limitations remain disclosed; this review does not retroactively identify its swallowed exception or measure its missing final bytes. No new plan resets the budget.

Resource observations in `PRECHECK.json`:

- GPU5 is the requested UUID `GPU-cba744b9-96c8-c3e2-f836-2f0126c038a0`, 4 MiB allocated, 0% utilization, with no reported compute process. This is not literally zero allocation. The local GPU5 lock and global one-loader lock have no matching `/proc/locks` holders at the observation. Lead must freshly recheck and acquire both; the reviewer acquired neither.
- Shared free space is 7,337,887,313,920 bytes. Use the new `CHECKPOINT_COMMITMENTS_2183G_V1.json`, SHA `d5bce1fec28d6ce3dab496287c9be41852ca154495ecf4cf686c7e780ebf4d52`, including the running GPU0 PROFILE's 80 GiB in addition to the old 2,103 GiB, plus the existing 1 TiB free floor. The old 2,103 GiB receipt must not be reused. Quota is still unverified.
- Ledger SHA remains `433852d03c46345c5418880693f0575a2111a30e1c1292c01b1161e366c5daca`. All seven prior rows remain; the already migrated exact dev W member is 26,085,776 bytes. Its output root does not yet exist. No new member, claim, migration or 768 GiB cache-cap expansion is required for this same planned extraction.
- The entire SP0 audit/control job root currently charges 299,008 bytes. An additional bounded 5 MiB retry control directory fits inside the original 32 MiB administrative allowance. Effective campaign metadata remains 1,264 MiB of 4 GiB; no duplicate new reservation is invented.
- Retry directory is absent. Original plan and failed receipt hashes match the pinned actual evidence. No actual source, result, ledger or output was changed.

After independent correctness PASS on the exact controller, resource admission can bind these helpers, original actual dev signature, migrated policy/receipt, new 2,183 GiB commitments and explicit failed/plan result pins. It must permit only dev16 extraction, CPU0/1, one freshly checked GPU5, 64 GiB sampled RSS, 24 GiB GPU allocation, 960 seconds plus 40 seconds cleanup, and the remaining read budget above. Admission/commitment validity must cover that full window. No train_plan, train extraction, fit, full cache, generic retry, or new dataset is included.

No model/native/teacher/image tensor was read. No GPU computation, extraction, migration, registration, process signal, dispatch or ledger mutation occurred. Only bounded metadata/stat/proc-lock/GPU-query inspection and owned review artifacts were used.
