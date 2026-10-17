# SP0 train64 W plan: one finite resource admission

**PASS_ONE_FIXED_TRAIN_PLAN_RESOURCE_ADMISSION.** Lead may invoke the existing supervisor once for `train_plan` using [ADMISSION.json](ADMISSION.json), SHA256 `9d39cae34cbb92c44bbee216cfa07b86edd4d3987f3191e37a858350cc1bcd55`. This approves the existing fixed plan/checksum operation only. It does not approve dev re-extraction, train extraction, fit, automatic retry, formal training or a new cache claim.

Start no later than **2026-10-17T09:22:58.772300+00:00**; expiry is **2026-10-17T09:33:38.772300+00:00**. The full 640-second interval covers the unchanged 600-second outer limit and up to 40 seconds of cleanup. The supervisor itself tests 630 seconds at validation; this admission deliberately requires the complete 640 seconds. A missed start window needs fresh resource review, not a changed original admission.

| Resource | Exact boundary |
|---|---|
| Host | `training-vla-zt-worker-0`; currently allowed `training-vla-zt` alias |
| CPU | `[2,3]`; reviewer sampled both idle; subsequent metadata uses `[0,1]` by Lead instruction |
| Observed RSS | 64 GiB, unchanged supervisor |
| GPU | None; `CUDA_VISIBLE_DEVICES=''` |
| Train role reads | 256 GiB = 274,877,906,944 bytes across train plan plus later extraction |
| Train reads consumed before this plan | 0; no train stage directory exists at review |
| Plan output | Existing 5 MiB attempt-directory cap, inside the existing 32 MiB arm control allowance |
| Physical floor | 1 TiB remains after reserving the full 32 MiB output envelope |
| Fresh memory/free disk | 859.073 GiB / 6780.739 GiB |

The same global shared-Lustre loader lock and one-shot `mkdir(exist_ok=False)` remain in force. The reviewer validated the mount and observed no matching local kernel lock, without acquiring or modifying a lock. Lead separately confirmed the full-future dev/train PLAN jobs had ended. Shared filesystem IO with subsequent metadata preparation remains a possible contention source; CPU separation is not an IO-isolation guarantee. Prefer finishing this one W plan before metadata starts when practical. No resource snapshot is a process-termination authority.

## Exact additive prerequisite and source

Lead created the live 14,337-byte reconciliation receipt with SHA256 `3722ef6e87871429be10718fb5fda5c332aa3ecc5aec24290de389c309079f1a`. Its status remains `ARTIFACTS_ACCEPTED_SUPERVISOR_FAILED`; the original failed RESULT remains `STOPPED_NO_AUTO_RETRY`, SHA256 `57406c7728bd2937d6723ebe3b22596fd04dd45d0fcabaa22c64311daf11166c`. The reviewer called only the real `stage_inputs` metadata path and `validate`, never `main`, `supervise`, `cache_plan`, extraction or model code.

The receipt preserves the dev role's 68,166,209,851 prior bytes, 153,770,280,621 manual-attempt bytes and 37,758,464 nominal independent validation bytes, both spent extraction attempts, and zero additional extraction attempts. Its remaining dev figure is still an upper bound before small metadata reads. The train role's existing separate 256 GiB budget is not a reset or enlargement of that dev budget. Actual train-plan charge must be subtracted before any separately approved train extraction.

The four helper pins match the Lead worktree byte-for-byte:

| Helper | SHA256 |
|---|---|
| `observed_cli.py` | `c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643` |
| `prepare_bindings.py` | `31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84` |
| `reconcile_artifact_exit.py` | `8deea58c4550866500378f6016cbca0f21e75fa937285b38350528c0bdd61de3` |
| `supervise_once.py` | `d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902` |

The original frozen runtime is clean at `581399eb309677c20fe7464791b4f18685e947a6`. Training source remains `fdf25f99b702754787904215bf1c2a2b7515ea75`; checkpoint tag is `step_005000_1792217229597178910`, deployment identity `c87c59804b6237205e4b4a7a7e355b569c7cef5eaa499023f699111ec36d0beb`. Native COMPLETE still matches SHA256 `1583c98960120ad2cde5859919507386b27259c4b3524a82b40de6ce0f1299fe`; nine inventory sizes match by stat. No payload content was reread by this review. Original native/content checks remain in the actual plan and later fit; metadata/stat evidence does not replace them.

The source correctness conclusion is inherited from the independent 37-case reconciliation-consumer review, not retested here. Its explicit pre-reconciliation payload-stat/content limitation is unchanged.

## Actual cost and entry point

`observed_cli.py` will call the original `spatial_probe.cache_plan(binding)` exactly once. Its reads include **37,366,035,746 bytes** of native checkpoint inventory, **41,433,288 bytes** across the fixed train64's 192 source-image references, and a **9,549,650-byte** source index, plus run/config/base controls, current JSONs, protected source and runtime imports. These are receipt/stat costs, not a newly measured plan throughput. The plan hashes payloads and is not metadata-only; it constructs no model, loads no native tensors, consumes no GPU, and reads no teacher/GT/metric payload. The existing counters charge `max(admitted hash + nominal torch.load, proc rchar)` and retain in-flight work; this is not a physical-IO counter.

Lead's only approved entry is:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py \
  --admission /mnt/project/VLA-Drive-st-w-train-plan-resource-20261017/reports/st_world_policy/20261017/w_train_plan_resource_v1/ADMISSION.json \
  --admission-sha256 9d39cae34cbb92c44bbee216cfa07b86edd4d3987f3191e37a858350cc1bcd55
```

Do not additionally run the raw plan CLI or invoke `observed_cli.py` directly. This admission requires `gpu=null` and `storage_policy=null`. No claim or ledger mutation is needed for the plan. The read-only ledger snapshot matches the completed v8 core migration and retains the 768 GiB cache cap, 1 TiB floor, 4 GiB metadata package and 1 GiB W package. Only the existing dev W member is present; the future train W claim, current policy/commitments and train extraction need their own exact review after an actual plan signature exists. The historical dev space receipt is not carried forward as a current extraction approval.

## Review cost and evidence

The preliminary receipt bound was 93,440 application JSON bytes, 16 payload stat checks, zero payload reads, at most 64 KiB receipt output (128 KiB including conservative staging), two CPUs, 2 GiB address-space/RSS budget and 180 seconds. Python startup IO was explicitly excluded from that JSON count. Lead reported receipt creation in 0.137 seconds.

The complete metadata preparation check used 325,346 application JSON bytes, 21.30 MiB peak RSS and 0.907 seconds. Fresh admission checks used 19.86 MiB peak RSS and 0.943 seconds, with affinity `[2,3]`, CUDA hidden, no torch import and a 180-second outer timeout. An initial collector stopped on the existing migration receipt's nested `result` envelope and was corrected; no workload was rerun.

[EVIDENCE.json](EVIDENCE.json) contains exact prerequisite/accounting/source pins and small stat observations. [FRESH_ADMISSION_EVIDENCE.json](FRESH_ADMISSION_EVIDENCE.json) records the fresh host, policy, source, validation and output bounds. The reviewer changed only this report directory, did not dispatch, mutate ledger/policy/cache/runtime, acquire a workload lock, signal a process or read weights/images/W tensors.
