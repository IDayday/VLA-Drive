# T3 exact32 CPU preparation v2 — unsigned dispatch proposal

The v1 attempt remains FAILED_NO_AUTO_RETRY at its original root. Its source migration correctly rejected the protected reader mismatch; no result is rewritten. The v2 output is `/mnt/project/st-world-policy-20261016/audits/temporal_T3_indexed32_preparation_v2` and is still absent. This is one new explicitly reviewed attempt, never an automatic retry.

The frozen runtime is `/mnt/project/VLA-Drive-st-temporal-packed-migration-fix-20261018`, commit `cf516259c8efc24503727f14b9e330ed32231293`. The old runtime `676e3e88d48d9444ab2cb928242d196381631b26` is unchanged. The only executable binding diff from reviewed v1 is four literal replacements: runtime path, runtime commit, output root, PLAN SHA. The GNU timeout launcher is byte-identical. All monitoring, single-JSON child accounting, same-mount/publication guards, counters, cleanup, no-retry and exact two-CLI order are inherited.

## Protected source compatibility

Exactly one of fifteen protected sources differs: `starVLA/dataloader/st_world_history.py`, old `ab35c5df8f72b8ebff90887515c18af1755919a761ab7c0adf3d09c0a6d61a7d` to new `bdb33b97f820e251913d6c99cb782791d977e01599c75da620d244bd831fb97c`. This is the existing packed metadata reader; fourteen other protected files must remain byte-identical. An explicit `mathematics_compatibility` record is required and hashed in the migration identity. Unknown/reversed reader hashes, another protected change, missing/tampered proof remain errors. No original hash or receipt was replaced.

Independent source evidence is pinned in PLAN and compatibility:
- `/mnt/project/VLA-Drive-st-history-scale-independent-v2-20261017/reports/st_world_policy/20261017/history_metadata_scale_independent_v2/REVIEW.json`: `5bad2b2136ce5823975b5f0918c6965a184e5b397ad32a3cde8340f1f2c273ac`.
- The same directory `SOURCE_BINDING.json`: `311ca1c5f264c96c0f677cef600adf7b5dd4dca1533397317e0298e8518cadfd`.
- Local explicit `MATHEMATICS_COMPATIBILITY.json`: `412733d3f1510276a06714b1afb831213f926615f58f11a6ab6e7d2781c2b20e`.

## Real source preflight

The runtime report `reports/st_world_policy/20261018/temporal_packed_history_migration_v1` retains the actual old `prepare_control_migration` refusal and fixed original CLI/validator coverage (7 tests). In this v2 report, `audit_native_preflight.py` invoked that same original CLI and validator, with no mocking and no composition preparation. `OPENED_PATH_AUDIT.json` retains every Python audit open path and rejects database/image/model/tensor suffixes. This is Python audit coverage, not a claim to trace libc or git-child system calls.

Second native audit: 4.834 seconds, 152,465,823 self-rchar bytes, 639,492,096 bytes peak RSS, no forbidden opens. Three focused binding tests passed in 1.585 seconds. The independent native review is separate; the author does not approve launch. No actual preparation, GPU execution, ledger write or reservation was performed here.

## Resource proposal and execution

Keep CPU 2/3, aggregate RSS 2 GiB, 4 GiB logical-read cap, 300 seconds plus 40 seconds cleanup, whole output 8 MiB including 1 MiB terminal controls. The old failed v1 reservation remains 8 MiB. Conservative prior 7880 MiB plus new 8 MiB yields 7888/8192 MiB, leaving 304 MiB. No typed claim changes. These figures are a proposal requiring the Lead's fresh actual host/lease/reservation admission.

`ADMISSION_PROPOSAL.json` is explicitly unapproved with null issue/expiry times. Do not pass it to launch as if approved. The previous CP expires at 2026-10-18T01:44:11Z; the Lead must use current evidence and a fresh valid time window. No old approval is reused.

After independent narrow review and the Lead issues actual admission, the existing executable entry is:

```bash
bash /mnt/project/VLA-Drive-st-teacher-channel-control-20261017/reports/st_world_policy/20261018/temporal_T3_cpu_preparation_v2/launch_once.sh \
  --admission "$T3_V2_ACTUAL_ADMISSION" \
  --admission-sha256 "$T3_V2_ACTUAL_ADMISSION_SHA256"
```

The two environment values above must name the newly issued actual file/hash; they are not provided or fabricated here. The controller executes `qualify_indexed_provider prepare-profile-migration` then `temporal_composition --cell T3` with exact PLAN inputs. It does not invoke the GPU qualifier or trainer. The generated PROFILE recipe is only four updates, original global batch 2 / warmup 1000 / 20k scheduler. It is not formal T3 numerical qualification or training approval. T4 is disabled.
