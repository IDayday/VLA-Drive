# Independent exact one-retry control review

**PASS_CODE_FOR_EXPLICIT_ONE_RETRY**, tested source
`4a9ef2bee701a089e06df7aea2f8ae0cc1fca2df`.

This is code and small-receipt review only. It does not issue an admission,
reserve resources, dispatch/retry a stage, modify a ledger or replace the old
failed result. Lead must independently review resources and issue exact new
helper/admission pins before any distinct attempt.

## Evidence

- Helper under review:
  `reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py`.
- SHA256:
  `747873b72ce6bcb2cfd3163bd89ce1560e4538d1069952d8deed176c349a67ef`.
- 13 implementer new tests independently rerun: **13 passed, 0.14 s**.
- 12 reviewer-owned negative/behavior cases: **12 passed, 0.15 s**.
- Complete logs: `OWNER_TESTS.log`, `INDEPENDENT.log`.
- CPU affinity 0,1; two-thread environment; CUDA hidden; external 90-second
  test limit. No GPU, actual child job, native weights, image or tensor reads.
  The implementer's 72-case result is not reused as this execution's result.

The reviewer used a new isolated worktree and changed only this report/test
directory. The production helper was not edited. Temporary fixture files were
used for failures and duplicate-directory cases. Actual production reads were
limited to the small plan/failure/preparation receipts and Git HEAD/status.

## Exact scope and identity

`manual_retry` is accepted only for `dev_extract`, with its exact three fields,
the hard-bound original failed RESULT path and SHA, original dev_plan result
path/pin, and one fixed `dev_extract_manual_retry_v1` directory. Arbitrary stage,
timestamp directory, extra fields or lexical traversal aliases are rejected.
The original failed receipt must remain STOPPED_NO_AUTO_RETRY,
MONITOR_UNREADABLE, same PREPARATION, and no remaining owned processes.

The independent cases re-pin synthetic wrong FAILED/COMPLETE states, stage,
reason or live-process counts to test semantic rejection after checksum checks.
The real original failed hash remains hard-coded and cannot use these fixture
overrides. Wrong hashes and changed preparation are also covered by the
independently rerun implementer cases.

`stage_inputs` is metadata-only and creates no attempt. Actual `main` creates
the fixed attempt with `mkdir(exist_ok=False)` under the existing shared lock,
before calling `supervise`. Independent tests invoke actual `main` with its
external admission/resource probes stubbed and a failing dispatch sentinel:
both an existing default directory and an existing explicit retry directory
raise `FileExistsError` without reaching `supervise` or changing prior files.
There is no loop, automatic retry, alternate-directory fallback or default
path that overwrites the original failure.

## Actual cumulative-read accounting

Read-only `stage_inputs` on the actual pinned receipts independently returned:

| Source | Charged logical bytes |
|---|---:|
| Original dev_plan | 37,533,999,575 |
| Original failed dev_extract | 30,632,210,276 |
| Previously consumed total | 68,166,209,851 |
| Original role cap | 274,877,906,944 |
| Explicit retry remainder | **206,711,697,093** |

The rule is the **sum of each attempt's maximum** of admitted hash bytes plus
nominal torch-load bytes versus process rchar. It is not the maximum of sums,
and not a sum that double-counts the two overlapping channels. An independent
crossed-channel example (plan rchar=100, failure hash=200) charges 300, not 200.
Admitted in-flight hash bytes are retained. Negative, Boolean, fractional,
missing or exhausted counters reject. No new plan can reset the budget; only
the remaining cap is passed to the fixed observed worker. This remains
conservative recorded logical I/O, not a claim of exact physical traffic or
unobserved work after an old snapshot.

Actual receipt hashes before and after review are identical:

- Failed RESULT:
  `18a351f58be8e1b9c561a7e585a7bd6e600118d674bf2beafb5192de337b9327`.
- Original plan RESULT:
  `dd765726eff1c937c31a1aca0f3cc47db8515f53790f84646ae1b94218857520`.
- PREPARATION:
  `34c174a6aa0c859a1b0a5f8b7dd800bbe189d51c770211b49c35d061e92b6431`.

`METADATA_AND_SOURCE.json` records the actual calculation, source pins and
absence of the retry directory at review time. It is not an admission.

## Downstream and inherited protections

`train_plan` without an explicit predecessor pin still reads the original
failed dev_extract and rejects it. Its optional predecessor must pin the exact
retry RESULT, with COMPLETE status, same PREPARATION, `dev_extract` stage,
correct attempt directory and original retry lineage. A success record without
that lineage is rejected. No other stage accepts this override. Subsequent
stages retain ordinary successful predecessor rules.

The unchanged PREPARATION still selects runtime
`581399eb309677c20fe7464791b4f18685e947a6` at
`/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017`; HEAD and clean
status were checked, with no runtime changes. AST comparison against reviewed
parent `bdd750a4489f4a5019cd4d33d3bde9571da344d2` confirms unchanged
`read_counter_snapshot`, `_supervise`, `supervise`, `OwnedSession`,
`abort_spawn_setup`, `shared_mount`, `fresh_gpu` and `acquire`. Thus the locked
periodic/final read, bounded first error, final wall enforcement and live
ownership protections are inherited rather than reimplemented. No teacher,
model, observed worker or original reader/writer implementation changed.

The original failure's cause remains **UNKNOWN_NOT_RECORDED**. The earlier
reproduced counter-replacement race does not retrospectively establish its
cause. The old attempt remains spent and immutable.

## Decision and execution boundary

The exact helper above may be considered by Lead for **one explicit new
dev_extract attempt only**, after new source/helper pins, live CPU/GPU/lease
and unchanged storage/checkpoint reserve checks are satisfied. The additional
fixed control directory must fit administrative space accounting. This review
does not approve those resources, modify a tensor claim, permit further retries
or start train_plan. The retry must complete and be explicitly pinned before
downstream progress.

To reproduce the CPU checks from the tested source with this report present:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout --kill-after=5s 90s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -vv -p no:cacheprovider \
  reports/st_world_policy/20261017/sp0_dev_extract_manual_retry_v1/test_manual_retry.py \
  reports/st_world_policy/20261017/correctness/sp0_manual_retry_v1/test_independent.py
```

Evidence was collected in the two separate runs recorded above; this combined
command is the equivalent reproduction, not a claimed third run.
