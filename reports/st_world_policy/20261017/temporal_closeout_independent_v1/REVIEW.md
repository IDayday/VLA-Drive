# Independent narrow closeout: exact32 temporal preparation

Reviewer: `/root/temporal_narrow_review`. Verdict: **PASS for the reviewed code and bounded fixture scope**. This is not a real T3/T4 training result, a live resource admission, or formal training approval.

Reviewed source:

- Temporal fix `676e3e88d48d9444ab2cb928242d196381631b26`, parent `a8f8cb879a806ea674ed7be9b1ff7d82483e8b2d`, in `/mnt/project/VLA-Drive-st-temporal-combination-20261017`.
- Exact32 history wrapper `0d44879b20d26ec105d90973ae525aef00a2fd1c`, in `/mnt/project/VLA-Drive-st-teacher-channel-control-20261017`.
- Prior findings: `2022a1dab7c1db594efcc34ee1dd7e1abbe19f3d`, `temporal_composition_independent_review_v1/REVIEW.md`.
- Report-only worktree starts at `9ec7807ddbe534eed939702461b851891d074cd0`; implementation was not changed by this reviewer.

Read AGENTS and the review/admission boundaries in `CODEX_MULTI_AGENT_START_20261017.md` and `WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md`. No unrelated historical pipeline was re-audited.

## F1: runtime exact population boundary — fixed

`temporal_composition._validate_exact_population` is called by the actual `validate_composition_recipe`, not only the CLI. It requires the non-full profile schema and an integer sample limit in 1..32. It binds the original SMOKE population count, original indexed descriptor, qualification-plan request identity and ordered scene list, source index, train split, token/log pairs and uniqueness. The same helper runs while preparing a new version.

Actual tests reject 0, 33, 64 and 256 with re-signed parent/current recipes, full-profile admissions, replacement/reordering, changed log/source/request identity and count. The original T2 validation is still called after reversing exactly history/cell/delta; this does not authorize a changed optimizer, teacher, precision or scientific coefficient. The actual deployment registration must still bind its execution topology and maximum four diagnostic updates.

## F2: failed preparation blocks corrected retry — fixed

`prepare` now creates a new private staging directory, validates descriptor/history attachment/PCA scope/full config there, writes a complete receipt containing input and artifact pins, then publishes the output through one exclusive atomic symlink. Artifact pins target the stable staging-version directory. A failed attempt retains `PREPARATION_FAILED.json` and its files; the output namespace remains unpublished. A corrected history dependency can therefore retry without overwriting failed evidence.

Tests inject failure at descriptor, attachment, scope and config boundaries. In each case the failed attempt remains byte-identical, corrected retry succeeds, identical successful retry reuses the published version, and changed inputs after success refuse. Legacy partially written directories remain preserved and refused; the implementation does not silently delete or reinterpret them. Directory publication is local-filesystem atomicity; this review did not simulate host power loss or filesystem failure.

## Fixed32 owned-controller binding — accepted

`run_history32_once.py` SHA256 `14f70bab1bc5f1d6d66946d71a411b05bef4b05e7d614166d948cf81f1dc0f40` loads the SHA-verified original `a8499c...` controller and changes exact-count constants only: population/root, current CP pin and remaining bytes, one-or-two admitted CPUs, admission schema/status, wrapper child path and required wrapper source pin. The `4b2f0f...` monitor and original ownership/accounting/publication/cleanup logic remain unchanged.

The wrapper additionally validates the exact builder command and budget; fixed32 selected SHA; all current builder and controller source hashes; current CP status/lease/free floor; fresh host/lease; exact 900s work +40s cleanup, 16GiB logical reads, 4GiB RSS, 31MiB artifacts +1MiB control, one invocation; packed current-only metadata; and no GPU. Source reads use bounded, nonblocking regular-file descriptors and reject symlinks/hash changes. The bootstrap read baseline is passed into the inherited monitor; the absolute deadline is handed off without starting a new 900s allowance.

Policy handling is inherited, not bypassed: actual `metadata_storage.metadata_writing` rechecks the pinned policy, opens the original storage ledger, checks unchanged core and declared registration/root/hash, counts existing files, and charges the explicit claim. CP capacity is checked again by the original callback throughout the invocation. This review did not open/write the live ledger or assert that an as-yet-unmaterialized claim exists.

Actual fixed manifest was separately read (10,610 bytes): SHA256 `7c78c58c40456649a93d152f9925ea5e1faa7e08bb65e98cd6c4af3b0c52b9b8`, 32 unique scenes, 2 logs. No image/log pickle or teacher tensor was read. Temporal implementation file SHA256: `6b8ebaa252956bd60f0d3eec7ce9c372a8abbefa2336cda8a60b839aaac74bc2`.

## Executed evidence and remaining admission boundary

`TEST.log` and `TEST_METRICS.json` record this review's execution, not inherited author results:

- 43 temporal tests passed; 8 wrapper tests passed.
- Four extra independent refusal checks passed: Lead as reviewer, duplicate CPU IDs, expired latest-start time, and insufficient remaining lease.
- All 15 actual small runtime source pins matched.
- 4.487s wall; 627,290,112 bytes peak RSS; 142,396,348 logical bytes read inside the test runner. The separate final manifest/hash read adds 39,377 bytes; small source-review reads are outside this runner counter and were well below the total 256MiB review allowance.
- CPU8/9, CUDA hidden, no real builder, no actual monitor subprocess, no ledger access, no SSH. Fixtures at external descriptor/history/PCA boundaries are explicitly synthetic.

```bash
cd /mnt/project/VLA-Drive-st-temporal-closeout-independent-20261017
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 
# The next command is the exact finite runner; the environment is also set within it.
timeout --signal=TERM --kill-after=5s 180s /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/temporal_closeout_independent_v1/run_tests.py
```

Before the one real exact32 metadata run, Lead must materialize and independently bind the actual current policy/registration/CP/admission, select available CPUs and verify the combined storage reservation/control adoption. Do not reuse the old2303GiB CP or a draft registration. The wrapper must retain the reviewed source bytes. This narrow code PASS permits that next preparation step; it does not declare history COMPLETE or real T3/T4 forward/backward PASS.
