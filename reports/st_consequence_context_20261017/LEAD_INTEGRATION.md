# Fixed-reference validation integration

The lead integrated `70777afd822ea9ba8e248e231ab4499be887ee7b` as
`e204cde8` and applied the shared-trainer patch in this directory. The trainer
still checks every official result and event/metric reconstruction. Its
four-candidate common-context check now compares validated reference **inputs**;
each execution array remains independently checked against its exact hash.

The affected integrated suites were executed on 2026-10-17:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_consequence.py tests/st_world/test_flow_training.py
```

Result: **59 passed**, 23 dependency warnings, 10.74 seconds. Full output:
`/mnt/project/st-world-policy-20261016/audits/consequence/context_mismatch_20261017/LEAD_INTEGRATION_TEST.log`.

This is integration evidence, not independent correctness approval or a complete
performance qualification. The earlier 240-update C profile failed before
update 163 after 162 complete updates; it is not a passed 240-update profile.
Its records and diagnostic weights are retained. A fresh bounded C profile must
use the new evidence schema and its own output identity after independent review.

The spatial runs retain their clean, immutable `fdf25f99` runtime. Their model,
teacher, optimizer and approvals are unaffected by this separate C metadata fix.
No active worktree or historical checkpoint was changed. The canonical simulator
and scorer, reference inputs, scoring configuration, and model training losses
are unchanged.
