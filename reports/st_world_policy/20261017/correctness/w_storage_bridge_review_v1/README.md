# Independent actual W bridge review — blocked on one bounded-read defect

2026-10-17. Reviewed source `ba09bf88d677e62b9503e1391ca47e9d02f5a8c7` in
`/mnt/project/VLA-Drive-st-w-storage-bridge-review-20261017`.

**FAIL / execution remains blocked pending a narrow fix.** Independent execution
of the owner's focused suite passed all 44 cases; the added FIFO counterexample
failed: **1 failed, 44 passed in 7.24s**. Full unmodified failure output is
`TEST_FAIL.log`. This is not an actual campaign extraction failure.

## Exact source identity

| File under w_storage_bridge_v1 | SHA256 |
|---|---|
| storage_bridge.py | `6790d988f11b14cc06c64250d4dedb7f1336843f4f8a589202ceefea6fbd7134` |
| observed_cli.py | `c4dbdd4bcbbcf9b4059236b715289347d9da852b206f3f121bd24f1e99eaade0` |
| supervise_bridge.py | `d533ab6dafba1b18b85d7d581860f7a98f7e2771a3f97e8b7c0574ab30516dda` |

## Confirmed defect and minimal fix

`storage_bridge.py:58`, `Reads.raw`, opens the path with
`os.O_RDONLY | os.O_NOFOLLOW`. Its regular-file check occurs only after open,
at lines 60–62. A canonical FIFO without a writer blocks inside open indefinitely.
Neither the bounded read nor its rchar checks execute. This function is called
in parent preflight before the original supervised child wall limit starts;
the 15-second git-operation timeout does not cover it.

`test_independent.py::test_fifo_input_rejects_before_waiting_for_writer` creates
only a private temporary FIFO and invokes the real reader in a CPU subprocess.
It expects immediate rejection; the subprocess instead reaches the one-second
test timeout. `subprocess.run` cleans up this owned test child. No real control
file, ledger, model or runtime was changed.

Required fix: add `os.O_NONBLOCK` to this open, retaining `O_NOFOLLOW` and all
existing descriptor regular-file/private-link/version/hash checks. This also
closes the same class of wait if a file changes to a FIFO before open. There is
no need for a new I/O framework or changes to computation, identity or budgets.
Then rerun this exact negative case and the affected reader/child-import checks;
retain this original FAIL evidence.

## Other reviewed seams

The implementation otherwise follows the minimal boundary from review
`41fa5aed636b11608fa6a8191a8a22a3f06a9cd0`:

- Two exact donor modules, source commits, whole-tree two-file delta, clean
  original compute source and ten unchanged dependencies are checked. Donor
  bytes are verified before installation and those same bytes are executed.
- Canonical module names and parent-package attributes are updated together.
  Stale/mixed imports, prior torch/compute imports and partial/repeated install
  are rejected. The original budget module remains the shared class.
- Original compute source, PLAN/signature/cache identity remain unchanged.
  The observer's four documented mechanical edits preserve its original
  computation, counters and internal native checks. The wrapper invokes the
  original supervisor validation and exact child call, and retains no-pin
  behavior, migration receipt, lease, checkpoint-floor and cleanup checks.
- Read-only bridge checks do not migrate or reserve storage. Actual donor
  validators and original budget behavior were exercised on temporary v2 W
  ledgers, including aggregate/member/cap/free/checkpoint rejection paths.
- The adapter's 128 MiB source/control read allowance and per-file 4 MiB bound
  are local checks, not an independent resource grant. Original supervision
  remains responsible for the extraction lease and whole-process accounting.

## Reproduction and scope

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -c \
  'import resource,runpy; resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30)); runpy.run_module("pytest",run_name="__main__")' \
  -q reports/st_world_policy/20261017/w_storage_bridge_v1/test_bridge.py \
  reports/st_world_policy/20261017/correctness/w_storage_bridge_review_v1/test_independent.py
```

CPU2, 2 GiB per-process address-space ceiling inherited by children, outer
180-second timeout, no visible CUDA. Tiny controls/source fixtures only. Owner
cases read the existing small signature and use private sparse git fixtures;
no real ledger or model payload was opened. No dispatch, real W extraction,
teacher/model qualification, throughput, or fresh resource approval occurred.
