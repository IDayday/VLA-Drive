# Independent matched SP1/SP2/SP3 control review

**PASS_NARROW_MATCHED_CONTROL_CORRECTNESS_ONLY**, bound to owner commit
`999328d5974bcf616696a93dfe347254a005cfe3`, controller SHA256
`c5c8521379b36c979bf8743f12c2a2baf80df5ca5d21941ba2fe9d453540c38c`.
This review grants no resource/dispatch permission and makes no claim about
actual extracted tensors or actual probe results.

24 tests passed; two author tests were deliberately deselected. The 18 selected
author control cases were independently executed, with six additional cases.
The deselected cases spawn live subprocesses on their hard-coded CPU2/3; this
review instead ran the real patched monitor with synthetic owned-process and
selector objects on CPU6/7. No live GPU or loader task was started or disturbed.
OS process cleanup remains the separate resource review's responsibility.

Checks covered the exact committed controls and affected behavior:

- All three PREPARATION/FIT_REGISTRATION and dev/train BINDING/SIGNATURE files
  match their committed delivery checksums. Checkpoint tags and identities match
  each arm's fixed inventory. SP1/SP2 retain144 tokens; SP3 retains288. Hidden
  dimension2048, FP32 master extraction identity, current-only observations and
  label-independent masks remain explicit. No checkpoint payload was reopened.
- The same ordered16 dev and64 train rows/logs are bound for every arm. Their
  observation-manifest and original fit-protocol pins agree. The six cache
  identities differ as intended; no arm silently reuses another arm's W cache.
- The actual frozen storage runtime reports commit
  `2a1e140a64d2817247ceb6214f7f2affa20d56c6`, clean. Its full git source delta
  from `581399eb309677c20fe7464791b4f18685e947a6` contains only the two approved
  storage files, whose small source hashes match. This establishes inherited
  compute/readout source identity; it is not a fresh numerical model test.
- The prepare adapter retains the original build function and all its existing
  checks, replacing only three arm literals, the declared memory-token count,
  the fixed original protocol path and per-arm source constants. The original
  checkpoint/source/observation and deployment-AST checks are not removed.
- The original supervisor validator actually accepts each arm's fixed CPU
  plan/fit fixtures and rejects wrong checkpoint identity/tag, old runtime and
  mismatched stage. Extract admission's real ledger/GPU branch was not executed.
- The five original stages and predecessor order remain unchanged. Failed or
  foreign predecessors are rejected. Legacy manual retry, old reconciliation,
  prior-stage override and storage-bridge exceptions are refused for these new
  arms, so a previous failed extraction cannot silently become COMPLETE.
- The only monitor-body patch rediscoveries after `done && active`. A synthetic
  stale active snapshot becomes a normal exit after rediscovery; an actually
  remaining owned descendant still reports
  `OWNED_DESCENDANTS_AFTER_LEADER_EXIT` and requests termination. Original
  observer/Counters code is reused unchanged; no counter totals or failure
  statuses are rewritten.

The prepared JSON paths deliberately bind the owner's immutable preparation
directory. A later copy must retain valid exact pins or generate/review a new
binding; path existence alone is insufficient. Runtime source, checkpoint and
common observation metadata are checked again by the original actual stage
call chain. The scientific comparison still requires completed matched
extraction and fit; none is claimed here.

Command executed:

```bash
taskset -c 6,7 timeout --signal=TERM --kill-after=5s 260s \
  env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /root/miniconda3/envs/ddp/bin/python -B \
  reports/st_world_policy/20261017/matched_probe_correctness_v1/run_review.py \
  > reports/st_world_policy/20261017/matched_probe_correctness_v1/TEST.log 2>&1
```

Measured test process:1.077s, peak RSS40,005,632 bytes, logical reads7,877,986
bytes, logical writes3,950 bytes, CPU6/7. Address space was bounded at1GiB and
the in-process timer240s. No Torch was imported. Separate known-small-source
inspection and repository administration are outside those process counters;
the complete review stayed below64MiB metadata/source reads and report output.

Accounting clarification: the preserved first test log's
`actual_process_launches=0` means **no experiment/monitor child launches**.
The runtime-source check did invoke three read-only git subprocesses. It does
not mean that the test process invoked no subprocess of any kind. No actual
tensor/DB/image/model reads, storage claims or ledger changes occurred.

This correctness PASS does not override any prior SP0 extraction failure or
permit automatic retries. Fresh stage/resource admission and actual typed-W
claims remain separate existing requirements. The Lead owns launch sequencing
and must preserve the active full-population PROFILE's GPUs and loader lock.
