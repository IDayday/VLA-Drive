# Fixed C1 gradient diagnostic: deterministic cuBLAS environment repair

This is an external launcher repair, not a changed gradient definition or a
permission to disable deterministic algorithms. Runtime remains the frozen
`83679937fe9702be39f0184154fdf9ebdc4eb98a`; boundary 3584, four original rank
payloads, 32 original scenes, inputs, weights and budgets are unchanged.

## Preserved failed attempt

The v1 `RESULT.json` reports `DIAGNOSTIC_REJECTED` at the first frozen encoder
linear call: deterministic cuBLAS required `CUBLAS_WORKSPACE_CONFIG` before
PyTorch/CUDA initialization. The retained supervisor receipt reports 362.422
seconds, 73,834,874,003 nominal charged bytes, peak observed RSS 26,140,655,616
bytes, exit code 1 and `live_owned_remaining=0`. Its
`OWNED_DESCENDANTS_AFTER_LEADER_EXIT` classification is retained, together with
the actual diagnostic error, not rewritten as a successful computation.
There were zero optimizer updates and zero new official scoring calls.

No v1 result, log, checkpoint or control file is changed. The previous wrapper
is recoverable exactly at commit `b74c1ae0bba51236fffd5fcee2ab732dce118e19`, SHA
`ecb122a09b67ebc266c949bc5f7f7771c12b3b0cbee5712b57fcb7d10a722b20`.

## Verified original value

`C1_C4_final4096_readiness_v1/EXISTING_COMMANDS.json` records C1's explicit
`:4096:8` export and scoring environment. The actual C1 launch receipt binds
the original runtime/registration, and the original run identity records
`numeric_execution.deterministic=true`. The existing
`resources/launch_reviewed_flow_v1.py` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before `Popen`; the original `train_flow_policy.py:544` rejects deterministic
launches without a legal explicit value. The launch receipt itself does not
contain a complete environment dump; the statement about the exact value is
supported by the launcher plus the pinned command evidence, not invented
`/proc` history. Small source/receipt hashes are in `ENVIRONMENT_FIX_V2.json`.

## Minimal change

The wrapper now pins `runtime_environment` to
`{"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}` in both Lead admission and independent
review receipts. Parent preflight rejects missing or any other value, including
`:16:8`, before reading registration/admission payloads or starting a child.
`child_environment` explicitly sets the same value in the environment passed
to `Popen`. Child preflight rechecks it before importing PyTorch and writes an
immutable `RUNTIME_ENVIRONMENT.json`. An already imported `torch` causes refusal
instead of pretending that a late environment change repaired initialization.

The wrapper now binds only the separate v2 registration:
`gradient_balance_boundary3584_v2_controls/REGISTRATION.json`, SHA256
`fb55ca585688cd2ca19ea55af55628c60c91554c00f68ef4a6ae1821d22b06ec`.
Compared with v1, that registration changes only its output namespace. Existing
attempt/output rejection, no automatic retry, owned-descendant classification,
TERM/KILL/reap, read admission, RSS/output limits and 1,800-second deadline are
unchanged, confirmed by AST comparison.

## Author checks and required independent review

Five tiny subprocess checks passed in 0.240 seconds / 0.192 CPU seconds with
18,493,440-byte peak child RSS: missing environment, wrong legal value,
arbitrary invalid value, actual `child_main` reaching its intercepted PyTorch
import only after the correct environment and receipt, and late environment
setup after a simulated PyTorch import. The import was intercepted; no actual
PyTorch/CUDA/model/cache work or new score was executed by these fixtures.
The prior independent resource and cleanup fixtures remain applicable because
the corresponding functions are AST-identical.

The wrapper author does not approve this delta. Independent reviewers must bind
the v2 registration, unchanged source SHA, new wrapper SHA and exact
`runtime_environment`. After their approval, Lead's execution command begins:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261018/gradient_balance_systems_v1/supervise_gradient_once.py \
  --execute --admission /mnt/project/st-world-policy-20261016/audits/consequence/gradient_balance_boundary3584_v2_controls/ADMISSION.json
```

The admission path is the new Lead-controlled artifact to be generated, not a
claim that approval already exists. No real v2 execution was started here.
