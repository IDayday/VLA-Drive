# CPU regression and lifecycle evidence

Original sealed source380c8f54: **47 passed**, one existing pynvml deprecation warning,
pytest3.72s. The enclosing bounded runner measured wall4.949350s, waited child
CPU4.708766s and sampled aggregate peak RSS643,301,376bytes. Exit0; no budget
guard was hit. Source SHA256 is in SOURCE_PINS.json; complete stdout and actual
execution record are TEST_OUTPUT.txt and TEST_EXECUTION.json. CLI --help and
--status were executed and captured separately.

The subsequent narrow SIGTERM/SIGINT ownership patch has separate affected-only
results in signal_ownership_delta_v1/README.md; these47 results are not relabeled
as a full rerun of that later source.

Executed command:

```bash
/root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261018/rollout_scale_endpoint_supervisor_v1/run_cpu_checks.py
```

The test runner bounds its own pytest process/session to120CPU seconds,2GiB
actual RSS,32MiB output and120wall seconds. It does not use RLIMIT_AS around
torch imports. No real model, checkpoint payload, images, metric cache, GPU
queries or canonical scoring were used. Fixtures use explicitly synthetic tiny
native files, reviews, process identities and provenance.

Coverage checks behavior rather than only command shape: exact source/reviews,
sealed world4 argv and one immutable official1728 ledger, at-most-two loading,
exact own LOAD_AUDIT plus master provenance, wrong checkpoint/manifest/scene
population refusal even after recomputing a signature, stale PID/rank/audit,
all partial attempt directories charged, second invocation explicit, third
refused, fresh storage refresh after five minutes with stable science/lease,
science mutation refused, full4096 terminal contract with no4095 fallback,
native pending/partial rejection, real threaded/reparented CPU children cleanup,
real child cleanup after LAUNCH write or log-close failure, known pre-Popen
log-open failure receipt, unknown GPU ownership refusal, driver-memory cap and
bounded nvidia query timeouts. CPU phase verifies owned exporter release without
requiring foreign GPUs to remain idle. External mature helper extraction is
checked without executing the old launcher's top-level code.

Earlier draft outputs are preserved. TEST_FIXTURE_STATE_FAILURE.txt records an
initial failed test whose fake process row omitted the newly required state;
the fixture was corrected. Initial/intermediate and preseal passing runs are
not claimed as the final-source result. TEST_COST_LEDGER.json lists retained
execution receipt costs. The last source check supersedes those drafts.

All tests are implementation evidence. Independent review is pending on the
clean commit; real4096 strict load, fresh host/UUID/lease, four-rank RSS/latency,
shared filesystem behavior, real canonical score completion and paired DEV
result remain WAITING_CHECKPOINT / not executed by this agent.
