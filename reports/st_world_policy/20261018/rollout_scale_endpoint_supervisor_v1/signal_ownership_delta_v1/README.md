# Deferred operator signals at owned-child transfer

Original seal: `380c8f5419a197fbd60d47e028caa029a2e5a0dd`, controller SHA256
`8b67d43d0652fac994778e89d7be42206b8e05b7da2379058bc22fdfdb957342`.
That source and its47-test evidence remain in git and the parent report folder.
This supplement supersedes its signal-window safety claim only.

Independent Systems reproduced a real CPU child left alive when SIGTERM was
sent after Popen created the child but before its handle returned to launch.
The old handler raised immediately, before jobs.append. Its original evidence
is preserved here in ORIGINAL_INDEPENDENT_REPRODUCTION.json; the independent
harness cleaned up its own fixture child. No real experiment was affected.

The narrow patch changes the execution handler to remember the first SIGTERM
or SIGINT. Monitor checks that request before starting another child, after
launch has registered ownership, at each loop and before successful completion.
It raises the same OPERATOR_SIGNAL reason only at these safe points; existing
owned-session cleanup then runs. Signals are not masked, and the child does not
inherit a blocked SIGTERM/INT mask. Model, argv, budgets, receipt identities,
runtime751 and scientific operations are unchanged.

Executed only the affected CPU suite:

```bash
/root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261018/rollout_scale_endpoint_supervisor_v1/signal_ownership_delta_v1/run_signal_checks.py
```

Result: **8 passed,41 deselected**; wall0.759727s, waited CPU0.615713s,
sampled RSS73,269,248bytes. Tests send real self SIGTERM and SIGINT from inside
the Popen wrapper after a real fixture child exists and before returning its
handle. They verify one durable charged attempt, exactly one reported started
invocation, no live child, a confirmed cleanup EXIT and restoration of prior
handlers. Existing LAUNCH publication failure, log open/close faults, threaded
and orphan cleanup, and two-loader scheduling are included. The unrelated
original47 checks were not rerun.

The first new fixture expected `str(signal.SIGTERM)` while the actual handler
receives integer15 (and2 for SIGINT). Both children were already cleaned up;
only these two string assertions failed. The fixture was corrected to int;
failure stdout/cost are retained. Total two-run waited CPU1.292041s, no resource
guard triggered. No GPU, model, image, DEV payload, metric cache, dispatch or
canonical score was used.

Independent reviewers must assess the clean delta. Execution remains
WAITING_CHECKPOINT_AND_FRESH_LEAD_ADMISSION. This implementation report grants
no real endpoint admission and does not claim independent review PASS.
