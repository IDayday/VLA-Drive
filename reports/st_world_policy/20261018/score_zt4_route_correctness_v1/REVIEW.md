# Independent zt4 score-route wiring review

Reviewer: `/root/spatial20k_correctness`, 2026-10-18 UTC.

Result: **PASS_CODE_WIRING_ONLY**. No dispatch or actual resource admission was performed.

Reviewed author commit: `cef257107dd802bc1b5ef9c88cb996a4c77fabe9`, clean isolated implementation workspace `/mnt/project/VLA-Drive-st-score-zt4-binding-20261018`.

Reviewed file: `reports/st_world_policy/20261018/consequence4096_score_route_extension_v1/score_once.py`.

Exact SHA256: `9432aa45d66512e8189e1cdfeb94361fb848e3e226644842e67d87c2d560e5af`.

The adapter pins original `/mnt/project/st-world-policy-20261016/resources/launch_completed_dev_score_v5.py` to SHA256 `2e62c84a2eecde3e9af585d0a6fb77f9ff33d27a2ec76cd0fa2b594b3a8f15c3`. It changes exactly the `ROUTE_HOSTS_V3` tuple by adding `training-vlawm-tsinghua-zt4`. Restoring that tuple makes the entire adapted AST equal to the original; no model, metric, export, registration, scoring, lock, budget, or clock function changes.

The actual wrapper `__file__` is supplied to the inherited launcher. The original file uses that value only for the launch receipt's `launcher_sha256`; this preserves honest execution provenance and does not relocate other dependencies.

Inherited v5 checks were inspected directly: both 4260-second time checks use the original immutable route and actual time; maximum two shared score slots, two workers, 1728 cumulative scoring calls, maximum two durable invocations, native3600/outer4200+60 seconds, complete1696-scene export, exact environment/host/route, existing-exporter exit, CPU quota/shared-flock filesystem, dispatch/export rank locks and passed child lock FDs all remain. Native `--retry-attempts 1` is unchanged, and no automatic launch retry/follow-up is introduced. Explicit manual resume still consumes the same original invocation and call budgets.

Nine author CPU tests were independently rerun. They cover exact AST delta and parent hash, changed source/allowlist rejection, live policy and campaign authorization, forbidden/arbitrary hosts, exact route boundary and expiry, completed/duplicate score rejection and two-attempt budget, and actual small C1/C4 route hashes/budgets. All passed. No checkpoint, export trajectory, metric cache, large index, GPU context or remote host was read or invoked.

Command:

```bash
cd /mnt/project/VLA-Drive-st-score-zt4-binding-20261018
python -B reports/st_world_policy/20261018/consequence4096_score_route_extension_v1/test_score_route.py
```

Actual stdout/stderr is preserved in `TEST_OUTPUT.txt`. This review does not claim that either C1/C4 export is already complete, that its exporter has exited, or that a route will still have4260 seconds remaining at dispatch. Those unchanged actual checks run in the original launcher. An expired route is rejected; this adapter does not supply a replacement execution lease.
