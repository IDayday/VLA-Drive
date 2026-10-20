# Independent limited review: dispatcher and deployment bytecode fix

Reviewer: effect_primary_evidence. No GPU task, model load, simulation or server
dispatch was performed by this review. The author’s real results are not inherited
as this reviewer’s evidence.

## Dispatcher `e1795aece89e4f54cfd07f5fb54788e8a9f0f093`

Source reviewed `run_campaign.py`, its fixed independent-stage tests and prior
export/controller contracts. Independently reran:

```
python -m pytest -q tests/action_effect/test_campaign_exports.py tests/action_effect/test_campaign_independent_stages.py
14 passed in 4.90s
```

The explicit default plan contains G1 delta, A0, B0, B1-trajectory, A1 and B1-effect.
Independent controls precede teacher-dependent arms; a stubbed real child failure
in G1 still allows A0/B0/B1-trajectory, while absent teachers leave A1/B1-effect
BLOCKED. A six-control G1 matrix is explicit and excludes nonexistent privileged
features. ScoreRefiner requires its own absolute-score teacher; it does not load
a delta teacher or external scorer to fill a missing input.

Expanded all/matrix execution requires an explicit aggregate new-update ceiling.
Reservations are written before child execution; failures retain reservations.
Existing named progress requires explicit resume, rollback is refused, and complete
terminal runs reserve zero further updates. Simulation accounting includes other
population prefixes, explicit arms and seeds beneath the same artifact root.
Each arm's evaluation task records failure separately, allowing remaining tasks
under the explicit continuation flag. Training completion remains separate from
held-out improvement and no NAVTEST/ACTIVE is dispatched.

**Limited approval:** affected source/CPU control behavior. The dispatch folder
lock and its UPDATE_BUDGET are not a global atomic allocator across independently
invoked dispatch folders or different artifact roots. Lead must continue to own
parallel campaign budget assignment. A cumulative counter scan is an accounting
snapshot, not proof of an atomic shared reservation between separate dispatchers.
A first review command used a nonexistent guessed test filename and collected no
tests; that harness error is excluded from the 14-pass evidence above.

## Deployment fix `629fffd8d227316e2a1c73502fe2325598c3d950`

The existing one-VLM/Euler10 source boundary was previously reviewed. This delta
adds a narrow exception for actual importlib atomic bytecode writes and persists
safe dependency diagnostics after success or failure.

An exception requires the `__pycache__` directory, strictly named
`NAME.cpython-NNN[.opt-N].pyc.NUMERIC_ID`, a real matching `.py` under explicitly
trusted source/library roots, and `O_WRONLY|O_CREAT`. Data reads, RDWR access,
unknown paths and real future/feedback/cache data remain prohibited. The matching
source is executable code, not a model supervision asset. Diagnostic traces contain
only sanitized paths and filename/line/function, not source lines, locals or file
contents. Worker failures retain the original error and attempt to preserve the
separate dependency receipt. Success pins that receipt.

Independently reran fixed author source:

```
python -m pytest -q tests/action_effect/test_online_deployment.py tests/action_effect/test_campaign_exports.py
18 passed in 4.94s
```

**Limited approval:** affected source/CPU dependency and wiring behavior. This
review did not run a new S0-only GPU job or claim a new inference latency,
trajectory match, scientific improvement or actual A/B deployment qualification.
