# FBC-01: preserve malformed-counter failure after child cleanup

Independent resource review commit `9b60ad61` tested the original controller SHA `10ece34548c88069e765de95c5053688ac104de5862e45fc01c342882bb6fa67` with 30 CPU cases, including seven real tiny owned child processes. It observed 29 PASS / 1 FAIL: corrupt COUNTERS was detected and the owned child was successfully reaped; a final receipt read then raised JSONDecodeError, and the exception handler called `owner.finish` a second time. The reaped PID no longer existed and FileNotFoundError obscured the original error. No unrelated process was killed and no owned process was left behind.

The fix only marks successful `owner.finish` as finished, skips repeat cleanup when later counter decoding fails, and retains the first monitor error's type/message. It does not alter the original supervisor helpers, budgets, build identities, numerical/model source or one-attempt policy.

New regression uses real local pipes and a fixture-owned session (no child dispatch). A malformed final COUNTERS must retain JSONDecodeError and call finish exactly once. Together with the original preparation/controller fixtures: **32 passed in 0.23 s**. The independent original failure evidence remains unchanged. No real build or ledger mutation was executed by the owner.

Fixed controller SHA256: `866c35bce29b2215def6f1fea215dd75d7a62dc650a39d15c6e861f170ec7216`.
