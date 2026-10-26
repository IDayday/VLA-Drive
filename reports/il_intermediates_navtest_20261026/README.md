# IL77k–87k checkpoint availability and exact Navtest continuation

The user requested additional IL checkpoints between77k and87k. Native files
were inspected, not inferred from directory names. Complete80k and87k were
hardlink-pinned before rolling retirement and registered for full Navtest.
77k is already88.144945;86k is already87.886957.78k/79k/81k/82k/83k/84k/85k
retain retirement metadata but no model.pt and cannot be evaluated.86k's
earlier evaluation pin preserves its complete weights despite native retirement.
No88k evaluation was added outside the requested interval.

The two additions use the unchanged original434cba8 evaluator, current three
views, native Qwen/source ActionDiT, canonical NAVSIM v1.1, single shift5 ODE
K10, seed2101,12146 scenes/136 logs each. They are user-requested exploratory
Test observations, not RL-parent reselection or new training.24,292 new logical
single-candidate inferences are added; the existing12-model scan is retained.

The earlier vla-zt3 worker records stopped writing for over100 minutes and SSH
returned authentication failure. Ten actor rank locks and ten job-worker locks
were all acquirable. With those locks held, two observations confirmed the
invocation payloads unchanged. Only those stale jobs received a separate
TRANSPORT_LOST_REQUIRES_RESUME receipt with **unknown process exit**, not a
successful exit. Original scene and invocation JSON remain unchanged. Their
raw copies/hashes and lock evidence are retained in the new control root.

The original owner therefore ends COMPLETE_WITH_FAILURES and closes its finite
vla-zt3 admission. A separately registered continuation uses fixed hosts only,
inherits seven completed full summaries and successful worker receipts, reuses
qualified native/merged parity, and restarts only missing shards with the same
world4 layout, source, weights and scene-bound noise. Rank locks remain the
single-writer guard. No old process is declared exited merely because SSH fails.

Stale RUNNING invocation records are accepted by the new **analysis-only**
summary only if their exact raw hashes and released-lock evidence match the
transport manifest. Measured loading/forward/scoring/worker time remains in
the cost ledger. Lost-host tail costs and actual exit times remain unknown;
physical costs are explicitly lower bounds. No fake end timestamp is inserted.
All full scene/score/population seals and official denominator checks remain.

Nine targeted tests passed. On the actual complete IL77 control, all seven
original cost fields exactly match the previous summary. Compilation passed.
Actor, sampler, reward, deployment and training sources are unchanged. Existing
normal training is neither stopped nor modified.

Control and immutable registrations:
`/mnt/project/qwen-sourceflow-native-20261018/IL77_87_NAVTEST_20261026T0755Z`.
The prior scan and both prior finalization failures are retained as execution
history; no unsuccessful attempt is relabeled as successful. Final paired
results and the checkpoint curve will be produced by the read-only finalizer.

Executed command forms (do not duplicate an owner that already exists):

```bash
python3 -m unittest tests.rl.test_finite_navtest_scan tests.rl.test_compare_rl_checkpoints tests.rl.test_navtest_transport_costs -v
/usr/bin/python3 -m scripts.analysis.qwen_sourceflow.resume_navtest_scan \
  --registration /mnt/project/qwen-sourceflow-native-20261018/IL77_87_NAVTEST_20261026T0755Z/registration.json
```
