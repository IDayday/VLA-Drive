# IL77k–87k checkpoint availability and exact Navtest continuation

Completed: all fourteen registered evaluations contain12146 scenes/136 logs,
and all continuation workers exited. The full [checkpoint scan and paired
analyses](../scadv3_checkpoint_scan_20261026/README.md) are now available.

| Original IL checkpoint | Full Navtest PDMS | Difference from77k |
|---:|---:|---:|
| 77k | 88.144945 | reference |
| 80k | 88.157880 | +0.012935 |
| 86k | 87.886957 | −0.257988 |
| 87k | 87.931899 | −0.213046 |

80k's tiny gain is not supported as a meaningful improvement: the paired
log-bootstrap95% interval is[−0.208908,+0.234895] PDMS points. No RL parent or
training schedule is changed based on this exploratory Test scan.

The separately registered IL77-parent RL24.6k reaches90.689893, versus its exact
IL77 control88.144945: **+2.544948** points, log-bootstrap95% interval
[+2.051573,+3.047949]. NC/DAC/TTC means rise by0.547505/2.082990/2.198255
percentage points, with new violations/recoveries32/100,50/303,84/351.
Zero-score recovery occurs in392 scenes and new zero scores in78 scenes.
On the fixed IL-safe population131 scenes acquire a joint NC/DAC/TTC violation.
Thus the mean gain and mean safety gains coexist with remaining individual
regressions; this is one training seed and one inference seed, not repeated-run
stability evidence. No evaluated model reaches92.

SCADV3's highest observed intermediate is45k90.474315, above its fixed50k
90.324199 but below original CANONICAL50k90.513164. Compared with its50k,
45k has higher EP and DAC but lower NC/TTC; checkpoint scanning does not establish
that the safety modification outperforms the original recipe.

Fresh three-frame/current-DINO IL14k is82.899980, up8.612728 over its5k74.287253.
It remains a separate, still-training model family; do not call this an RL gain
or compare it with the paused warm-start temporal model as a matched control.

Automatic report generation succeeded. Automatic publication conservatively
refused because its helper stripped leading whitespace from `git status
--porcelain` and then misparsed the first modified path as unrelated. Direct
inspection found only the expected report files; a scoped manual commit/push
publishes them. The failure receipt is retained. No inference, score or model
weights were rerun or changed for this publication issue.

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
