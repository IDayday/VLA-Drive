# UTF8 controller recovery, no loss/actor change

The first DELTA qualification failed after one disposable optimizer update when a runtime locale change made pathlib use ASCII to read the UTF8 dispatch policy. This was a UnicodeDecodeError in resource-boundary checking, not a trajectory/loss/optimizer arithmetic failure. The sole owner correctly stopped automatic progression; none of the three formal2000-update arms had started. CONTROL, TIME_RESIDUAL and HUBER had completed their three-update eight-card qualifications successfully.

All previous workers were verified exited by exact PID/start-time. Original owner/state/failure evidence is copied under controller_recovery_20261028T2250Z; the failed DELTA output remains intact. The recovery verifies every one of24 scientific source files against the successful qualification identity, binds the prior failure hash and refuses a live prior owner or any unrelated failure.

The execution fix explicitly propagates PYTHONUTF8=1 through remote worker startup and its children. Both temporary hosts passed a real CPU reproduction: after locale.setlocale(LC_ALL,'C'), the same dispatch policy still loads with UTF8 mode1. No action code, objective, calibration constant, model, data order, optimizer, scheduler or sampler changes.

The incomplete DELTA qualification is replayed from the unchanged registered63k parent into a different evidence directory; old partial logs are not appended or relabeled. Successful qualifications are reused. All formal arms still start independently from the registered original63k native model+Adam+RNG, never qualification weights. The recovered owner uses fixed hosts only, so the nearing temporary-host expiry creates no new migration obligation.

22 targeted tests pass, including duplicate-owner rejection, scientific-source hash rejection, explicit UTF8 requirement, preservation of failure ancestry and the existing loss/mask/gradient tests. Native training recovery mathematics are unchanged.

```bash
WORLD_POSTCOT_ROOT=/mnt/project/qwen-world-postcot-20261025/loss_designs_20261028 WORLD_POSTCOT_TASK=qwen_world_il_loss_designs_20261028 PYTHONUTF8=1 python -m tools.qwen_world_postcot.loss_designs_owner --recover-owner /mnt/project/qwen-world-postcot-20261025/loss_designs_20261028/controller_recovery_20261028T2250Z/RECEIPT.json
```

Launch once only through a durable owner. Source/state checks refuse duplication. This is an acknowledged engineering recovery, not a new research configuration or a claim of performance improvement.
