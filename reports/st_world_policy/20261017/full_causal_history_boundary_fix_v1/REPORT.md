# H64 finite wrapper boundary fixes

Base: Lead `4568c578`; independent negative review `b20c9b5b` retained via cherry-pick `760b6f9e`. Earlier tree was not fast-forwardable, so a new isolated `/mnt/project/VLA-Drive-st-history-pilot-boundary-fix-20261017` was created. No reset/stash, live TRAIN source change, ledger/claim mutation, raw input/model/GPU read or dispatch.

| Finding | Narrow repair | Current evidence |
|---|---|---|
| H64-1 blocking START/RESULT lock outside monitor | Scoped adapter wraps the **actual existing flock acquisition** with LOCK_NB. No probe/reacquire race and no alternate ledger. Same behavior around child metadata admission. A separate admitted control directory holds a real nonblocking supervisor lock throughout START, child, RESULT and failure handling. Wrapper-entry absolute900s work alarm includes controls/publications; remaining work is passed into existing monitor, reserving5s for terminal evidence and retaining existing40s cleanup bound. | Actual original `StorageBudget._locked` fixture rejects held lock immediately and succeeds after release; second supervisor rejected; real SIGALRM fixture fires/restores handler. |
| H64-2 controls before core BOOT_IO omitted | `/proc/self/io` and monotonic baseline captured at wrapper module entry, before argparse/controls/helper import. Existing core BOOT_IO/START receive these immutable values, retaining repaired Linux wait accounting. Recompute final charge after guarded RESULT publication. | AST checks entry baseline predates controls and is supplied to core; prior accounting tests preserved. No child count is added twice. |
| H64-3 exceptional monitor leaves only START | Catch failure from START/monitor/RESULT and emit immutable bounded separate RESULT.json; no ledger dependency in failure path. Conservatively charge full16GiB when terminal accounting unavailable. Cleanup status UNKNOWN on exception, NOT_STARTED only before monitor; never invent a cleanup PASS. | Injected START lock contention, monitor exception and RESULT contention each produce FAILED_NO_AUTO_RETRY. Existing receipt overwrite rejected. Failure writer works even if ledger API is poisoned. |

## Separate control evidence

New proposed evidence root: `/mnt/project/st-world-policy-20261016/audits/history_data/causal_history_cost64_v2_controls`.
It is **not created here**. Lead must explicitly include it in the independently reviewed resource admission and create the plain directory. It has no claim ledger or model data: maximum1MiB and8 tightly named files, including lock/invocation/result/retained temporary files. Metadata artifact claim is reduced to at most31MiB, so the combined cap remains32MiB. Current templates are proposals, not approval. Canonical path and symlink/filename checks apply; exclusive immutable creation preserves previous failed evidence. Holding the original global ledger elsewhere cannot block this failure writer.

If START is refused, actual build never begins. If RESULT is refused after child completion, the control result is still FAILED_NO_AUTO_RETRY, with actual returned cleanup evidence when available. Intermediate ledger result is marked BUILD_RETURNED_PENDING_TERMINAL_ACCOUNTING; only separate terminal control can report COMPLETE after post-publication read accounting. No automatic retry or deletion is introduced. More detailed underlying monitor exception cleanup is UNKNOWN until independently inspected, not inferred from no returned object.

## Tests and scope

`TEST_LOG_FINAL.txt`: focused wrapper/boundary suite, 20 PASS /1 deselected. The excluded old tiny child fixture launches CPU6/7, outside this repair's CPU8/9 test authorization. All executed tests ran on CPU8/9 with CUDA hidden and a40s outer timeout; no production child was launched. Old independent `test_reproduce_*` fixtures/report are deliberately preserved as evidence of the old code, not silently rewritten to produce green resource approval.

The commands were not wrapped in a whole-test `/proc` read/RSS accountant, so this report does **not** claim measured test read/RSS budgets or sign resource PASS. Test invocations totaled under10s from pytest output; fixtures touch only tiny temporary files and control/source code, with normal Python/PyTorch imports. Independent resource review should meter its rerun. No actual shared disk quota, image/log readability or64-scene cost was validated.

## Remaining admission

Independent reviewer must recheck exact final wrapper and proposal. Lead alone may issue actual admission/31MiB original-ledger metadata claim plus1MiB evidence reservation after active writer availability. The actual2303GiB checkpoint receipt and1TiB free floor remain pinned and must cover the new actual start. Full history remains unadmitted due cost/scale work; no scientific metadata/trainer/selector changes were made.

CLI remains `run_pilot_once.py --admission <actual independently issued JSON> --admission-sha256 <actual SHA>` under explicit taskset6,7/CUDA hidden. Actual paths/hashes are in updated PROPOSED_CONTROLS. No command was dispatched.
