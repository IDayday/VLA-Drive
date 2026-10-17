Execution binding v4 adds the actual C0/C1 registrations, approvals, immutable invocation LAUNCH receipts and runtime identities observed at 2026-10-17T02:47:58.378235+00:00. [The complete plan](PREREGISTRATION_EXECUTION_v4.json) preserves all W bindings and scientific definitions from [v3](PREREGISTRATION_EXECUTION_v3.json).

| Arm | Actual runtime identity | Observed execution state | First invocation |
| --- | --- | --- | --- |
| C0 | `484fc7d2db6ceba5fb531e4f42423471a2730df4a2afc8ce8ce711dbfc435aa9` | LAUNCHED_INITIALIZING | `1792205094228388879` |
| C1 | `9c279380654e586896608fc66be9dbcc45181d953b40271d6374dba5fc0847df` | LAUNCHED_INITIALIZING | `1792205106721054236` |

Both runs use frozen source `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`, the original common S0 with fresh optimizers, full101592 train scenes/1176 logs, seed42, world4/micro4/caller accumulation2/global32, the original4096-update clock with warmup100 and absolute stop4096. Native snapshots are scheduled every512 updates. C0 binds zero training score calls; C1 binds hard total admission cap532448 with one eligible retry per request. These are existing approved execution facts, not changes to scientific definitions.

Actual runtime identities bind the complete Flow manifest identity `7255bb2db29fd0bf2748215edb8abbeed0352f8bfe7f841e1fabd9682b4f6797`. The full manifest and its existing independent verification are inherited; no image, ego, metric-cache or checkpoint payload was reread or rehashed for this metadata update. Each approval and its three distinct role reviews match the actual registration fingerprint. Actual launches match registration/approval hashes and the VERIFIED gate; no diagnostic weights are used.

The plan's top-level TRAINING_RUNNING state and SP0-SP3 states are inherited observations from v3. Per-arm states above reflect this version's small receipt snapshot. LAUNCHED_INITIALIZING means actual launch and runtime identity exist but a RUNNING status receipt was not yet observed; it does not claim an optimizer update. No checkpoint completion is claimed.

Dev remains1696 scenes/16 logs; consequence diagnostics remain512/1024 with common endpoint4096 and no scientific early elimination. Paired-log bootstrap remains2000 PCG64 replicates with seed20261017. Estimands, zero transitions, high-score degradation, progress/stop diagnostics, invalid handling, contrasts, W5k/20k clocks and the maximum-two-arm shortlist are unchanged. Canonical JSON equality recovers v3 after removing only the two new execution bindings/status deltas, and recovers the original plan after removing all execution metadata.

C2-C4 remain **NOT_REGISTERED / NOT_RUN**, with no fabricated formal binding. Original pending text remains historical/general. `_formal_binding` source is unchanged and the actual Flow identity/config/approval schema satisfies its metadata predicates; a valid complete checkpoint/export at a registered endpoint remains necessary. [Validation](EXECUTION_BINDING_VALIDATION_v4.json) records the checks. No tests, analysis, export, scoring, GPU work or dispatch were run. This revision grants no evaluation execution authorization; `formal_launch_allowed` stays false.
