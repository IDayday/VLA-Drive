Actual execution binding v3 records **TRAINING_RUNNING** for SP0-SP3 as observed at 2026-10-17T01:38:17.410706+00:00. The [complete plan](PREREGISTRATION_EXECUTION_v3.json) extends [v2](PREREGISTRATION_EXECUTION_v2.json) only with SP1/SP3 actual registration, approval, launch and runtime identity pointers. Existing SP0/SP2 bindings are identical.

The original PREREGISTRATION.json and PREREGISTRATION.md remain unchanged. Populations, seeds, statistical estimators, invalid-prediction rules, contrasts, shortlist, all endpoints and the common learning-rate clock are exactly preserved. Removing only execution metadata reproduces the original plan object byte-for-byte after canonical serialization. No prediction, score, effect estimate or arm selection was created.

| Arm | Actual run identity | Execution state |
| --- | --- | --- |
| SP0 | `703ac3d75f4651a29688ed226de7ebf0593c12df359cc13086aa47106b442789` | TRAINING_RUNNING |
| SP1 | `df7eb9ed951262e4117209a8305330a721278d285eaa2c0b6aa4237161d8ef36` | TRAINING_RUNNING |
| SP2 | `4a2a36978a1e41170ce00b482d281ffb25c883bf6d501e64958775807f960156` | TRAINING_RUNNING |
| SP3 | `bc0680b480dc4913d56effb366a9429ee3668f506dc5e1ab62e0991debaa1127` | TRAINING_RUNNING |

All four runs bind frozen source `fdf25f99b702754787904215bf1c2a2b7515ea75`, full101592 train scenes across1176 logs, seed42, the20000-update common learning-rate horizon, warmup1000 and screening stop5000. Dev remains1696/16; paired-log bootstrap remains2000 PCG64 draws with seed20261017; at most one contrast/two arms may continue to20k. These are retained definitions, not newly chosen rules.

SP1/SP3 actual launch receipts match their registration/approval digests and VERIFIED runtime gate. Each actual identity contains101592 current and101592 ego pins, the approved full teacher and fixed384 PCA v2 application namespace. All three role reviews have the exact same run fingerprint and distinct reviewers outside implementer_agents. No SMOKE or PROFILE weights initialize these formal runs.

The machine-readable binding pins every actual path/SHA256. [Validation metadata](EXECUTION_BINDING_VALIDATION_v3.json) records actual schema and provenance checks. `_formal_binding` is unchanged from the already validated v2; no tests, scoring or formal analysis were rerun. No completed checkpoint or5000-update completion is claimed. C0-C4 remain unbound. The original pending checklist is retained as historical/general text; this revision only resolves real execution identities for the spatial first wave.
