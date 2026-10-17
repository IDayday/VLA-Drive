# Evaluation preregistration: actual execution binding v2

Status: **TRAINING_RUNNING** for SP0 and SP2 only. Recorded 2026-10-17T01:14:35.403439+00:00. This version binds already approved, actually started runs; it creates no approval, checkpoint, prediction, score or scientific result.

The complete machine-readable plan is [PREREGISTRATION_EXECUTION_v2.json](PREREGISTRATION_EXECUTION_v2.json). It inherits the definitions in the original [PREREGISTRATION.json](PREREGISTRATION.json) and [PREREGISTRATION.md](PREREGISTRATION.md) without changing any population, estimator, bootstrap, seed, contrast, shortlist or endpoint definition. Original file SHA256 values remain `a41c9ab06c5e05f9593deb8c4f06c52cc7f35942cc24bfa5b4dbb1f99b03e6a7` and `fd6af35d3c5e3e8ff1133b87e983a83f7b22a65a14d575019b78b6a34487e46f`. Only top-level status, the two arm statuses, actual `formal_runs` pointers and execution-version provenance are added/updated. The original files remain unchanged.

Both real runs use frozen training source `fdf25f99b702754787904215bf1c2a2b7515ea75`, the full 101592 train scenes across 1176 logs, seed 42, the original 20000-update learning-rate horizon and 1000 warmup updates, with a 5000-update screening stop. Full dev remains 1696 scenes across 16 logs. Log-paired bootstrap remains 2000 replicates with PCG64 seed 20261017; the shortlist remains at most one contrast/two arms for 20k. No scientific rule is amended.

| Arm | Actual run identity | State |
| --- | --- | --- |
| SP0 | `703ac3d75f4651a29688ed226de7ebf0593c12df359cc13086aa47106b442789` | TRAINING_RUNNING |
| SP2 | `4a2a36978a1e41170ce00b482d281ffb25c883bf6d501e64958775807f960156` | TRAINING_RUNNING |

Each machine-readable entry pins the actual REGISTRATION.json, RUN_APPROVAL.json and run identity.json by absolute path/SHA256. Their three independent reviews and full fingerprints match; both run identities record the launch checker’s VERIFIED approval. Registered full train metadata has 101592 current and 101592 ego record pins, with the approved exact teacher/PCA namespace.

SP0 pointers:

- `registration`: `/mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP0_fulltrain_seed42_official_5k_v1/REGISTRATION.json`; SHA256 `85624cb410dab8a531edc3a793ddc052ed1f905c488174289638f55bcf489738`.
- `approval`: `/mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP0_fulltrain_seed42_official_5k_v1/RUN_APPROVAL.json`; SHA256 `fb7ecf8bb57811f5535ed3a9f2f9cc7805088a15b0485e9a088bb3ba31aa22ed`.
- `run_identity_file`: `/mnt/project/st-world-policy-20261016/formal/spatial/runs/SP0_fulltrain_seed42_official_5k_v1/identity.json`; SHA256 `33b15316a564379f811bcef430e96842e1a13b366fe53212ab7344e5cdc12444`.

SP2 pointers:

- `registration`: `/mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP2_fulltrain_seed42_official_5k_v1/REGISTRATION.json`; SHA256 `86aa23add97a0b463bec1f74d9dbb2fc8b70a4282c9578b40f637c617bdb9690`.
- `approval`: `/mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP2_fulltrain_seed42_official_5k_v1/RUN_APPROVAL.json`; SHA256 `4d2a556bc9caa68408ba8283d9d1421bf29ece3eef773319495bd5f49a08ccc2`.
- `run_identity_file`: `/mnt/project/st-world-policy-20261016/formal/spatial/runs/SP2_fulltrain_seed42_official_5k_v1/identity.json`; SHA256 `46e1c623ab2fb914b7344c3a72187fa960211660c3f5e1929fb4e98360581448`.

SP1/SP3 and C0–C4 remain unbound. Their PROFILE/SMOKE runs are not formal arms. The retained original pending checklist describes historical/general prerequisites; this version resolves the actual reviewed-run pointers for SP0/SP2 only. Later arm bindings require another version.

The current `_formal_binding` function reads the actual registration, approval, role reviews, identity and config schemas. Its AST is identical in the Evaluation worktree, Lead tree and frozen fdf runtime. Because a completed 5000-update checkpoint/export has not been supplied, a lineage-only in-memory probe with `completed=None` is correctly rejected at the endpoint check. The other final predicates were checked against real configs. No fabricated endpoint or export was saved, and no formal analysis was run. All seven unbound arms still reject immediately. Details and hashes are in [EXECUTION_BINDING_VALIDATION_v2.json](EXECUTION_BINDING_VALIDATION_v2.json).

This file records execution state only. It makes no effect estimate, scientific ranking, selection, completed-budget claim or new launch authorization. Navtest remains unread and unrun.
