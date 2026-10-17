# Next registered fixed-dev queue and C2 helper candidates

Dependency snapshot: **2026-10-17T04:39:15.456171+00:00**. Only existing C1/C2 512/1024 milestones are listed. This is preparation, with no resource reservation or launch authorization.

| Arm | Training update observed | Endpoint | Static metadata | Native target | Evaluation resources |
| --- | ---: | ---: | --- | --- | --- |
| C1 | 390 | 512 | READY | WAITING_CHECKPOINT | WAITING_RESOURCE |
| C1 | 390 | 1024 | READY | WAITING_CHECKPOINT | WAITING_RESOURCE |
| C2 | 269 | 512 | READY | WAITING_CHECKPOINT | WAITING_RESOURCE |
| C2 | 269 | 1024 | READY | WAITING_CHECKPOINT | WAITING_RESOURCE |

Each run has only its sealed initialization step-0 native receipt at the snapshot. Those small receipts have valid canonical identities, exact world4 inventory and matching file sizes; no weight payload was rehashed. No 512/1024 tag or receipt is guessed, and initialization/latest is never substituted. Exact evaluation registrations for these four endpoints do not yet exist.

C1 actual registration, approval, launch, runtime identity, config and 6714 source match the original ten-job intent. C2 matches the separately pinned additive intent and actual deployed training identity. Both training leases are valid at the snapshot: C1 on tsinghua-zt4 GPUs4–7 until 2026-10-19T02:15:40.967189Z; C2 on rl-zt2 GPUs4–7 until 2026-10-19T03:11:42.537643Z. These are training leases, not evaluation-slot admissions. The fixed 1696/16 dev manifest and metric-index token/log mapping match; no cache payload was reread.

After the snapshot, Lead launched C3 (PID1184046). The exact new LAUNCH/registration/lease is pinned in RESOURCE_UPDATE.json. Thus rl-zt2 slots0–3 are protected C3 training and slots4–7 are protected C2 training. No fresh idle claim is made, no slot is held for this queue, and cache-debug or real training resources are not shared, paused or preempted. Both candidates retain the existing rl-zt2 host gate; no alternate host or automatic rerouting is enabled. The old C3-null evaluation addendum remains immutable and is still rejected by the candidates; no C3 runtime identity is filled here.

Once exact checkpoints exist, Lead can serve ready milestones as qualified resources become available. Preserve both 512 and 1024 diagnostics. Each export stays one GPU/world1, full1696, seed42, one original Euler10 trajectory, FP32 masters/TF32 off. Campaign maxima remain four independent exports and two CPU score jobs with two workers each; these maxima do not reserve capacity. Per checkpoint: 1GiB output reserve; at most two export invocations (7200s inner/7800s outer); at most two score invocations (3600s inner/4200s outer) with a single 1728-call ledger across resumes. Scoring waits for full export COMPLETE and process exit.

The existing v1 helpers remain byte-identical and support the original ten jobs, including C1. They correctly reject the C2 additive namespace. Lead authorized two v2 **candidates**, kept only in this private audit directory for independent review before deployment. No resources helper was replaced.

| Pinned additive C2 field | Deterministic v1-schema admission field/check |
| --- | --- |
| training_registration / training_approval / training_launch / runtime_identity_file | training_binding registration / approval / launch / run_identity_file; every pin must equal TRAINING_BINDINGS.bindings.C2 |
| training_binding_ref | Exact pinned TRAINING_BINDINGS.json; no arbitrary plan path |
| bindings.C2.config | training_config; must equal registration.artifacts.config and runtime identity config |
| bindings.C2.registered_execution.world_size | training_world_size=4; execution must equal actual registration |
| runtime_identity / training_source_sha / evaluation_source_sha | Exact actual non-null runtime identity and 6714 source |
| comparison / control_arm / control_job_id | Preserve C2_minus_C0, C0 and matching fixed step; do not accept another pair |
| job_id / stage / expected_completed / roots | Exact existing C2 512,1024,4096 IDs and distinct fixed roots; no new scientific endpoint |
| All original raw intent fields | Retained in normalized job; scorer requires entire C2 job equality plus exact additive pin map |

Only five explicit additive files are admitted: POLICY, INTENTS, COMMANDS, TRAINING_BINDINGS and EXECUTION_PLAN. Their inherited population/inference/score/dedup/resource contract must equal the original policy. C2 uses the existing frozen CLI commands. C3-null bindings reject before locks, directories, admission or Popen, including after its real training launch until a new immutable evaluation binding is reviewed. Original ten jobs take the original v1 branch.

Implementer validation: **29 bounded metadata checks passed**, including actual C2 512/1024/4096 pins, unchanged original-ten routing, C3-null/undeclared endpoint/wrong control pair/wrong source/wrong runtime/wrong population or budget/pin rejection, and actual step0 rejection for a fixed512 request. Native/manifest/budget validator ASTs remain unchanged. The scorer candidate retains the export rank-lock lifetime, durable admission before Popen and inherited descriptors; these were checked structurally. Both --help commands exit0; syntax passes. Only AST-extracted functions and help ran; no launcher main, GPU, model, scorer, export or real admission executed. This is implementation validation, not an independent approval.

Private candidates/diffs/validation: `/mnt/project/st-world-policy-20261016/audits/evaluation/NEXT_REGISTERED_QUEUE_v1`. [HELPER_HANDOFF.json](HELPER_HANDOFF.json) pins exact originals, candidates, five additive files and validation. [QUEUE.json](QUEUE.json) preserves the point-in-time dependencies; [RESOURCE_UPDATE.json](RESOURCE_UPDATE.json) records the later C3 launch. No prior evaluation result was re-audited and no preregistered definition, seed, comparison, endpoint, bootstrap or Navtest boundary changed.
