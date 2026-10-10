# Formal readiness fix matrix

| Requirement | Entry and implementation | Test | Actual evidence | State |
|---|---|---|---|---|
| R1 independent evaluation and formal gate | evaluate-base / qualify-base / freeze-base; base_qualification.py | Base qualification tests | BASE_EVALUATION.json; BASE_QUALIFICATION.json; formal_freeze_rejected.log | CODE COMPLETE; full navtest measured; formal acceptance unregistered |
| R2 reproducible initialization | selector_module, randomness.py, router.py | process-independent init, retained Router rows, resume RNG | REAL_SELECTOR_INIT.json; round_*_scorer_identity.json | COMPLETE |
| R3 common newest-valid resume | resolve_resume; both trainers | stale result, repeated interruption, corrupt/partial, explicit-old fork | REAL_EXACT_RESUME.json; REAL_RESUME_RESOLUTION.json; REAL_SCORER_AUTO_RESUME.json | COMPLETE |
| R4 data commit and recovery | build_round; RoundDataTransaction | six fault points, content-stale and legacy orphan | REAL_STAGE_RESUME.json; original runtime DATA_COMMITs | COMPLETE for E2; E3 real input UNTESTED |
| R5 workflow/expansion separation | check_expansion; accepted registry; previous_serving snapshot | 1→Scorer-only2→3; rejected pool, configured gates, standalone gate | REAL_WORKFLOW.json; round_003_WORKFLOW.json | COMPLETE |
| R6 genuine residual | E6_unconstrained_residual.yaml → residual_v2 | zero delta, real dependency, gradients, frozen Base | RESIDUAL_V2_PROBE.json | COMPLETE engineering; science UNTESTED |
| Retrieval scale | RetrievalIndex.query_many | reference equality/ties/exclusions | SCALE_PROBE.json | COMPLETE exact optimization; full-scale estimate only |
| Configuration/space | config.py, storage.capacity_plan, bootstrap call sites | nondefault knobs and disk shortage | CAPACITY_PLAN.json; COST_PROBE.json | COMPLETE checks; no automatic deletion |
| Regression/GPU/DDP | pytest tests/iqe; gpu_qualification.sh | original + new behavior cases, Gloo and NCCL resume | full stdout logs; REAL_GPU_INTERFACE.json | PASS; generic GPU probe scope explicitly limited |
| Real engineering qualification | review_real_qualification.py, real_resume_probe.py | real frozen features/GT/banks | REAL_WORKFLOW.json; three frozen audits; Scorer results | PASS diagnostic; not formal Base qualification |
| Explicit user navtest steering | python -m iqe.navtest; navtest_export.py | inference-only schema; population/identity/hash rejection | dedicated navtest artifact root and NAVTEST_REPORT | PASS:12146 real scores; no train/selector writeback |

Full main training, new full candidate banks, full Scorer training, deployment activation and push were not started. The user subsequently explicitly authorized Base checkpoint navtest; that evaluation is outside the original default prohibition, and is not counted as an incremental training bank.
