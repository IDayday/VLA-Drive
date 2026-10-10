# Workflow, expansion, pool and serving identities

Separate records now distinguish workflow round, expansion attempt, accepted pool hash, selector update and previous serving bundle. No-gain history iterates actual registered expansion attempts. Scorer-only rounds do not count, reduce the expert budget or invalidate an accepted pool. `scientific_expansion` and `consecutive_no_gain_expansion_limit` are consumed explicitly, including standalone build-round/train-expert. Accepted-pool membership is candidate_only/served; frozen is permitted only in explicit pending evaluation, rejected never enters the next old pool. Expert IDs remain append-only.

Real diagnostic sequence:

| Workflow | Action | Pool after workflow | Updates |
|---|---|---|---|
| 1 | add expert_1; train Scorer | 0,1 | expert2; Scorer2 |
| 2 | Scorer only, replay same pool | 0,1 | Scorer2 |
| 3 | add expert_3; replay all candidates; warm-start Scorer | 0,1,3 | expert2; Scorer2 |

All completed, then all stage receipts were verified and reused. [Workflow evidence](evidence/REAL_WORKFLOW.json), [reuse](evidence/REAL_STAGE_RESUME.json). Third workflow is expansion attempt2, not3. The full-mode unit fixture additionally proves no false NO_COVERAGE_GAIN after a selector-only round, and verifies genuine consecutive rejected expansions do stop at the configured limit.

The real third bank contains132 actual labels:88 reused immutable old labels +44 newly officially scored expert3 trajectories. Round1 reused88 valid historical labels because the actual generated trajectory/checkpoint identities matched; additional fresh direct scores were compared exactly with the bank. Round2 did not invent an expert2. Every selector identity lists its full replay pool.

Calibration on four natural selector_cal scenes found **no beneficial switch**, chose always-Base, and reported CANDIDATE_GAIN_NOT_REALIZED. No formal bundle/ACTIVE was changed. Temporary smoke bundles test save/load and previous-bundle comparison only. Their existence is not a release gate pass or a scientific improvement claim. Old raw/physical/ADE/FDE/yaw drift in round3 was0 for expert0 and expert1; parameters/buffers and corresponding checked PDMS were unchanged.
