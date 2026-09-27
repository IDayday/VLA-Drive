# P0 implementation gap audit

Audited source:227e781dd54f623b12b41bd7f2d2a7b302e0a014. Findings describe mechanisms, not proven causes of the old research results.

| Requirement | Authoritative old call site | Finding |
|---|---|---|
| Fixed65actors | JointTrajectoryPolicy.encode_current in joint_world/policy.py concatenates native-action mean plus all64 postQwen agent_features | No deployment slot selection |
| Existence is not eligibility | JointTrajectoryFlow.forward in joint_world/flow.py concatenates existence to currentxy | No active KV mask |
| Full actor communication | GraphBlock.forward uses actor MultiheadAttention with learned relative_bias only | All actors communicate directly; position bias does not enforce locality |
| Mask whole slots indiscriminately | actor_mask in flow.py; train_graph.py/train_corpus.py call it with xy.shape[1] | All65slots are eligible, including unmatched hypotheses |
| Valid future is only supervision | joint_targets in targets.py uses match_reference then target.future_valid_mask; training_loss_sums masks loss | Target validity does not constrain deployment input; preserve this separation while adding CURRENT-based graph selection |
| Graphego differs from actualplan | policy.rollout_condition passes graphhidden through graph_to_world+adapter, then original action_model.predict_action | Score actualDiT ego separately from graph ego |
| Matching can be poor | rehab.match_reference runs Hungarian over allslots using scaledbox/classcost, without maximumdistance rejection | Source code shows permissive assignments. Whether/how they damage learning requires measured association audit, not speculation |
| Conditional sampler only diagnostic | tools/joint_world/diagnose_context.py::impute | New formal sample_conditional needed with unknown-value isolation and stepwise clamping |
| Arbitrary step caps | tools/joint_world/train_corpus.py bounds epochs<=8,batch<=16 | New campaign runner must implement16/32passes and independent48GPUh accounting |

Implemented first-stage changes: current calibrated observation contract; raw existence/support/reliability/relevance fields; A/B/C/D assignment with explicit risk context; rooted direct+second-hop relationships; graph/padding-aware attention and planner memory; 50/25/25 qualified task masks; identity-derived source-slot noise; label-side filtered original-slot association; formal conditional sampling. Unit tests cover poison, direct/indirect communication, padding/permutation, mask fallback, source-track identities and context retention. Real graph audits, formal training and online public-origin regression are still pending.

User steering now forbids private foundation weights in formal runs and requires final Navtest (see USER_STEERING.md). Old current caches are allowed only in explicitly labelled historical engineering audits, not main algorithm comparisons. No world/private action checkpoint has been loaded for this campaign's training. PublicQwen file identity has been independently checked against the official repository.
