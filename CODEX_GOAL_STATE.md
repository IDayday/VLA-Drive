# V3 pretraining correction — IN_PROGRESS

User-authorized bounded correction from b4f08d05c0202fb483bc3247cab97abe59694d16. New branch fix/joint-local-scene-v3-pretrain-review-20260927; worktree /mnt/project/VLA-Drive-v3-pretrain-review-20260927.

Only CPU tests, necessary GPU forward/backward, <=20 total synthetic optimizer updates and real-data no-update checks/rebuilds. Real optimizer updates MUST remain0. No ALL/MASK effect experiment, no full VLA training, no Navtest, no old V2 restart. End after correction push/review delivery.

Artifact root /mnt/project/v3-pretrain-review-artifacts/20260927. New review ledger caps2GPUh, synthetic20, real0. Existing V3 ledger and annotated_v1 are unchanged. Old V2step1887 sealed.

Implemented schema4 modeled_state_mask (ego xy+yaw, neighbors xy), safe padding/context before projections, strict active-input validation; decision_local/nearest builders, verified classes0/1/2 trajectory vs3–6context, FOV-only eligibility, navigated corridor and dedup/context overflow audits; cross-call RoleScheduler; fixed same-target query evaluation; strict config/trainer resume and independent budget.

Original9 counterexamples reproduced on b4f08d0 with0updates; reports/joint_local_scene_v3/pretrain_review/BEFORE.json.49CPU tests passed, no optimizer in unit suite. Eight real data samples verified source/track/future alignment before full new train7284/holdout64 schema4 rebuild; both complete0fail. Current-only source is prior ROI/FOV-filtered, missing raw population explicit.

Next: stage source commit, CPU and GPU synthetic4vs2+2 checks (16totalupdates if both), necessary GPU padding/state/gradient and4real no-update checks; finish audit report, push only fix branch, verify remoteSHA and stop. No long experiments.
