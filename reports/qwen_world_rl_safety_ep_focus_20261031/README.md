# SafetyEP focused process review — 2026-10-31

The user narrowed current RL work to PPO_SAFETY_EP. Four active native LR/data branches exited cooperatively at committed optimizer boundaries; all checkpoints, Adam state, RNG, scene cursors and pending rollouts were retained. Their closure is an administrative focus change, not evidence that their algorithm families fail. Previously closed branches remain closed. Independent IL work is outside this stop scope.

Current live report: `/mnt/project/qwen-world-rl-optimization-20261030/report.html`, updated every 30 minutes and at important milestones. Files in this package are historical snapshots; archived owner files must never be executed as launch or resume instructions.

## Findings

The process diagnosis covers ten complete single-trajectory NAVTEST evaluations through 24k updates. SafetyEP scored 91.349892 at 10k and 91.146974 at 24k. The paired 44-log bootstrap interval for the difference crosses zero. Between those checkpoints, 51 zero-score scenes were repaired while 65 new zero-score scenes appeared; 44 of the latter came from the fixed 10k high-score cohort. Old-zero gains contributed +0.347103 to the overall PDMS difference, while the high-score cohort contributed -0.558873. Overall TTC improved and EP declined; simply reducing progress pressure is not supported as the primary solution.

The 10k checkpoint is an observed peak, and its high-score cohort is conditional on historical predictions. This cohort is a descriptive diagnostic, not an independent test proving forgetting or anchor interference. Future comparisons should predefine the IL70 high-score cohort and retain the historical 10k cohort only as a secondary diagnostic. Bootstrap uncertainty is conditional on the fixed training seed; no training seed replication was added.

In 1280 bounded TRAIN rollout groups, 111 had zero policy-gradient advantage: 110 were successful EP-saturated groups and only one was an all-failure cost tie. Mixed groups never assigned positive advantage to unsafe candidates. The selected windows do not estimate failure frequency across the entire 103k population. Scalar losses and total gradient norms do not establish the direction or relative magnitude of PG and the beta-50 physical anchor.

The full 103288-scene stream is shuffled without replacement and exposes two fresh scenes per optimizer update. Even 50k updates do not complete an epoch. A replay loss activated only by a fresh token matching an earlier buffer entry would therefore remain mostly inactive; meaningful success retention requires explicitly sampled auxiliary historical scenes.

## Research priorities

1. Measure PG versus physical-anchor gradient norms and cosine by action/Memory scope, with actual eight-rank reduction. Keep the live SafetyEP source unchanged.
2. Retain verified successful trajectories discovered by the new run, using a bounded independent auxiliary replay minibatch and freshly rebuilt current Memory. This borrows the success-reuse motivation of SIL/AWR; it is not an implementation of either original algorithm.
3. Compare a per-Monte-Carlo CFM-ratio ordinary PPO update with the current transition update, initially preserving W3 rollouts, SafetyEP advantages, reference, data and LR. The ratio is a surrogate, not an exact endpoint likelihood. Do not bundle a sampler change.
4. Audit the rare indistinguishable failure groups and current-observation sufficiency before changing failure credit or adding condition supervision. Neither missing information nor absent success witnesses can be repaired by inventing advantages.

All proposed comparisons start from the same pinned IL70 checkpoint, fresh Adam, training seed 42, optimizer update zero and scene cursor zero. They do not initialize from RL10k or reuse its optimizer state. No new variant was registered or launched as part of this analysis package. The detailed proposal separates conditional follow-ups from experiments ready to run; unknown coefficients and gradient scales remain explicitly unresolved.

## Evidence

- `SAFETY_EP_FOCUSED_RESEARCH_20261031.json`: current focus and initialization policy.
- `owners/SAFETY_EP_FOCUS/STOPPED_SUMMARY.json`: four actual cooperative exits, no native GPU process signals.
- `analysis/SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json`: paired scores, component/cohort accounting, rollout credit, sampling limits and source binding.
- `analysis/SAFETY_EP_FOCUSED_DIRECTIONS_20261031.json`: prioritized hypotheses, primary papers, exact official code revisions, controls and limitations.
- `analysis/SAFETY_EP_CREDIT_COUNTERFACTUAL_20261031.json`: CPU-only alternative-credit calculation, not a training result.
- `analysis/diagnose_safety_ep_process_20261031.py` and `analysis/build_safety_ep_focused_directions_20261031.py`: analysis builders; consume live evidence paths and are not training entrypoints.
- `SNAPSHOT_MANIFEST.json`: exact copied source/evidence hashes.

Primary references include [Self-Imitation Learning](https://arxiv.org/abs/1806.05635), [Flow Matching Policy Gradients](https://arxiv.org/abs/2507.21053), [Flow Policy Gradients for Robot Control](https://arxiv.org/abs/2602.02481), and [Sometimes You Gotta Run Before You Can Walk](https://arxiv.org/abs/2609.25831). Transfer assumptions and differences from this driving task are recorded in the detailed proposal.
