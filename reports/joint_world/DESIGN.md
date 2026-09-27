# Masked joint trajectory world model — implementation and evidence plan

User goal: jointly model ego and surrounding trajectories using actor-wise random masking, learn traffic interaction, investigate useful BEV objectives, and transfer the representation to planning. Fixed current CAM_L0/CAM_F0/CAM_R0 images. No RL, scorer, PDMS supervision, reward-labelled representation training, or future observation in deployment.

Baseline: source96ff2ee (V1.1), released Qwen3-VL2B + original Flow-Matching DiT, checkpoint9445f9d…; base loading remains strict. The old structured-world experiment is evidence, not a completed solution: final64 PRE49.47%/22.73% and POST49.95%/32.28% geometry recall/precision. Detection errors must be reported with downstream trajectory coverage. Planned old box4 repair was NOT launched before the user changed the objective.

## Algorithm

1. Current three-view visual features → image-conditioned fixed scene/agent queries → append-tail frozen Qwen. Current instance heads assign losses by current matching; GT never chooses model slots or initializes queries. Preserve native action conditions at initialization.
2. Form a fixed-size graph with ego slot0 plus K predicted agent slots. All future trajectories use ego(t0) metres; XY channels are separate from original ego normalized action/yaw channels. No GT track count/validity determines deployment computation.
3. A compact conditional flow model alternates time attention, actor attention with relative-position bias, and current-scene cross attention. Agent ordering is equivariant. The ego is distinguished only by role. Actor-wise random masking hides complete trajectories; masks are sampled without consulting labels. Missing labels enter loss masking only. At least50% of scene examples hide ALL futures, matching deployment.
4. Partial-mask GT context is a TRAINING-ONLY imputation task, explicitly privileged. This is conditional observational modelling, not proof of causal responses to an intervention. Train/evaluate an all-masked control and a context-shuffled diagnostic; report all-masked inference separately. At deployment, jointly generate ego/other trajectories from noise, with no known-future context.
5. Bridge generated graph features to original DiT through a zero-gated residual attention adapter. The planner loss must use a deployment-style all-masked rollout, never the GT-conditioned imputation hidden states. This separates lawful auxiliary teacher context from the deployed planner. Backpropagate ego imitation into graph/bridge; initially freeze original Qwen/vision/DiT. Preserve independent original-reference output and measured gate0 fidelity.
6. Compare original baseline, matched graph capacity without masked-context auxiliary training, masked graph, and a matched graph+BEV variant only after the image path is usable. All use one sample, the same ego noise,10FM steps, same split and frozen model-selection protocol. No trajectory scoring/selection.

## BEV objectives and transfer

Reuse the audited pretrained visual backbone + calibrated current-camera BEV construction as optional input (not a pretrained BEV claim). Supervise spatial support-aware current occupancy, future occupancy/flow and interaction geometry (pair proximity and relative motion), rather than RGB reconstruction. Future/map labels are supervision only. Forecast occupied space and interaction features must be consumed by graph attention and the planner bridge. Missing annotation/unsupported cells must not become free-space negatives. Auxiliary metrics alone cannot establish planning value.

## Evidence and finite budget

Use the remaining prior envelope conservatively: prior campaign2596 updates/1.671009907 GPU-hours; new updates plus prior <=24000, GPU-hours plus prior <=48. Start64-scene CPU unit checks, GPU frozen-baseline integration/label-poison/gate0/save-load, then bounded true-data learning. No new unlimited hyperparameter search; at most2 documented repair hypotheses. Train-log holdout before development planning evaluation; navtest never used for selection. Keep original1696 development protocol for paired planning, failure rows and component CSV. PDMS is evaluation-only and must never enter training files or model inputs.

Required conclusions: engineering status, interaction-learning evidence, planning result vs original baseline AND matched control, BEV evidence/cost. All conditional-GT metrics must be labelled as privileged diagnostics. Joint graph likelihood or improved reconstruction alone is not interaction understanding. Report fixed intermediate/final checkpoints, scene/log paired uncertainty and training-vs-sampling seeds.

## Reference provenance (ideas only, no copied implementation)

- WCog-VLA, https://arxiv.org/abs/2607.08375, v1: agent-token structured supervision and joint trajectory generation motivate interfaces. We do not use its RL, reward or Game-CoT pipeline, or claim reproduction.
- Scene Transformer, https://arxiv.org/abs/2106.08417: joint scene prediction and trajectory masking are prior art. The proposed contribution must be demonstrated in image-conditioned VLA representation-to-planning transfer, not claimed as inventing masking.
- MotionLM, https://waymo.com/research/motionlm/: context for joint conditional motion modelling; our implementation is continuous flow matching rather than token language modelling.
