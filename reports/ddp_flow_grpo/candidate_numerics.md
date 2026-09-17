# Candidate chunk numerical investigation

The final BF16 action-only integration gate compares chunk sizes 1 and 2 on the same saved candidates, all ten transitions, and original three-view images. Tolerances remain `atol=2e-6`, `rtol=2e-3` for parameter gradients. Both released variants fail this gate. This is a **TESTED / FAIL**, not a PASS or a missing-data blocker.

Enabling the same PyTorch deterministic algorithms used in training fixed the separate checkpointing on/off gate for both variants. It did not fix candidate-chunk gradient differences. Original failed and corrected reports are retained.

A separate fixed-seed two-candidate probe (`candidate_gradient_probe.py`, `chunk_gradient_probe.json`) traced the difference through the real model. Maximum absolute forward mean/log-prob differences were 1.1920929e-7 / 4.7683716e-7; the condition-gradient relative L2 difference was 2.1142e-6. Some parameter gradients differed substantially more after BF16 backward/accumulation. This identifies numerical sensitivity in backward; it does not prove every difference is harmless or excuse the failed gate. This probe uses a two-candidate rollout, whereas the integration gate selects the first two candidates from its saved eight-candidate rollout; their statistics must not be conflated.

An experiment accumulated condition gradients through separate transition slots in FP64. Both full real-checkpoint diagnostics still failed the unchanged gate (`integration_action_*_reduction/`). The experimental patch is saved in `candidate_reduction_experiment.patch`; it was removed from the delivered runtime because it did not resolve the failure. Network precision, visual freezing, observations, rewards, and loss tolerances were not altered to obtain a pass.

The bounded training runs use the original, audited candidate chunk size 1. Their successful execution does not establish equivalence for another chunk size. Engineering readiness remains NOT_READY until the gradient gate is resolved and revalidated without weakening its acceptance criterion.
