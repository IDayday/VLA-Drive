# EpisodeDrive Model Candidate Audit

- Frozen checkpoint: `7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d`
- Exported/scored scenes: 2,378/2,378 (100.000%)
- Logs: 1,192; raw proposals / retained candidates: 64 / 16
- Raw candidates scored offline: 152,192
- Selected candidates scored/targeted: 38,048 / 2,378 scenes
- Mean raw-64 to selected-16 oracle regret: 0.0029111468449878916
- Mean close-geometry/different-outcome candidates retained: 1.8124474348191757
- Mean baseline-selected / random / selected-16 oracle score: 0.9626258155901473 / 0.8269500682584393 / 0.9842051938039421
- Selected-16 oracle headroom (mean baseline Top-1 regret): 0.02157937821379493
- Ground truth forcibly inserted: no

K=16 combines original-scorer-high proposals, farthest-point trajectory
diversity and close-geometry proposals with different offline outcomes. Candidate
order is deterministically shuffled. Official metrics are used only to build and
evaluate the offline bank; they are absent from deployable model inputs.

The existing `configs/base_model_navtest.yaml` file is used only to instantiate
the frozen EpisodeDrive architecture/checkpoint. Every exported sample is
selected from the legal trainval manifest and `feature_cache_navtrain_full`;
no navtest/navhard scene or reactive-response cache is opened.
