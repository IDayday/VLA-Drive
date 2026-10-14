# Query specialist and internal Scorer study

Code starts at `fdc9a7bfbcd19287c9164e80ca2edcf6d9d0d69c` on isolated branch `research/iqe-specialist-scorer-transfer-study`. No active Base or deployment is modified. The Base is the existing complete 70k Query checkpoint; its immutable binding is in `evidence/BASE_BINDING.json`.

The full legal expert subset contains 74,311 scenes; the Scorer-only fit subset has 16,581 scenes. They remain log-disjoint. The 16,843 target-audit candidates are current Base failure/opportunity scenes from expert fit and stage validation, not a new 16k training cap. Stage validation has 1,024 fixed scenes, calibration 1,024 and development 1,696. No new navtest is run.

Historical subset results use only real official per-scene labels; no historical Scorer weight or prediction is used. Optimized target ancestry is retained and independently rescored. Only explicitly admitted GT/map/reference construction sources may enter the core target contrast; external and ambiguous sources remain audit-only. All Scorers initialize their new parameters randomly.

The code adds a thin research layer around the existing adapters, feature cache, official single-candidate scoring backend and optimizer-boundary trainer. The shared trainer only adds an optional post-update callback for private EMA and explicit validation milestones. The original Base forward/loss and protected model files are unchanged.

Current evidence: initial full regression 449 passed / 3 skipped; focused current tests 17 passed. Real four-scene private-reader strict clone, backward/update and Base retention passed. Real 256-scene spatial extraction reused the same visual/language forward, with W/A differences exactly zero and raw trajectory maximum difference 3.5763e-7 from batch regrouping. Labels do not enter the feature whitelist. These are engineering checks, not heldout scientific gain.

Full target scoring and the bounded expert matrix are in progress. Final training/selection conclusions must come from current completed artifacts, not this preliminary report. Runtime outputs: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
