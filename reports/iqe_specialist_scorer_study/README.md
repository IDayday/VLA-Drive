# Query specialist and internal Scorer study

Code starts at `fdc9a7bfbcd19287c9164e80ca2edcf6d9d0d69c` on isolated branch `research/iqe-specialist-scorer-transfer-study`. No active Base or deployment is modified. The Base is the existing complete 70k Query checkpoint; its immutable binding is in `evidence/BASE_BINDING.json`.

The full legal expert subset contains 74,311 scenes; the Scorer-only fit subset has 16,581 scenes. They remain log-disjoint. The 16,843 target-audit candidates are current Base failure/opportunity scenes from expert fit and stage validation, not a new 16k training cap. Stage validation has 1,024 fixed scenes, calibration 1,024 and development 1,696. No new navtest is run.

Historical subset results use only real official per-scene labels; no historical Scorer weight or prediction is used. Optimized target ancestry is retained and independently rescored. Only explicitly admitted GT/map/reference construction sources may enter the core target contrast; external and ambiguous sources remain audit-only. All Scorers initialize their new parameters randomly.

The code adds a thin research layer around the existing adapters, feature cache, official single-candidate scoring backend and optimizer-boundary trainer. The shared trainer only adds an optional post-update callback for private EMA and explicit validation milestones. The original Base forward/loss and protected model files are unchanged.

Current evidence: full regression at a938fb32: 502 passed / 3 skipped, 53.43s, with full output in `evidence/pytest_regression_precompute.txt`. Real four-scene private-reader strict clone, backward/update and Base retention passed. Full spatial coverage is complete for 20,325 fixed scenes, preserving original capture batch layout when regrouping would change FP32 numerics. Labels do not enter the feature whitelist. These are engineering checks, not proof of Scorer selection gains.

All six expert runs completed 5,000 updates each. The locked K4 pool uses E11/3000/EMA, interaction/3000/raw and safe_progress/1000/raw alongside the same Base. Fixed stage_val Base is 91.39146%, pool Oracle 96.52152%. These are validation candidate-coverage results, not test or Scorer results. As of 16:18 UTC, stage_val, calibration and dev deployment banks are complete; the last D_S official column and final integrity aggregation are running. S00/S10 start after D_S completes; S01/S11 start after the rich bank completes. No extra expert, seed or training budget is introduced. Runtime outputs: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
