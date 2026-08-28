# Candidate Scoring Audit

## Gate B: PASS

- Traffic policy: `non_reactive`
- Scenes / candidates: 500 / 6000
- Successful candidates: 100.000%
- GT score mean (min–max): 0.9328180881222203 (0.0–1.0)
- Repeated-run max absolute difference: 0.0
- Batch-vs-single max absolute difference: 0.0
- Candidate order preserved: True
- Simulated state shape: 41 states × 11 fields at 0.1 s for every successful candidate: True
- Scenes with at least one differing factor: 491/500

The evaluator exposes no-at-fault collision, DAC, DDC, progress, TTC, comfort and aggregate score.  This v1/custom single-scene interface does not expose TLC, lane keeping, history comfort or extended comfort; those columns are null and accompanied by an availability map, not synthesized.

Progress comparability: the deployed scorer normalizes each candidate against the cached PDM baseline progress. It does not use the maximum progress of the submitted candidate set.

## Blockers

- None.
