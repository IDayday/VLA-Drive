# Gate C0 Reproduction Report

## Gate C0: PASS

- Base commit required: `6e96cf7321b134c42c2cf0fbbc315cd61c925b11`
- Sampled scenes/logs/candidates: 32 / 6 / 384
- Candidate trajectory max absolute error: 0
- Official score/simulation maximum absolute error: 0
- Structured target maximum absolute error: 0
- Candidate order preserved: True
- Official score present in model inputs: false
- Input hashes: `input_hashes.json`

The check regenerates prior deterministic candidates, reruns the deployed
PDMSimulator/PDMScorer and rebuilds every structured target for the sampled
scenes. Existing cache files are read-only and are never overwritten.
