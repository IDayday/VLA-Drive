# Six-Factor Independent Relabel Report

The historical five-factor schema remains preserved and is explicitly classified as
`INCOMPLETE_DDC_MISSING`. This run uses the immutable v2 order
`[NC, DAC, DDC, EP, TTC, Comfort]`; raw progress is diagnostic only.

| Gate | Scenes | Candidates/run | Run1 logical SHA256 | Run2 logical SHA256 | Max reconstruction error | Status |
| --- | ---: | ---: | --- | --- | ---: | --- |
| G0-R2a | 1 | 256 | 9bee714595771092e0852437af87338bd5426becc067b2ea4d1331d63bca43d1 | n/a | 1.9868215e-08 | SINGLE_SCENE_SIX_FACTOR_PASS |
| G0-R2b | 10 | 2560 | 182081108f89b57531af9fc0ed5727f25d56b44ee01fd6e83c9691ba42cf145d | 182081108f89b57531af9fc0ed5727f25d56b44ee01fd6e83c9691ba42cf145d | 3.97364298e-08 | TEN_SCENE_DETERMINISM_PASS |
| G0-R2c | 200 | 51200 | 803cd8b6d0e6b18d3f84ed45fe568803ce130bf0885f1ba14aa305e012d67883 | 803cd8b6d0e6b18d3f84ed45fe568803ce130bf0885f1ba14aa305e012d67883 | 3.97364298e-08 | SIX_FACTOR_RELABEL_PASS |

Every evaluator invocation received the complete 256-anchor set in one call, so EP
normalization is scoped to all 256 candidates within each scene. No candidate was
added, removed, chunked, or offset. The published-label comparison is a non-blocking
upstream reproduction audit only.
