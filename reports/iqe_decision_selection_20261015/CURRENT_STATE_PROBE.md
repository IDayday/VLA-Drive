# Privileged current-state diagnostic

| Policy | Scenes | Logs | PDMS /100 | Gain pp | 95% log CI pp | Regret pp | High→0 | New NC/DAC/TTC |
|---|---|---|---|---|---|---|---|---|
| P_current internal_fit | 2048 | 162 | 97.2269 | +3.1527 | [2.7209, 3.6055] | 0.3176 | 0 | 0/0/2 |
| M_scene matched internal_fit | 2048 | 162 | 96.9054 | +2.8312 | [2.4075, 3.2528] | 0.6391 | 0 | 0/0/3 |
| M_set matched internal_fit | 2048 | 162 | 95.1877 | +1.1135 | [0.6035, 1.5701] | 2.3569 | 9 | 0/9/21 |
| P_current internal_val | 512 | 40 | 88.4160 | -5.8557 | [-9.3936, -2.6451] | 9.4965 | 26 | 20/13/51 |
| M_scene matched internal_val | 512 | 40 | 92.0176 | -2.2540 | [-4.5118, -0.5028] | 5.8948 | 13 | 8/8/25 |
| M_set matched internal_val | 512 | 40 | 94.1894 | -0.0822 | [-1.8865, 1.6808] | 3.7230 | 9 | 3/6/11 |

Triggered only by internal validation; both main models had nonpositive gain. 2048 fit / 512 validation scenes were selected by fixed hash; 2000 updates, one random initialization. The final endpoint is fixed and cannot enter a deployment bundle. Matched M_scene/M_set comparators are their internal diagnostic endpoints, not full-D_S refits. Data quantity, parameterization and information differ.

Inputs: t0 timestamp-attested agents, provided current velocity, static map geometry and current ego, plus the fixed candidate trajectories. No reference future, GT, future velocity differences or simulation outcomes. The official occupancy t0 can contain future-only one-frame objects; the extractor explicitly rejects mismatched object timestamps. Across all 2560 actual records, 19489 future-only objects were rejected; deleting future/reference containers leaves all extracted inputs identical. Range is 50 m and current three-camera frusta are enforced; occlusion is not verified, maps/annotations remain privileged. Current state improving selection would not alone prove W/A discarded the same visible information. Failure would not prove planning selection is intrinsically impossible.

Actual source, normalization-fit counts, log counts and per-record hashes are in current_probe/PLAN, FIT_NORMALIZATION and immutable record receipts; no new official scoring or dev reads were used.
