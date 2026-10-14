# Current-Base target opportunity

| Specialty | Failures | GT opportunity | Paired | Paired logs | GT U | Optimized U | Target gain ≥0.01 | Near-identical pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interaction | 404 | 373 | 116 | 90 | 0.00224534 | 0.000686654 | 8 | 25 |
| road_direction | 1732 | 319 | 125 | 89 | 0.00405811 | 0.00140272 | 39 | 38 |
| safe_progress | 14476 | 40 | 24 | 22 | 2.80443e-05 | 0.000608739 | 23 | 0 |

The raw opportunity audit considered 16,843 current failure/opportunity records from the full D_E plus stage_val, not a new fit cap. Targets were rescored against the current Base reference context. The core paired comparison uses original GT≥0.9, optimized target≥0.95, NC/DAC full, optimized target no worse than GT and at least 0.02 above Base, plus geometry/source audit. DDC has weight zero in this protocol and is separately diagnostic. 
Historical assets include mixed provenance. Only explicit GT/construction/PDM-reference/corridor/map sources enter the core; diffusiondrivev2, DrivoR and ambiguous proposal ancestors are audit-only. The original source labels are retained. This does not claim the entire historical asset library was independently generated. Current labels are official environment scores; no old teacher or external Scorer weights are loaded.

Pilot `road_direction` uses the same 125 observations/order in all four cells. The matched H_val has only 1 scene(s), insufficient for a reliable specialty-specific heldout claim. Full fixed stage_val remains the endpoint selection set.

Generated 2026-10-14T07:26:31.328995+00:00. Implementation `0e0ed8d2964c5215d7fcfa712e3f5ab1121ab24d`. Base step 70000, SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`. All current scores use NAVSIM v1.1 fixed official reference; internal score unit [0,1]. Artifact root: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
