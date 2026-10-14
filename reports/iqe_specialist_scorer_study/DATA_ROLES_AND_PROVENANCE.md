# Fixed roles and provenance

| Role | Scenes | Source groups | Base seen yes/no/unknown |
| --- | --- | --- | --- |
| dev_report | 1696 | 16 | {'no': 1696, 'unknown': 0, 'yes': 0} |
| expert_fit | 74311 | 845 | {'no': 0, 'unknown': 0, 'yes': 74311} |
| scorer_fit | 16581 | 211 | {'no': 0, 'unknown': 0, 'yes': 16581} |
| selector_cal | 1024 | 62 | {'no': 1024, 'unknown': 0, 'yes': 0} |
| stage_val | 1024 | 53 | {'no': 1024, 'unknown': 0, 'yes': 0} |

D_E and D_S use disjoint complete source logs. The entire legal expert pool was considered; there is no 16k expert limit. D_S is held out from the new experts, not from the already trained Base. The checkpoint-provided fit_scene_ids supplied Base exposure membership. Network optimization uses D_E for experts and D_S for Scorers. stage_val chooses endpoints; selector_cal calibrates only the optional margin. The fixed dev role has 16 independent groups, so bootstrap intervals do not establish broad statistical reliability. Historical completed Navtest banks are read only for descriptive analysis; current campaign training/selection never accesses Navtest/final_test.

Generated 2026-10-14T07:26:31.328995+00:00. Implementation `0e0ed8d2964c5215d7fcfa712e3f5ab1121ab24d`. Base step 70000, SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`. All current scores use NAVSIM v1.1 fixed official reference; internal score unit [0,1]. Artifact root: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
