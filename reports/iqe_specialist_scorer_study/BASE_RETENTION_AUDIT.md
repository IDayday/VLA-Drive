# Frozen Base retention

| Cell | Parameter/buffer hashes identical | Raw drift | Physical drift | Training VLA calls |
| --- | --- | --- | --- | --- |
| E00 | True | 0 | 0 | 0 |
| E01 | True | 0 | 0 | 0 |
| E10 | True | 0 | 0 | 0 |
| E11 | True | 0 | 0 | 0 |

No Base training process, checkpoint, 22k identity, or formal ACTIVE is modified. Complete shared weight hashes remain tied to the original immutable checkpoint; private training modules contain no shared VLA parameters.

Generated 2026-10-14T07:26:31.328995+00:00. Implementation `0e0ed8d2964c5215d7fcfa712e3f5ab1121ab24d`. Base step 70000, SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`. All current scores use NAVSIM v1.1 fixed official reference; internal score unit [0,1]. Artifact root: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
