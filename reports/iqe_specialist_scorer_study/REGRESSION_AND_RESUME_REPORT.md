# Regression and actual integration

Repository regression:465 passed,3 skipped,4 warnings,54.30s (pytest_regression_v4.txt), including original-batch replay and unchanged-GT exclusion. These are distinct evidence runs, not an invented aggregate suite count.

| Real check | Status | Scope |
| --- | --- | --- |
| GPU private-reader exact resume | PASS | 32 real scenes,4 continuous vs2+2,raw/EMA/Adam/scheduler/RNG/cursor/loss/next IDs; terminal adds0 |
| Fresh official reference invariance | PASS | two real scenes,Base/GT audit,reorder/delete/duplicate/chunk,exact bank agreement |
| Independent-process research bundle | WAITING_INPUTS | real images,strict loading,finite outputs and stable selected candidate |

No training fixture is presented as real integration. The CPU/Gloo distributed loss test is separate from the actual single-GPU training path. No current Base/optimizer/data cursor or other experiment was restarted for these checks.

Generated 2026-10-14T07:26:31.328995+00:00. Implementation `0e0ed8d2964c5215d7fcfa712e3f5ab1121ab24d`. Base step 70000, SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`. All current scores use NAVSIM v1.1 fixed official reference; internal score unit [0,1]. Artifact root: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
