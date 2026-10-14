# Fixed Scorer runtime evidence, 2026-10-14 17:36 UTC

The scientific identity is `b96a9aebec4f3d38130c69a12364a1c64fdba80c8895775a0babfcc852ffc1a7` throughout. `596620e4fbdc50554b8952f9e040ba502fff83ca77caded56c3489cc73395a8b` is the SHA256 of its serialized resolved JSON file, including its serialization, and is not another scientific configuration. Source of the fixed training function remains unchanged during runtime routing.

| Cell | Actual state | Host/GPU | Training PID | New parameters | Initial state |
| --- | --- | --- | --- | --- | --- |
| S00 | 6000 COMPLETE | rl-zt2/4, exited | 769549 | 3,301,639 | e046acb770c265fb… |
| S01 | 145 real updates | vla-zt/7 | 2737547 | 3,301,639 | same as S00 |
| S10 | 896 real updates | rl-zt2/6 | 776999 | 4,898,311 | 7d484e0c72a87bf1… |
| S11 | 29 real updates | rl-zt2/4, after S00 | 781344 | 4,898,311 | same as S10 |

All four new heads start randomly, with no external Scorer weight. They share the same frozen K4 pool and scene order. Global batch remains128, with6000 total updates,100-update warmup, then constant learning rate1e-4. The rich cells use rich labels only in the first3000 updates. These live counters are timestamped observations, not claims of final completion.

The sole completion controller is rl-zt2 PID781267. It observes the separately supervised S01 worker2737511 by an atomic, identity-bound heartbeat; it cannot silently relaunch a stale/failed remote worker. S00 and S10 were not interrupted by this host reassignment. Unrelated anchor-cache jobs appeared on the formerly allocated GPUs5/7; they were preserved. S11 took S00's naturally released GPU4. Four new worker/adoption behavior tests passed. Receipts are in `runtime/scorer_route_20261014T1731Z/` under the campaign root.

## Rich bank completion

`banks/rich/scorer_fit/BANK.json` has complete coverage of all16,581 D_S scenes, no invalid predictions, and bank hash `f3d2a5284c59d91815002090228252bb101145c492ef2d651207d6420b6058a8`. All deployment trajectories are retained. New unique official requests are66,324 intermediate-endpoint trajectories plus60,710 construction/eligible optimization trajectories;33,162 duplicate endpoint requests were avoided. Candidate cap remains12, source order and geometry rules are unchanged, and no GT is a deployment candidate. Eight-thread assembly verified all committed records before bank completion.

## Storage-only spatial reader migration

The initial Lustre-mapped production segment was slower despite numerical equivalence:93 updates took1757.56seconds including its native invocation overhead. A single bounded RAM copy now holds the exact16,581 D_S spatial records:235,005,829,200bytes, or218.86GiB. All source/archive bytes are SHA256-verified; tensors are neither recomputed nor compressed. S10 and S11 map the same files. stage_val/cal/dev continue to read their original project records. At17:36 the host had191GiB available; the new spatial launch required128GiB reserve plus20GiB estimated private working memory. The shared RAM files cannot be removed while either process maps them.

The real128-scene qualification compared complete inputs, loss, gradients, Adam state, actual parameter update and RNG. All were equal; both updates produced model hash `f23931f683e12b53da57e53b50ae59fa60901a80c94f90be640080b10ded48f8`. This qualification uses separate disposable parameters and does not choose a scientific checkpoint. Its three fetch timings are diagnostic, not a long-run throughput claim.

S10's native parent is `scorers/S10/step_000220.pt`, SHA256 `21bf861267ce3922953fc47c584322af0bb63a8365e53c6af4cb4e0656940063`. The resumed221st update consumed exactly the recorded next128 IDs, with finite loss0.1266438395 and gradient norm0.1374414414. Optimizer, scheduler, training RNG and both sampler cursors were retained. Source and resume evidence: `runtime/spatial_RAM_migration_20261014T1714Z/RESTORE_VERIFICATION.json` and `qualifications/spatial_reader_ram_mmap_v1/REPORT.json`.

Recent100-update measured means at17:36 were S00 final window0.4363s, S01 early window1.0704s and S10 warmed window1.1608s. S11 had only29 initial updates averaging4.1061s while verifying its first records; that is not a steady estimate. No precision, model, exposure or loss change is attributed to this speed improvement. Full measured cost accounting retains all stopped and resumed segments.

## First complete Scorer result

S00 finished all6000 updates. Its predeclared stage_val best is5000: selected91.23774%, Base91.39146%, gain−0.15372 percentage points. The5000 interval for the paired gain is[−1.37573,+0.95593]percentage points across53 logs; it had12 high-to-zero switches. The6000 endpoint selected90.54250%. This is evidence that semantic deploy-only training has not yet converted Oracle coverage into reliable selection. It does not justify altering the other cells or using test results to tune this matrix. Final fixed-dev reporting waits for the four registered Scorers.

No Base/G2 task, old checkpoint,22k identity or ACTIVE was changed. No push was performed. The independently authorized Oracle Navtest is outside this training/selection campaign.
