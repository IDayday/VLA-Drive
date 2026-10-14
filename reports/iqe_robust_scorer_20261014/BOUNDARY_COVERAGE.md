# Existing Scorer supervision near trajectory boundaries

Read-only D_S audit. No validation, dev, test, new trajectory, optimizer or threshold selection.
XY ADE is geometric proximity, not a simulator clearance metric; close candidates can still differ dynamically.

| Bank | ADE limit (m) | Safety-crossing pairs | Scenes | Logs | Large reward-gap pairs |
|---|---:|---:|---:|---:|---:|
| deploy | 0.1 | 3 | 3 | 3 | 3 |
| deploy | 0.25 | 148 | 136 | 65 | 147 |
| deploy | 0.5 | 1247 | 953 | 146 | 1240 |
| deploy | 1.0 | 5456 | 2696 | 166 | 5433 |
| deploy | 2.0 | 9976 | 3419 | 170 | 9933 |
| rich | 0.1 | 1457 | 1141 | 139 | 1441 |
| rich | 0.25 | 5593 | 2417 | 171 | 5540 |
| rich | 0.5 | 15758 | 3227 | 178 | 15615 |
| rich | 1.0 | 50833 | 4588 | 184 | 50488 |
| rich | 2.0 | 90317 | 5050 | 188 | 89822 |

Neither abundant total negatives nor a small ADE crossing count proves the cause of generalization failure.
These counts describe supervision coverage and cannot be substituted for trained heldout performance.
