# SCADV3 intermediate and latest-checkpoint Navtest

The finite scan is **running**, not complete. It preserves canonical NAVSIM v1.1,
single-candidate source-shift5 ODE K10, inference seed2101, and the complete
12,146-scene / 136-log Navtest population. Training is unchanged.

Previously completed, sealed full-population results:

| SCADV3 optimizer update | PDMS |
|---:|---:|
| 1,300 | 88.993109 |
| 12,900 | 90.285245 |
| 26,900 | **90.435625** |
| 32,000 | 90.098149 |
| 50,000 fixed endpoint | 90.324199 |

26.9k exceeds SCADV3's fixed50k endpoint by **0.111426 points**, with paired
log-bootstrap95% interval **[−0.086043, +0.329000]**. Its G+ contribution is
1.218550 points and G− is1.107124;75 nonzero→zero and96 zero→nonzero scenes.
NC new violations/recoveries are36/28, DAC42/72, TTC101/90. This supports a
higher observed intermediate mean, not an established general improvement.

Original CANONICAL50k is90.513164. SCADV3 26.9k is lower by0.077539 points,
with95% interval[−0.343744,+0.165040]. Compared with CANONICAL, its NC/DAC/TTC
new violations/recoveries are54/19,65/41,151/56. The intermediate peak does not
establish that SCADV3 solved safety degradation.

The locked new checkpoints are SCADV3 5/10/20/25/30/35/40/45k, original IL86k,
IL77k-parent canonical RL24.6k, and fresh three-frame/current-DINO IL14k.
An exact-source IL77k control is also included, giving12 full evaluations and
145,752 logical single-candidate inferences. Latest weights are pinned once;
this campaign does not continually chase newer checkpoints.

Four SCADV3 merged exports and their sealed native/merged parity proofs are
reused. Other RL exports require fresh parity before inference. DINO requires
native-predict parity before admitting its remaining shards. IL77-parent RL
uses its own permanent IL77 reference and evaluator source; it must not inherit
IL53 scores as a matched control. No source-file equality assertions are removed.

The user newly authorized vla-zt3 for this finite Navtest only. Admission uses
actual GPU UUIDs, visible ownership, headroom and loading reservations. Training
is protected; temporary admission closes after all owned workers exit. Other
expired temporary hosts remain excluded. At most two evaluators share an
initially idle GPU and one shares an active-training GPU if measured capacity
permits. Worker occupancy is not exclusive GPU compute time.

This is a user-requested **exploratory Test scan**. Any highest intermediate
checkpoint is identified as Test-selected; the fixed50k endpoint remains in the
report. Bootstrap intervals do not correct for selecting among checkpoints.
No new training seed, inference-seed matrix, oracle or best-of-N is introduced.

[registration.json](registration.json) fixes checkpoint/source identities and
budgets. [historical_results.json](historical_results.json) contains the five
full historical summaries. The two `historical_*.json` files contain paired
comparisons and safety transitions. [validation.md](validation.md) documents
the executed checks and completion/report commands.
