# Integrated SafetyEP: results and measured trick bottlenecks

The only full integrated NAVTEST point is actual update250: **90.33735971** on all12146 scenes, invalid0. This is +1.54444981 vs the same IL70 parent, -0.29010488 vs Control2k (nominal scene clock; different Adam/PG/compute), and -1.11623785 vs the historical global best Control10k91.45359756. It precedes retention admission. No integrated actual2k score exists yet. Current reference remains fixed to actual10k, with full evaluations2k/5k/10k.

The audit exhaustively reads all eight rank journals and all sealed behavior files through actual625: **10048 unique TRAIN scenes /160768 candidates**. No random subset or four-inner double counting.

| Observed mechanism | Complete evidence | Interpretation |
|---|---|---|
| Extra G16 discovery | strict-good in first8:8406/10048=83.6584%; full16:9056/10048=90.1274%; +650scenes/+6.46895pp | Within the same G16 draws, not a separately trained G8 policy or final PDMS gain. |
| Historical gate |23/5952selected slots=.3864%; global nominal anchor coefficient .0572% |25% is conditional on the gate. Early low gate rate does not prove long-run failure. |
| No PG |385groups/3.8316%; all-zero4, of which1 still has SafetyEP cost signal and3 have no PG | All-zero canonical reward and zero structured advantage are different. This small early TRAIN tail is not the12146-scene NAVTEST zero population. |
| PPO blocking | Logged strict-good positive surrogate blocking mean1.006% | Does not currently justify prioritizing a larger clip range. Coefficients are not parameter-gradient norms. |
| Same-version discovery/deployment |6016TRAIN B1 probes;884scenes have a strict-good G16 candidate but non-strict B1, including39unsafe and12zero B1 | A concrete conversion opportunity. Measured before updating on the fresh scene; not proof these scenes stay unlearned. |
| Prepared rescue pairs |825unique TRAIN scenes with strict-good candidate minus nominal B1 score>=.02 | Offline diagnostic pool only; no NAVTEST feedback and no live training-data change. |

The existing bank only admits already strict-good nominal B1 witnesses; a sampled-good/nominal-bad scene is not a teacher admission. First priority is a separate safety-filtered self-imitation/flow-matching auxiliary test using such candidates. Start with5% of the fixed coefficient budget, keeping fresh SafetyEP/PG unchanged; calibrate actual gradients before treating coefficient allocations as comparable strength. This adapts the principle of [Self-Imitation Learning](https://arxiv.org/abs/1806.05635), not its actor-critic implementation verbatim.

Second, test a small5% historical floor plus20% regression increment as a separate ablation, or independently test risk-prioritized history selection within the current query budget. Neither is deployed; both can preserve a suboptimal old policy and require deployed high/zero-cohort checks.

Third, change only cosine endpoint2500->5000 in another fresh same-IL70 run. Keep warm125/keep625/peak LRs/floor.1/horizon12500. The original puts75% of actual10k on the floor; that may stabilize or undertrain and is not yet a diagnosed cause. The running reference is unchanged.

[ScaleRL](https://arxiv.org/html/2510.13786v1) motivates examining batch allocation and effective learning signal, while [DAPO](https://arxiv.org/html/2503.14476v1) motivates its dynamic-sampling alternative. These LLM results do not establish gains in this driving task. [DPPO](https://arxiv.org/abs/2409.00588) supports studying structured diffusion exploration and training/deployment noise differences, not blindly changing this flow policy's sampler.

Prepared proposals are explicitly **not registered or dispatched**. The current integrated reference will complete its user-committed10k budget. No optimizer, scheduler, GPU owner, source identity, routing, or model was changed by this diagnostic.
