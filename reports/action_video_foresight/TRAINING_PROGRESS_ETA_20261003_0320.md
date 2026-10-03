# First-seed formal training progress and estimate, 2026-10-03

Actual snapshot: 2026-10-03T03:20:47.450962+00:00. All five remain RUNNING from immutable1493ded, each8A800 GPUs, globalbatch32, common100000updates. CurrentC1 DINO and frozen vehicle MAE remain active in every arm. No training algorithm, source, checkpoint or data was changed for this status check.

| Arm | Updates | Progress | Scene exposure | Recent elapsed seconds/update | Remaining training hours |
|---|---:|---:|---:|---:|---:|
| S0 | 16558 | 16.56% | 529816 | 1.824 | 42.3 |
| S1 | 18716 | 18.72% | 598872 | 1.799 | 40.6 |
| S2 | 17945 | 17.95% | 574200 | 1.915 | 43.7 |
| S3 | 19816 | 19.82% | 634064 | 1.804 | 40.2 |
| S4 | 21144 | 21.14% | 676560 | 1.788 | 39.2 |

The estimate uses the last1000 completed optimizer records and their elapsed end timestamps, including routine saving and data waits. S4 needs about39hours; the slowest primary arm about44hours. Allow roughly44–48hours for all five first-seed100k runs with future milestone export/scoring and ordinary runtime variation. At this snapshot that means late2026-10-04 to early2026-10-05 UTC. This is an extrapolation, not a deadline guarantee. Second training seeds, capacity/MAE controls and locked final Navtest are excluded.

Complete original development results,1696scenes/16logs,zero failures: S1 5k79.0481/10k84.8536 PDMS; S4 5k79.5559/10k82.3019. These are early development results, not Navtest or final method rankings.

S0/S2/S3 exported every requested prediction on both5k and10k, but their original scoring jobs failed before CPU evaluation: remote hosts cannot resolve the canonical-host SSH alias. Training continues. Recovery now executes the unchanged1493ded score/ego code directly on the audited canonical CPU host with the same FP32 exports, original full-precision caches, official full traffic environment and reference runtime. The independent6659529 recovery worker uses at most2x16CPU workers and0GPUs. It also watches only the already registered future development milestones. No new Navtest observer, model inference or optimizer update is added.

Original failed states/logs are retained. Corrected results use versioned sidecars and separate score directories; summaries require the original-failure hash and the registered checkpoint/run/population identities. The first recovery attempt failed before scoring because the inherited checkpoint field arm means C1 visual configuration, while the formal registration arm means S0–S4. The corrected loader verifies the full signed run identity and registration byte hash rather than confusing these namespaces; all six real exports passed that check.7focused CPU tests passed. Neither attempt changes scientific inputs, metric formulas or thresholds.

Exact source/command/PID and allocation are in `development_local_recovery_launch_v2.json`. Original6SSH failures and the initial observer failure remain in the external artifact ledger. Training source1493ded stays clean and running.

By03:34UTC the six missing scores have actually completed, each1696scenes/16logs/0failures. Common10kPDMS: S0=83.573641,S1=84.853557,S2=84.904337,S3=85.068252,S4=82.301894. Common5k:78.856143/79.048115/79.546458/80.203551/79.555889 respectively. Strict canonical runtime identities match and the original failed states are preserved. See EARLY_DEVELOPMENT_5K_10K_20261003.json. These are early development results, not final rankings or Navtest. Latest live updates at03:34UTC: S0=16999,S1=19164,S2=18365,S3=20261,S4=21588; allRUNNING.
