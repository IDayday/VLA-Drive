# Current execution evidence

Working Python is `/root/miniconda3/envs/ddp/bin/python` (3.10.20). Actual CPU command: `python -m pytest -q tests/iqe`, with that Python, produced **167 passed,2 skipped,4 warnings in37.09s**. Full output is `evidence/REGRESSION.txt`. The skips explicitly require a GPU opt-in, and were subsequently executed on the empty authorized worker:

```bash
IQE_RUN_GPU_TESTS=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/iqe/test_training_resume_ddp.py::test_nccl_uneven_labels_and_exact_resume \
  tests/iqe/test_closeout_terminal.py::test_nccl_terminal_resume_all_ranks
```

Result: **2 passed,1 warning in16.37s**, full output `evidence/NCCL_REGRESSION.txt`. The final environment-binding change also passed the12 focused performance tests (`evidence/ENVIRONMENT_BINDING_TEST.txt`). These are current results, not inherited prior-report counts. Initial development failures (wrong default Python3.9 environment and a corrected syntax error) remain in runtime logs; the commands above use the corrected environment/source. No dependency upgrade was performed.

Separate real Query tests on8A800 include64-scene online/cache equality; missing-label ranks and actual batch1 tails; each candidate's first-update numerical probe; two complete200-update combinations; and both combinations' complete-model/optimizer/RNG exact interrupted resume. The generic two-rank regression tests are not presented as substitutes for these actual eight-rank model tests. All throughput windows keep the original101592-scene ordered population, global32/micro4, four objectives and100k schedule. Alignment of actual IDs, LR, valid denominators, used parameters and end RNG across all three200-update runs is recorded in `evidence/LONG_RUN_ALIGNMENT.json`.

There were exactly **three200-update runs** (P0, FP32 combination, visual-BF16 combination);985 total real Query updates include short screens, traces, probes, cache and resume diagnostics. Only189 steady unprofiled updates per long run determine the reported long-window speed. One-/two-step resume/cache tests and seven-step traces do not contribute to throughput estimates. No full training, navtest or ACTIVE update was started.

Resource accounting in `CAMPAIGN_COST.json` separates:

- Dedicated8-GPU reservation,06:15:38→07:45:14UTC: **11.947GPU-hours**, including gaps/control work; below16GPU-hour cap.
-30 receipted bounded processes: **10.748 reservedGPU-hours**, including startup/state verification/save. One early trace lacks a process-exit receipt and is not silently imputed into this sum.
-Recorded actual training-loop allocation: **5.120GPU-hours**. This is elapsed loop×8, not GPU-utilization-weighted energy time. It overlaps the process/reservation figures and must not be added to them.

The final worker query shows8 idle GPUs,0MiB allocated and no compute processes (`evidence/WORKER_END.txt`). No background full run or automatic next-stage controller remains.

Normal-update GPU seconds per1024 sample exposures: P0 **696.36**, FP32 combination **630.41**, fastest combination **354.57**. Complete checkpoint saves measured49.55/49.42/44.56s respectively. At the retained1000-update cadence, estimated amortized update means are2.76972/2.51196/1.42958s. Projecting those measured rates over10000 updates gives7.69/6.98/3.97node-hours, excluding startup and future variability; these are projections, not completed10000-step jobs.

Performance artifacts live in `/mnt/project/iqe-perf-v2-runtime-3855e06`; raw traces, probes, visual tensors and checkpoint weights are not committed. Small metrics, checksums, CLI help and complete test output are committed. `CANDIDATE_RESULTS.csv` contains only eligible throughput windows. P5's two-step equality and randomized IO pilot are deliberately reported in `FROZEN_VISION_CACHE_BUDGET.md`, with no fabricated steady training rate.
