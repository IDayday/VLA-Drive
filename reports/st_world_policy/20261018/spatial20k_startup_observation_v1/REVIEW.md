# Read-only continuation startup check

Observation: 2026-10-18 02:47:57–02:48:08 UTC, two process-tree observations per host separated by ten seconds. Native status/receipt files were then read once. Reviewer `/root/spatial20k_efficiency` did not dispatch, signal, load/re-hash checkpoint payloads, alter source or issue another approval.

**Decision: keep both spatial continuations and the history metadata worker running. There is concrete progress, not evidence of a hang.**

| Run | Confirmed completed step | Exposure | Native step5000 restore receipts | State |
|---|---:|---:|---:|---|
| SP2 on Tsinghua zt6 | 5011 | 160344 | 4 ranks | RUNNING |
| SP3 on Tsinghua zt2 | 5007 | 160216 | 4 ranks | RUNNING |

Both `RESUME_RESOLUTION.json` records identify the original completed 5000 boundary, new 20000 endpoint and preserved optimizer/RNG/sampler. Latest observed current/ego-FM losses were finite. The initial period included original S0 model reconstruction, full parent validation and native model/ZeRO state loading. SP2 log explicitly completed its model and optimizer partition reads at 02:47; subsequently both jobs emitted restored-state receipts and advanced optimizer steps. Low early GPU memory/utilization should not be interpreted as stalled training while these initialization stages execute.

Over the 10.34-second observation, SP2's four main ranks accumulated 48.39 CPU seconds and 10.62 GB logical reads; SP3 accumulated 48.73 CPU seconds and 8.45 GB logical reads. These are process counters, not storage-wire throughput. SP2's sampled GPU utilization was 72–91%, with approximately 46–48 GiB device memory in use (nvidia-smi, not PyTorch allocated-memory accounting). SP3's instantaneous utilization sample was zero despite subsequently verified optimizer advancement; single samples cannot establish a hang. Numerous inherited helper descendants were idle (aggregate non-main CPU about 0.01 seconds per job), so this observation does not identify active helper oversubscription. No new performance diagnosis or change is warranted from this short startup window.

History attempt2 has a real running child, PID 786489, under reviewed controller PID 786421. It accumulated 3.18 CPU seconds, 14.33 MB logical reads and about 716 MiB RSS in 10.35 seconds. Intermittent Lustre waiting was visible. `CONTROL_HISTORY_STARTED.json` and `METADATA_ADMISSION.json` exist; a final COMPLETE receipt does not yet exist. This is actual CPU/IO progress, but the low sampled rate does not establish that the full population will finish within four hours. Preserve the current bounded monitor and its terminal/partial evidence; do not automatically repeat the attempt.

There were no model tests or third process sampling window. Raw process inventory is retained outside the repository; the committed summary contains resource aggregates and scientific progress only. No steady training-speed or ETA claim is derived from this window.
