# Formal Query base training

RUNNING on `training-vla-zt2-worker-0`, 8×NVIDIA A800 80GB. Main PID2124247; efficiency monitor PID2124248. Snapshot: optimizer step **32**, global batch32, microbatch4/rank. All original ego IL/current-DINO/future-clip/interaction objectives remain enabled; gradients/losses finite. Generic Qwen initialization only; no old learned driving checkpoint.

User later explicitly authorized formal training after efficiency qualification. This supersedes the initial default prohibition on automatic full training. Full candidate scoring, formal incremental experts/Scorer, navtest and deployment were not started automatically. The new Query base must finish and be locked before those formal stages. CPU two-round smoke already exercises their complete implementation separately.

Final code commit `334c0008ecd41d521a47cc3de68d890b1249af76`; full source/data/config/checkpoint dependencies are recorded in FORMAL_TRAINING_LAUNCH_V2.json. Formal budget100000 optimizer updates, lr1e-5/warmup5000/global32. Input3 current cameras, original1024×576, original Qwen processor/prompt/state/normalizer. No input, loss, candidate count or effective-batch reduction was used to improve throughput.

Final eight-step qualification (profile_ddp_v3, distinct short schedule/output): after3 warmup updates, p50=3.249s, p95=3.892s; threshold30s was set before runs. Full CLI time115.3s includes setup/checkpoint; trainer accounted647.32 GPU-seconds. Full training observed steady p50/p95=3.952/4.274s at monitor step26; this snapshot is not a completion claim. The monitor records every30s, deadline7 days, and makes no semantic or hyperparameter changes. Live records supersede this snapshot.

Full data manifest: fit90892/1056logs, stage_val5563/56, selector_cal5137/64, dev1696/16. Group/log/observation separation passed for103288 observations/1192 groups. Complete image/target/context hashing used16 ordered I/O workers (392.85s), preserving the serial manifest exactly on384 real comparison records. Training loader stays2 workers/rank. No final_test data.

Failures and corrections are retained: first eight-card attempt failed before updates because original reentrant language checkpointing conflicted with DDP. Non-reentrant checkpointing keeps recomputation/dropout RNG and passes value/gradient equivalence; real8-card updates then passed. First full launch PID2121632 was stopped at0 updates after detecting O(N²) repeated fit-ID-set construction. Precomputing the same set once retains all source validation; the corrected code completed another8-card qualification and was restarted with the current PID. No optimizer progress was discarded and no unrelated process was terminated.

Artifacts:

- Launch: `/mnt/project/iqe-runtime-audit-20261009/FORMAL_TRAINING_LAUNCH_V2.json`
- Active training pointer: `/mnt/project/iqe-runtime-audit-20261009/FORMAL_TRAINING_ACTIVE.json`
- Log: `/mnt/project/iqe-runtime-audit-20261009/formal-query-base-vla-zt2-v2.log`
- Live progress/efficiency: `outputs/iqe/main/query_base/status.json`, `EFFICIENCY.json`, `steps.jsonl`
- Final profile checkpoint: `/mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/profile_ddp_v3/query_base/step_000008.pt`
- Formal checkpoint interval1000 updates, approximately22GB each /2.2TB for100k. At this snapshot the first scheduled formal checkpoint has not yet been written; real existing smoke/profile checkpoint paths are in RUNBOOK. SIGTERM of this task at a later optimizer boundary writes a resumable checkpoint; do not kill unrelated jobs.

Scientific status UNTESTED. There is no formal Query-base quality result, no formal incremental gain result and no published deployment. E12 comparison remains BLOCKED until a runnable aligned old LoRA-DiT ensemble is bound; E3 new-observation contribution is UNTESTED because no such observations were supplied or found. IQE main closed-loop implementation is complete; aggregate implementation status remains PARTIAL to avoid treating the unbound E12 config as a working baseline.
