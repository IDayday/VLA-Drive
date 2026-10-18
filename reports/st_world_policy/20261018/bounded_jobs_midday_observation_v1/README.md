# Midday bounded-job observation

The observation was taken on 2026-10-18 after the earlier eight-hour SP invocations had elapsed. Both SP runs report `PAUSED/BUDGET_PAUSED`, with complete checkpoints at SP2 update 11,762 and SP3 update 11,582. This is a planned invocation boundary, not a training failure. Their last recorded gradients and losses are finite.

C1 and C4 reached their 4,096-update terminal states. C2 and C3 each completed all 1,696 fixed DEV exports. Official score jobs still require fresh execution authorization; expired score routes are not usable launch evidence.

The T0 metadata preparation completed. The history TRAIN metadata attempt failed during result publication with a storage-budget exception. Its conservative exception charge is not a measurement of actual bytes read. The existing output should be inspected before considering another builder attempt.

Lead released only positively identified owned pressure workers. Fresh checks then found Tsinghua-zt2, zt4 and zt6 GPU 0–3 empty and their native GPU locks available. These are instantaneous observations, not reservations. Other actual SimWAM work was protected, including local GPUs occupied after a temporary free interval. No process was signalled by this reviewer.

`OBSERVATION.json` preserves the small relevant status and resource evidence. Raw host process inventories are deliberately excluded.
