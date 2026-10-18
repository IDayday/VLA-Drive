# Feature research paused; consequence research continues

User instruction: “暂停关于特征层面的开发，集中开发有关后果的研究”.

Recorded and verified on 2026-10-18 UTC. This is a partial campaign pause,
not a request to stop the consequence branch or unrelated Base/SimWAM work.

## Executed pause

- **SP3: PAUSED at optimizer update 12,071.** The original trainer consumed
  its current update, honored its native `STOP_REQUESTED`, and saved native
  model, four optimizer partitions, four RNG files and sampler/scheduler
  state. `RESULT.json`, `status.json` and checkpoint `COMPLETE.json` all record
  `OPERATOR_PAUSED`. No training-process signal was needed.
- Checkpoint:
  `/mnt/project/st-world-policy-20261016/formal/spatial/runs/SP3_fulltrain_seed42_official_20k_v1/checkpoints/step_012071_1792335350235289141/`.
  All nine payload files exist and match receipt sizes: 37,371,024,930 bytes.
  The writer's checksums are preserved. This administrative check did not
  independently rehash 37 GB or load/resume the model.
- At **15:05:22 UTC**, read-only inspection on the recorded Tsinghua zt2 host
  found no remaining process matching this exact SP3 registration.
- **History preprocessing attempt 3: operator-cancelled.** After verifying
  PID/start-time/command identity, the lead sent one TERM through pidfd to
  the original supervisor. Its existing cleanup mechanism exited its owned
  child. The terminal receipt reports zero owned processes remaining; the
  known supervisor/wrapper/child PIDs are absent. Committed shards and partial
  evidence are preserved. The history population is not declared complete.
- The supervisor's original `FAILED_NO_AUTO_RETRY` / `SUPERVISOR_SIGNAL_15`
  receipts are retained unchanged. The accompanying classification records
  intentional user cancellation, not an algorithm failure. Final child
  accounting was unavailable after cancellation and is not fabricated.
- **T4 development: PAUSED_INCOMPLETE**, preserved in separate worktree
  `/mnt/project/VLA-Drive-st-temporal-t4-profile-20261018`, commit
  `4ea9e47c54ecbd69f53c899fb529e28ed7870655`. It was not merged or launched.
  T0 timing was never launched; SP2 remains previously paused.

The shared dispatch policy and campaign AGENTS instructions prohibit new
feature development, preparation, profiles, training and automatic resume.
No checkpoints, shards or old reports were deleted. No unrelated Base or
SimWAM process was changed. Resuming the feature scope requires a new explicit
user instruction.

## Current consequence work

The five original DiT/Flow C0–C4 arms have completed 4096 updates each and
fixed DEV evaluation on 1696 scenes / 16 logs. See
[complete results](../consequence4096_complete_results_v1/RESULTS.md).
There is no established practical held-out gain: C1 and C3 have tiny positive
differences whose log-bootstrap intervals include zero; C2 and C4 are slightly
negative. This is DEV, not Navtest.

Consequence work now focuses on explaining those results: sparse group-relative
feedback, the actual policy-versus-BC gradient contribution, and the direction
of a differentiable road-margin surrogate on existing real trajectories.
The road surrogate implementation is undergoing independent correctness and
systems review. A CPU fixture pass does not establish real directional benefit,
and a command-space surrogate does not equal the official simulator's gradient.
No new formal training, reward retuning from DEV, or feature experiment is
authorized by this status report.

## Evidence

Immutable control receipts:
`/mnt/project/st-world-policy-20261016/audits/operator_pause/features_20261018/`.

- `REQUEST.json`: original scope and pause request; not overwritten.
- `POLICY_UPDATE.json`: policy identity before/after adding the scope hold.
- `HISTORY_SIGNAL_REQUEST.json`: verified supervisor stop request.
- `TERMINAL.json`: terminal states, artifact hashes, nine payload size checks,
  and fresh live-process check. SHA256:
  `21c773a1677fc054ffab235f32383dd5a55fbae2f3138b1be4dcd7a06fe7f5e4`.

No automatic push, restart, model evaluation, or full payload read was performed
for this pause verification.
