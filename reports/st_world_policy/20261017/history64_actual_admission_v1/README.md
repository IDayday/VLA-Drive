# History64 packed metadata cost pilot — launched, result pending

This is one original 64-scene/64-log CPU-only cost pilot, not a formal model run or a new teacher tensor cache. The original frozen population and time-zero target schema are unchanged. Runtime b847f5a99dd44ee5256362d380d9d96efb47d35b; one invocation; CPU6/7; 900 seconds plus 40 seconds cleanup; 16 GiB aggregate logical-read cap; 4 GiB RSS; 31 MiB artifact plus a separately adopted 1 MiB control tree. The actual independent admission and its review are in `../history_packed64_admission_precheck_v1/`.

`LIVE_QUIESCENCE_EVIDENCE.json`, `QUIESCENCE.json`, and `MIGRATION_RECEIPT.json` bind the additive same-ledger transition. All old claims and failed output remain charged and preserved. The former TRAIN metadata attempt was controlled-stopped because its finalization reads could not fit its registered budget. This does not mean its metadata is complete. No training received a signal.

The 8 GiB metadata envelope, 768 GiB campaign cache cap, 1 TiB free floor, and separate 2303 GiB checkpoint commitments are unchanged. This pilot adds no GPU work and no teacher tensor output. `LAUNCH.json` is the actual exec request, not a completion receipt. No automatic retry.

Actual artifacts: `/mnt/project/st-world-policy-20261016/audits/history_data/causal_history_cost64_v2`. Terminal accounting: `/mnt/project/st-world-policy-20261016/audits/history_data/causal_history_cost64_v2_controls/RESULT.json`. Read the final receipts before treating this as a completed cost measurement.
