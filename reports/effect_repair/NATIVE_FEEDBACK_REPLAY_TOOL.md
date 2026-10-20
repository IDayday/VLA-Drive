# Bounded native-feedback replay audit entry

Implemented `python -m tools.action_effect.audit_navsim_feedback`. This is a CPU-only engineering audit, not new model training or a scientific repair sample generator. It leaves official NAVSIM simulation/scoring unchanged and accepts only already-complete TRAIN G0 populations.

The selected population is fixed to the first at most four registered scenes, without inspecting their scores to pick favorable examples. For each scene it executes two separate native calls using precisely the same original baseline request. The request's native `repetition` remains zero because the locked v1 protocol does not support stochastic repetition as a scoring setting. The two audit slots have independent call-ledger admissions and result receipts; they cannot collapse into one content-cache hit. Re-running with `--resume` reuses only this audit's own completed slots.

Limits: four scenes, exactly two slots per scene, at most sixteen total durably charged reservations including explicit retries, one CPU worker and at most 1800 seconds. Planned normal work is eight calls. `CallLedger` reserves before each call; a partial admission cannot be counted as free. Existing native evidence is never overwritten. No GPU, optimizer, VLM or image decoder is instantiated.

Every slot stores the complete lossless canonical record, its independent checksum receipt and the validated feedback wrapper. Comparisons include:

- Same full request and fixed reference-input identity.
- Official PDMS and all named components.
- Complete native events and official first-event times.
- Submitted trajectory and time grid.
- Candidate executed states/body corners and reference interpolated/executed states.
- Native metric arrays and raw/effective progress decomposition.

Cost/PID/wall time are deliberately excluded from equality. Exact equality is declared before execution; no tolerance is adjusted after observing results. A mismatch in request, reference input, submitted trajectory, terminal metric or native event is `FAIL`. When those critical fields remain exact but native state/decomposition floating values differ, the result is `INCONCLUSIVE`, retaining exact maximum array differences. Only complete exact equality is `PASS`. Event timestamps remain observations, not causal credit.

Transient failures use an independently charged retry within the declared budget. Invalid predictions remain invalid evidence, never a successful zero. Unresolved slots yield `PARTIAL` with no completion receipt. A completed audit may truthfully have `FAIL` or `INCONCLUSIVE`; its receipt indicates all requested comparisons were executed, not that repeatability passed. CLI returns zero only for `PASS`, two for partial/negative/inconclusive comparisons, and one for preflight/program errors.

## Current evidence

- `--help` executed successfully.
- Eleven CPU tests: **11 passed, 1 warning in 3.53s**; the warning is the environment's deprecated `pynvml`.
- Tests are explicitly synthetic: exact comparisons including valid zero scores, event/metric/reference changes, one-ULP state changes, two actual fake-backend calls across distinct slots, zero-call resume, charged transient retry, exhausted retry, invalid prediction, deleted completed feedback and budget rejection. They establish engineering behavior, not native simulation results.
- Real TRAIN preflight/dry-run read the four existing S0 feedback chains successfully and returned `NOT_RUN`, `planned_native_calls=8`, `simulation_calls=0`. Scenes: `fc90fd5b50ae59fe`, `3d41f95debf352a9`, `13b2b15448c15fc9`, `2e3748d48c5f5139`.
- **No actual simulation was launched by this author.** The Lead dispatcher owns the real eight-call invocation and its resource accounting.

## Actual dispatch command

After integrating this tool into the immutable dispatcher snapshot, use its source directory. Existing source/config and source G0 paths are real; the new output below is this tool's proposed independent artifact namespace.

```bash
cd /mnt/project/VLA-Drive-action-effect-repair-20261020
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  -m tools.action_effect.audit_navsim_feedback \
  --config configs/action_effect/main.yaml \
  --g0-root /mnt/project/action-effect-repair-20261020/g0/population_000064 \
  --output /mnt/project/action-effect-repair-20261020/feedback_audits/native_replay4_v1 \
  --max-samples 4 --repeats 2 --max-simulation-budget 16 --retry-budget 8 \
  --max-seconds 600
```

Use the identical command plus `--resume` after interruption; do not increase budgets or change source/config inside the same output. Add `--dry-run` to verify source chains without admitting any call. CPU affinity may be assigned by the dispatcher using its approved idle cores. The tool has no multiworker or GPU mode.

The audit does not claim to establish responsive NAVSIM v2 validity, open-loop causal credit, unseen-log generalization or true physical-clearance measurements. Its backend is the explicitly verified, fixed-reference NAVSIM v1.1 protocol.
