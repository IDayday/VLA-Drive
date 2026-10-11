# Closeout fixes from 924b003

The fixed starting commit is `924b003ad84ffc2d51e1cb4f5e140663f3280152`; the remote readiness branch matched it at intake. Work occurs in a separate worktree. The original Base and the navtest observer worktrees have not been edited.

| Issue | Trigger and reproduction | Fix | Behavioral evidence |
|---|---|---|---|
| C1 qualification history collision | Fail qualification, then qualify another checkpoint under the same output root | `iqe/base_qualification.py::qualify_base`: immutable content-addressed attempts; separate LATEST and explicitly SELECTED pointers. `--select` is intentional. Legacy files remain read-only. Formal lock rejects replacement. | `test_closeout_qualification.py`: failed→qualified, idempotence, later failure, immutable lock, explicit path, legacy failure and changed evidence |
| C2 failures cached forever | One official scoring call times out; rerun finds EVALUATION/scene JSON | `base_evaluation_cache.py`, `evaluate_base`: immutable predictions, per-call bounded scoring attempts, verified success receipts, retry missing labels only. Partial CLI exits nonzero; invalid predictions are terminal quality failures, never reward zero. Legacy attempts migrate without deletion. | Retry call counts, no VLM reload, subset denominators, budget exhaustion, changed contexts/checkpoints/protocols, invalid prediction, report/receipt crash window |
| C3 one extra optimizer step after terminal checkpoint | Early stop saves a checkpoint, process fails before result.json | `training/checkpoint.py::termination_state`, `training/trainer.py`: explicit reason/terminal/step/patience/best snapshot; reliable legacy inference; terminal resume restores best and finalizes without fetching a batch. Rank0 I/O failures propagate to all ranks. | Fault injection for early stop/max steps, both modern and legacy; parameter/RNG/sampler equality; resumable pause; two-rank Gloo |

No Base forward, normalization, input, auxiliary loss or protected architecture file was changed by C1–C3. Strict state_dict loading and existing source hash checks remain. R2 initialization, R4 data transactions, R5 workflow and R6 residual_v2 are regression-tested, not redesigned.

The old draft quality rules remain untouched. New prospective engineering qualification is explicitly scoped to an incremental experiment; it does not assert convergence, statistical benefit or permission to deploy. The currently known 14k/20k navtest observations are disclosed and do not supply training cases or tuned thresholds.
