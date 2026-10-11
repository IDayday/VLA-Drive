# C3 — terminal checkpoints survive missing result.json

PASS. `training/checkpoint.py::termination_state` records MAX_STEPS, EARLY_STOPPED, OPERATOR_PAUSED or RUNNING_CHECKPOINT, terminal flag, stopped step, patience and historical best identity. Reliable legacy step/patience fields are interpreted explicitly.

`training/trainer.py` resolves/restores the complete checkpoint first. A terminal checkpoint restores its historical best and writes missing result/receipt with no batch fetch, optimizer step or training RNG advance. A pause/normal checkpoint resumes. Rank0 checkpoint/result failures propagate collectively; all ranks agree on termination. `result.json.COMPLETE.json` closes the result/receipt crash window without changing state.

Fault injection in `test_closeout_terminal.py`: checkpoint succeeds then result write fails, for early stopping and max steps, modern and legacy metadata. Restores compare parameters, sampler, RNG and best; fetch is forbidden. Separate operator-pause tests prove continuation. Two-process Gloo and two-GPU NCCL both pass. NCCL used free GPUs before another workload arrived; no original Base resources were occupied.

Real actual-22k expert: two continuous updates equal pause-after-one plus resume in sample IDs, loss, gradient norm, parameters, Adam and scheduler. Restoring its MAX_STEPS checkpoint into a result-less output produced zero updates and zero sample advancement. Evidence `REAL_RESUME.json`; actual checkpoint `.../incremental_qualification_gpu/rounds/round_001/expert/step_000002.pt`. CPU/Gloo/NCCL fault injection establishes early-stop behavior; the real-data control establishes ordinary and max-step resume. Neither is claimed to be a new scientific result.
