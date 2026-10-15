# Current regression and real recovery evidence

Actual suite on implementation `048c4b7024690ec63848758f7bbda00182f4026a`:
**602 passed, 3 skipped**, 70.98 seconds, four dependency warnings.
The complete log is `evidence/regression_release.log`; the process command,
exit code and elapsed wall time are `evidence/REGRESSION_RELEASE.json`.
The three skipped tests require explicitly admitted GPUs; they were executed
separately below, not silently counted as CPU passes.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q -ra tests/iqe
```

Three real two-GPU NCCL checks passed on `training-vla-zt-worker-0`, A80080GB:

- Terminal checkpoint recovery on both ranks: PASS, no new optimizer batch.
- Two-flow global objective with uneven valid labels: PASS.
- Uneven-label gradients plus optimizer-boundary exact recovery: PASS.

The initial consecutive GPU invocation passed terminal recovery, then its two
remaining resource-admission checks rejected the preceding test's lingering
13% utilization sample despite zero allocated memory. These are preserved in
`regression_nccl.log` as **2 blocked resource-admission attempts**, not hidden.
Freshly verified idle GPU pairs2/3 and4/5 ran the two remaining tests successfully;
no policy check was disabled, no other process was stopped, and no GPU reset was
used. Individual retry logs and return codes are included. Commands:

```bash
IQE_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/iqe/test_closeout_terminal.py::test_nccl_terminal_resume_all_ranks
IQE_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/iqe/test_internal_ddp.py::test_global_gap_stream_objective_two_process_nccl
IQE_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/iqe/test_training_resume_ddp.py::test_nccl_uneven_labels_and_exact_resume
```

The new cost-CE itself has a two-process Gloo proof with unequal rank counts
and a fully tied rank. The NCCL checks above exercise the reused native trainer
and reduction/recovery infrastructure; they are not falsely described as
multi-GPU training of the two new production heads, which used single GPUs.

Both new models also passed **real cached-context CUDA recovery** with32 actual
legal scenes. Each compared a continuous four-update run with two updates plus
fresh construction/recovery/two updates. Exact equality held for module,
Adam, scheduler, sampler, consumed IDs, next IDs, Python/NumPy/CPU/CUDA RNG,
loss sequence and gradient-norm sequence. Reloading the terminal checkpoint
consumed zero samples and made zero updates. Eight diagnostic updates per model
were executed in total; these weights never initialized main training.
`evidence/RESUME_M_scene.json` and `RESUME_M_set.json` retain actual receipts.

Initial real-image integration passed on four requested scenes replayed in
their four actual source batches (16 scene records across those batches, each with three camera views).
All W/A, raw/physical and head-output differences were zero for this check.
Trained-head integration and fresh-process bundle evidence are separately
archived in `INFERENCE_CALL_AUDIT.md`; initial-head success is not substituted
for final trained deployment verification.

Main tests cover data/role/hash rejection, fixed last refit, scratch provenance,
fit-only normalization, mask/tie/K1/permutation behavior, gradients stopping at
W/A/candidates, true prerouting and mixed-batch order restoration, invalid
Decoder fallback, bundle schema/source checks and current-state future removal.
The Base/source framework and existing protected files were not modified.

One local scheduling-only correction replaced an unstarted dependency-waiting
controller so parallel fresh-process checks use thread-specific atomic status
temporary files. The stopped process had no child job or training state;
receipt `evidence/WAITING_CONTROLLER_FIX.json` preserves its identity. No Base,
G2, old experiment or unrelated process was signaled. Scientific configurations,
main checkpoints and training source hashes were unchanged.
