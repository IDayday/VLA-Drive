# Current execution evidence

Environment: Python3.10.20, PyTorch2.5.1+cu124, CUDA12.4, Transformers4.57,
A80080GB. The current full regression completed **294 passed,3 skipped** in
51.35s, with four dependency deprecation warnings. Full stdout is preserved in
final_regression_current.log. The skips are explicit two-GPU opt-ins; they are
not counted as GPU execution.

```bash
cd /mnt/project/DriveVLA-M0-iqe-internal-scorer
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m pytest -q tests/iqe

# The following ran on verified idle GPUs earlier in this experiment.
# Do not rerun now over the restored Base/cache jobs.
IQE_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q tests/iqe/test_internal_ddp.py
IQE_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/iqe/test_training_resume_ddp.py::test_nccl_uneven_labels_and_exact_resume
IQE_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/iqe/test_closeout_terminal.py::test_nccl_terminal_resume_all_ranks
```

Internal Gloo+NCCL objective tests:2 passed. Existing NCCL uneven-label/exact
resume:1 passed. Terminal NCCL also passed in nccl_regression_released.log.
An earlier attempted GPU check failed while a captured pressure filler owned
resources; nccl_regression.log remains available rather than being deleted.
Subsequent passes used actually freed authorized GPUs.

Real evidence is separate:23,744 current-input feature records;1,000 expert
updates;2,000 scratch-scorer updates;47,488 actual official candidate labels;
fixed dev evaluation;32-image no-label-leak/freeze/call-count audit;64-real-pair
bounded fitting diagnostic; deterministic4 versus2+2 resume;8 newly repeated
selected official scores exactly equal their bank labels; and30 real-image
deployment timing samples. No mock or tensor test substitutes for these runs.

Scientific outcome is negative/inconclusive for the requested held-out benefit,
as detailed in RESULTS.md and VERDICT.json. Passing engineering tests does not
change that conclusion.
