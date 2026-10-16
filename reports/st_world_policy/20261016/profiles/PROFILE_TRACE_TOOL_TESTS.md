# Measurement-tool behavior tests

This is a summary of the observed test output, not a raw pytest log.

2026-10-16; source commit 2732dcc4ef6fafcbfc27b6e1ef35dfc27b40f6fa
Command: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q tests/st_world/test_profile_pipeline.py tests/st_world/test_profile_trace.py
..................... [100%]
21 passed, 1 pynvml deprecation warning in 2.30s
Behavior: real interval unions/overlap, separate devices, nonadditive NCCL accounting, invalid timestamps; ZeRO partition gradient state, tied-parameter deduplication, masters distinguished from Adam states; prior full-window/rank/topology tests retained.
No GPU timing result or formal launch approval is inferred from unit tests.


## Actual Kineto category regression

The real trace contained both CPU `user_annotation` and `gpu_user_annotation` for the same module. Commit `0c54ed2c8eb3dd14762ca5aa88c8c911d541f626` separates them and derives logical collective input bytes only from complete shape/dtype metadata. Same CPU command with both profile test files: **22 passed, one existing pynvml warning,2.07s**. Complete unabridged stdout/stderr: `/mnt/project/st-world-policy-20261016/audits/systems/tests/profile_trace_category_regression.log`. This is behavioral tool validation, not model numerical approval.
