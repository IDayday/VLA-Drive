# Bounded common-coordinate diagnostic v2

Preserve immutable v1 failure. Same eight preselected TRAIN observations, same checkpoints, same requests and original FP32/no-TF32 model path. No training and no new feature cache.

The primary exception is now retained with traceback. Strict successful model release remains mandatory before loading the second model. Fresh CUDA process requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`, supplied by the controller before child Python startup; conflicting inherited values are rejected.

CPU regression: 9 diagnostic and 8 control tests pass. Actual A800 tiny CUDA matmul without workspace raises the documented deterministic-algorithm error; with the fixed setting it returns exact expected values. This does not recover or prove the original heavy forward's masked exception.

Lead may run `bash launch_once.sh --admission <actual reviewed v2 admission> --admission-sha256 <actual hash>` only after independent correctness/resource review of these pins. Existing 1800-second single-GPU bound, 320 GiB logical-read stop threshold, 96 GiB RAM and 16 MiB output reservation remain. New output roots preserve all failed v1 evidence. No automatic retry.
